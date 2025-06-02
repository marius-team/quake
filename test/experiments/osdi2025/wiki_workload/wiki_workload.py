import json
import time
from pathlib import Path
from typing import Union, Dict, List, Optional

import numpy as np
import torch

import quake
from quake.utils import to_path, compute_recall
import pandas as pd
import matplotlib.pyplot as plt
from quake.index_wrappers.quake import QuakeWrapper

TOTAL_NUM_UPDATES = 104
TOTAL_NUM_EMBEDDINGS = 11_790_930
K = 100

wikidata_dir = Path("/Users/jason/projects/compass/data/wikidata")
metric = "ip"

def get_node_mappings() -> dict:
    """
    This function retrieves the node mappings from the S3 bucket.
    """
    node_mappings = None

    with open(wikidata_dir / "node_mappings.json", "r") as f:
        node_mappings = json.load(f)

    return node_mappings


def update_id_to_month(n) -> str:
    """
    This function converts the update id to the corresponding month and year.
    """
    # starting point: August 2015 corresponds to update 27
    start_month = 8
    start_year = 2015

    # calculate the number of months to add
    months_to_add = n - 27

    # calculate the target year and month
    target_year = start_year + (start_month + months_to_add - 1) // 12
    target_month = (start_month + months_to_add - 1) % 12 + 1

    # get the month and year as a string
    return f"{target_year}{target_month:02d}"


def compute_ip_ground_truth(
        queries: torch.Tensor,
        base   : torch.Tensor,
        # ids    : torch.Tensor,
        k: int,
) -> np.ndarray:


    flat_quake_index = quake.QuakeIndex()
    build_params = quake.IndexBuildParams()
    build_params.metric = "ip"
    build_params.nlist = 1
    build_params.num_workers = 6
    ids = torch.arange(base.shape[0], dtype=torch.int64)
    flat_quake_index.build(base, ids, build_params)

    search_params = quake.SearchParams()
    search_params.k = k
    search_params.batched_scan = True
    res = flat_quake_index.search(queries, search_params)
    return res.ids.numpy(), res.distances.numpy()


def wiki_sampler(current_ids: torch.tensor, query_size: int, update_id: int) -> torch.tensor:
    """
    This function samples query ids from the current_ids.

    :param current_ids: The current ids in the index.
    :param query_size: The size of the query.
    :param update_id: The id of the update.

    :return: The sampled query ids.
    """
    assert update_id >= -1 and update_id <= 103

    month_str = update_id_to_month(update_id)
    if month_str == "201412":
        month_str = "201312"
    elif month_str == "201310":
        month_str = "201210"

    # for each of the current ids, get the corresponding pageviews for the given update_id
    df_id = pd.DataFrame(current_ids.numpy(), columns=["id"])
    df_pageviews = pd.read_parquet(wikidata_dir / f"pageviews/pageviews_{month_str}.parquet")
    # df_pageviews = pd.read_parquet(Path(f"data/wikidata/pageviews/pageviews_{month_str}.parquet"))

    df = pd.merge(df_id, df_pageviews, on="id", how="left")
    df["pageviews"] = df["pageviews"].fillna(0).astype(int)

    pageviews = df["pageviews"].values
    assert len(pageviews) == len(current_ids)

    # sample the query ids based on the pageviews
    np.random.seed(42)
    probs = pageviews / pageviews.sum()
    query_ids = np.random.choice(current_ids.numpy(), size=query_size, replace=True, p=probs)

    assert len(query_ids) == query_size
    return torch.tensor(query_ids)

class WikidataWorkloadGenerator:
    """
    Generates a dynamic workload from the Wikipedia data for updates
    and sample embeddings for queries (for now).

    The workflow is as follows:
    1. Takes the embeddings from the Wikipedia dataset for the initial base embeddings and updates.
    2. After each update, samples a set of resident embeddings for queries.
    3. The workload is saved to a directory as a set of operations with a corresponding runbook.
    """

    def __init__(
            self,
            workload_dir: Union[str, Path],
            num_updates: int,
            query_batch_size: int,
            metric: str = "ip",
            seed: int = 1874,
            with_patterns: bool = True,
            overwrite: bool = False,
    ):
        # download the wikipedia dataset if not already downloaded
        self.num_updates = num_updates

        assert num_updates <= TOTAL_NUM_UPDATES

        self.workload_dir = Path(workload_dir)
        self.query_batch_size = query_batch_size
        self.metric = metric.lower()
        self.seed = seed
        self.with_patterns = with_patterns

        assert self.metric == "ip"
        assert self.query_batch_size > 0

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.workload_dir.mkdir(parents=True, exist_ok=True)

        # delete the runbook if it exists and overwrite is True
        runbook_path = self.workload_dir / "runbook.json"
        if runbook_path.exists() and overwrite:
            runbook_path.unlink()

        self.operations_dir = self.workload_dir / "operations"
        if self.operations_dir.exists() and overwrite:
            for file in self.operations_dir.iterdir():
                file.unlink()

        self.operations_dir.mkdir(parents=True, exist_ok=True)

        # set generator state
        self.current_base = None
        self.current_ids = None

        self.runbook = {}

    def workload_exists(self):
        return (self.workload_dir / "runbook.json").exists()

    def generate_workload(self):
        """
        Generate the workload.
        """

        self.runbook["parameters"] = {
            "query_batch_size": self.query_batch_size,
            "num_updates": self.num_updates,
            "metric": self.metric,
            "pattern_sampling": self.with_patterns,
        }

        initial_embeddings = torch.from_numpy(np.load(wikidata_dir / "wikidata" / "embeddings_initial.npy"))
        initial_embeddings /= torch.norm(initial_embeddings, dim=1, keepdim=True)
        self.current_base = initial_embeddings
        self.current_ids = torch.from_numpy(np.load(wikidata_dir / "wikidata" / "ids_initial.npy"))

        self.runbook["initialize"] = {
            "size": self.current_ids.shape[0],
        }
        self.runbook["operations"] = {}

        n_inserts = 0
        n_queries = 0
        n_operations = 0

        # the first operation is a query operation
        n_resident = self.current_ids.shape[0]
        assert n_resident == self.current_base.shape[0]

        if not self.with_patterns:
            rand_perm = torch.randperm(n_resident)[: self.query_batch_size]
            sample_ids = self.current_ids[rand_perm]
            queries = self.current_base[rand_perm]
        else:
            sample_ids = wiki_sampler(self.current_ids, self.query_batch_size, -1)
            indices = torch.tensor([torch.where(self.current_ids == s_id)[0][0] for s_id in sample_ids])
            queries = self.current_base[indices]

        entry = {"type": "query", "sample_size": len(sample_ids), "n_resident": n_resident}
        torch.save(sample_ids, self.operations_dir / "0.pt")

        # compute the ground truth based on the resident set of vectors
        start_time = time.time()
        ids, dists = compute_ip_ground_truth(queries, self.current_base, K)
        ids = self.current_ids[ids]
        gt_time = time.time() - start_time

        entry["gt_time"] = gt_time

        # save the ground truth
        torch.save(ids, self.operations_dir / "0_gt_ids.pt")
        torch.save(dists, self.operations_dir / "0_gt_dists.pt")

        print("Operation 0", entry)
        self.runbook["operations"][0] = entry

        n_queries += 1
        n_operations += 1

        # for each of the updates, do one insert and one query operation
        for i in range(self.num_updates):
            print(f"Processing update {i}...")
            new_embeddings = torch.from_numpy(np.load(wikidata_dir / "wikidata" / f"embeddings_update_{i}.npy"))
            new_embeddings /= torch.norm(new_embeddings, dim=1, keepdim=True)
            new_ids = torch.from_numpy(np.load(wikidata_dir / "wikidata" / f"ids_update_{i}.npy"))

            # insert the new embeddings
            self.current_base = torch.cat([self.current_base, new_embeddings], dim=0)
            self.current_ids = torch.cat([self.current_ids, new_ids], dim=0)

            n_resident = self.current_ids.shape[0]
            assert n_resident == self.current_base.shape[0]

            entry = {"type": "insert", "size": new_ids.shape[0], "n_resident": n_resident}
            # torch.save(new_ids, self.operations_dir / f"{n_operations}.pt")

            print(f"Operation {n_operations} (insert)", entry)
            self.runbook["operations"][n_operations] = entry

            n_operations += 1
            n_inserts += 1

            # the next operation is a query operation
            if not self.with_patterns:
                # randomly sample a set of resident embeddings for queries
                rand_perm = torch.randperm(n_resident)[: self.query_batch_size]
                sample_ids = self.current_ids[rand_perm]
                queries = self.current_base[rand_perm]
            else:
                sample_ids = wiki_sampler(self.current_ids, self.query_batch_size, i)
                indices = torch.tensor([torch.where(self.current_ids == s_id)[0][0] for s_id in sample_ids])
                queries = self.current_base[indices]

            entry = {"type": "query", "sample_size": len(sample_ids), "n_resident": n_resident}
            torch.save(sample_ids, self.operations_dir / f"{n_operations}.pt")

            # compute the ground truth based on the resident set of vectors
            start_time = time.time()
            ids, dists = compute_ip_ground_truth(queries, self.current_base, K)
            ids = self.current_ids[ids]
            gt_time = time.time() - start_time

            entry["gt_time"] = gt_time

            # save the ground truth
            torch.save(ids, self.operations_dir / f"{n_operations}_gt_ids.pt")
            torch.save(dists, self.operations_dir / f"{n_operations}_gt_dists.pt")

            print(f"Operation {n_operations} (query)", entry)
            self.runbook["operations"][n_operations] = entry

            n_operations += 1
            n_queries += 1

        self.runbook["summary"] = {
            "n_inserts": n_inserts,
            "n_queries": n_queries,
            "n_operations": n_operations,
        }

        # save the runbook
        with open(self.workload_dir / "runbook.json", "w") as f:
            json.dump(self.runbook, f, indent=4)


class WikidataWorkloadEvaluator:
    """
    Evaluates a Wikidata workload using the same interface as WorkloadEvaluator,
    but with explicit NumPy-based embedding handling.
    """

    def __init__(
            self,
            workload_dir: Union[str, Path],
            output_dir: Union[str, Path],
            wiki_dataset_dir:  Union[str, Path],
    ):
        self.workload_dir = to_path(workload_dir)
        self.wiki_dataset_dir = to_path(wiki_dataset_dir)
        self.output_dir = to_path(output_dir)
        self.runbook_path = self.workload_dir / "runbook.json"
        self.ops_dir = self.workload_dir / "operations"

        # will hold embeddings and ids
        self.current_base: Optional[torch.Tensor] = None
        self.current_ids: Optional[torch.Tensor] = None
        self.index_initialized = False

    def _init_index(self, name: str, wrapper, build_params: Dict,
                    m_params: Optional[Dict]):

        idx_dir  = self.workload_dir / "init_indexes"
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_file = idx_dir / f"{name}.index"

        vecs_init      = torch.from_numpy(np.load(self.wiki_dataset_dir / "embeddings_initial.npy"))
        init_ids  = torch.from_numpy(np.load(self.wiki_dataset_dir / "ids_initial.npy"))

        self.current_base = vecs_init
        self.current_ids = init_ids

        max_id = int(self.current_ids.max().item())
        inv_map = torch.full((max_id + 1,), -1, dtype=torch.long)
        inv_map[self.current_ids] = torch.arange(len(self.current_ids), dtype=torch.long)
        self.inv_map = inv_map

        if not idx_file.exists():
            print(f"[{name}] building base index …")
            wrapper.build(vecs_init, ids=init_ids, **build_params)
            wrapper.save(idx_file)
            print(f"[{name}] stored → {idx_file}")
        else:
            wrapper.load(idx_file,
                         num_workers=build_params.get("num_workers", 0), use_numa=build_params.get("use_numa", True), parent=build_params.get("parent", None))
            print(f"[{name}] loaded ← {idx_file}")

        if isinstance(wrapper, QuakeWrapper) and m_params:
            mp = quake.MaintenancePolicyParams()
            for k, v in m_params.items():
                setattr(mp, k, v)
            wrapper.index.initialize_maintenance_policy(mp)
            print(f"[{name}] maintenance policy: {m_params}")



        return wrapper

    def evaluate_workload(
            self,
            *,
            name: str,
            index,
            build_params: Dict,
            search_params: Dict,
            do_maintenance: bool = False,
            m_params: Optional[Dict] = None,
            batch: bool = False,
            max_q: int = 1000,
    ) -> List[Dict]:
        """
        Evaluate inserts, deletes, and queries by loading NumPy embeddings directly.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = self._init_index(name, index, build_params, m_params)

        runbook = json.load(open(self.runbook_path))
        results: List[Dict] = []
        totals = dict(query=0.0, insert=0.0, delete=0.0, maintenance=0.0)

        print(f"─ Evaluating workload on {name} ({len(runbook['operations'])} ops) ─")
        for op_id, op in runbook["operations"].items():
            op_no = int(op_id)
            typ = op["type"]
            print(f"[{name}] op {op_no:4d} | {typ:<6}", end="", flush=True)

            if typ == "insert":
                upd_idx = (op_no - 1) // 2
                emb_up = np.load(self.wiki_dataset_dir / f"embeddings_update_{upd_idx}.npy")
                id_up = np.load(self.wiki_dataset_dir / f"ids_update_{upd_idx}.npy")
                new_emb = torch.from_numpy(emb_up).float()
                new_emb /= new_emb.norm(dim=1, keepdim=True)
                new_ids = torch.from_numpy(id_up).long()

                t0 = time.perf_counter()
                index.add(new_emb, ids=new_ids)
                latency_ms = (time.perf_counter() - t0) * 1e3
                recall = None

                # update current state
                old_len = len(self.current_ids)
                self.current_base = torch.cat([self.current_base, new_emb], dim=0)
                self.current_ids = torch.cat([self.current_ids, new_ids], dim=0)

                max_new_id = int(self.current_ids.max().item())
                if max_new_id >= self.inv_map.size(0):
                    # need a larger map
                    new_map = torch.full((max_new_id + 1,), -1, dtype=torch.long)
                    new_map[: self.inv_map.size(0)] = self.inv_map
                    self.inv_map = new_map

                # positions of the just‐inserted IDs
                new_positions = torch.arange(old_len, old_len + len(new_ids), dtype=torch.long)
                self.inv_map[new_ids] = new_positions

            elif typ == "delete":
                # assume ops_dir contains .pt of ids to delete
                ids = torch.load(self.ops_dir / f"{op_id}.pt")
                t0 = time.perf_counter()
                index.remove(ids)
                latency_ms = (time.perf_counter() - t0) * 1e3
                recall = None

            elif typ == "query":
                # load sample IDs and map
                samp_ids = torch.load(self.ops_dir / f"{op_id}.pt", weights_only=True)
                indices = self.inv_map[samp_ids]
                queries = self.current_base[indices]
                queries /= queries.norm(dim=1, keepdim=True)

                t0 = time.perf_counter()

                randperm = torch.randperm(len(queries))
                queries = queries[randperm[:max_q]]
                t_infos = []
                if batch:
                    sr = index.search(queries, **search_params)
                    pred_ids = sr.ids
                    t_infos.append(sr.timing_info)
                else:
                    parts = []
                    for q in queries:
                        out = index.search(q.unsqueeze(0), **search_params)
                        parts.append(out.ids)
                        t_infos.append(out.timing_info)
                    pred_ids = torch.cat(parts)
                latency_ms = (time.perf_counter() - t0) * 1e3

                gt_ids = torch.load(self.ops_dir / f"{op_id}_gt_ids.pt")[randperm[:max_q]]
                recall = compute_recall(pred_ids, gt_ids, search_params.get("k")).mean().item()
                op["recall"] = recall

                total_parent_time = 0
                total_time = 0
                total_boundary_time = 0
                total_aps_time = 0

                total_buffer_init_time = 0
                total_copy_query_time = 0
                total_job_enqueue_time = 0
                total_job_wait_time = 0
                total_result_aggregate_time = 0

#                 res->timing_info->buffer_init_time_ns =
#         duration_cast<nanoseconds>(s2 - s1).count();
# res->timing_info->copy_query_time_ns =
# duration_cast<nanoseconds>(s3 - s2).count();
# res->timing_info->job_enqueue_time_ns =
# duration_cast<nanoseconds>(s4 - s3).count();
# res->timing_info->job_wait_time_ns =
# duration_cast<nanoseconds>(s5 - s4).count();
# res->timing_info->result_aggregate_time_ns =
# duration_cast<nanoseconds>(s6 - s5).count();

                for t_info in t_infos:
                    total_parent_time += t_info.parent_info.total_time_ns / 1e6
                    total_time += t_info.total_time_ns / 1e6
                    total_boundary_time += t_info.boundary_distance_time_ns / 1e6
                    total_aps_time += t_info.aps_time_ns / 1e6
                    total_buffer_init_time += t_info.buffer_init_time_ns / 1e6
                    total_copy_query_time += t_info.copy_query_time_ns / 1e6
                    total_job_enqueue_time += t_info.job_enqueue_time_ns / 1e6
                    total_job_wait_time += t_info.job_wait_time_ns / 1e6
                    total_result_aggregate_time += t_info.result_aggregate_time_ns / 1e6



                print(f" | parent {total_parent_time:.2f} ms"
                      f" | total {total_time:.2f} ms"
                      f" | boundary {total_boundary_time:.2f} ms"
                      f" | aps {total_aps_time:.2f} ms" 
                        f" | buffer init {total_buffer_init_time:.2f} ms"
                        f" | copy query {total_copy_query_time:.2f} ms"
                        f" | job enqueue {total_job_enqueue_time:.2f} ms"
                        f" | job wait {total_job_wait_time:.2f} ms"
                        f" | result aggregate {total_result_aggregate_time:.2f} ms")


            else:
                raise ValueError(f"Unknown op type {typ}")

            n_splits = 0
            n_deletes = 0
            split_time_ms = 0.0
            delete_time_ms = 0.0
            refinement_time_ms = 0.0
            maintenance_latency_ms = 0.0

            if do_maintenance:
                t0 = time.perf_counter()
                nlist_before = index.index_state()["n_list"]
                info = index.maintenance()
                nlist_after = index.index_state()["n_list"]
                maintenance_latency_ms = (time.perf_counter() - t0) * 1e3
                n_splits = info.n_splits
                n_deletes = info.n_deletes
                delete_time_ms = info.delete_time_us / 1000.0
                split_time_ms = info.split_time_us / 1000.0
                refinement_time_ms = info.refinement_time_us / 1000.0




            n_resident = index.index_state()["n_total"]
            nlist = index.index_state()["n_list"]

            totals[typ] += latency_ms
            totals["maintenance"] += maintenance_latency_ms
            print(f" | lat {latency_ms:8.2f} ms" + (f" | rec {recall:.3f}" if recall is not None else ""))

            row = {
                "operation_number": op_no,
                "operation_type": typ,
                "latency_ms": latency_ms,
                "n_resident": n_resident,
                "n_splits": n_splits,
                "n_deletes": n_deletes,
                "nlist": nlist,
                "split_time_ms": split_time_ms,
                "delete_time_ms": delete_time_ms,
                "maintenance_latency_ms": maintenance_latency_ms,
                "refinement_time_ms": refinement_time_ms,
                "recall": recall,
            }
            print(row)

            results.append(row)

        # four-panel
        df = pd.DataFrame(results)
        self._four_panel_plot(df)

        # time breakdown
        self._time_breakdown_plot(totals, name)

        # CSV output
        df.to_csv(self.output_dir / "results.csv", index=False)
        print(f"Results → {self.output_dir / 'results.csv'}")
        return results

    # ---------------------------------------------------------------------
    def _four_panel_plot(self, df: pd.DataFrame):
        lat_ins = df[df.operation_type == "insert"]
        lat_del = df[df.operation_type == "delete"]
        lat_q = df[df.operation_type == "query"]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax = axs[0, 0]
        if not lat_ins.empty:
            ax.plot(lat_ins.operation_number, lat_ins.latency_ms, marker="o", label="Insert")
        if not lat_del.empty:
            ax.plot(lat_del.operation_number, lat_del.latency_ms, marker="s", label="Delete")
        if not lat_q.empty:
            ax.plot(lat_q.operation_number, lat_q.latency_ms, marker="^", label="Query")
        ax.set(xlabel="Op #", ylabel="Latency (ms)", title="Latency"); ax.legend()
        # other panels blank except recall
        ax = axs[0, 1]
        ax.text(0.5,0.5,"Partitions not tracked",ha="center")
        ax.axis("off")
        ax = axs[1, 0]
        ax.text(0.5,0.5,"Resident not tracked",ha="center")
        ax.axis("off")
        rec = df[(df.operation_type=="query") & df.recall.notna()]
        ax = axs[1, 1]
        if not rec.empty:
            ax.plot(rec.operation_number, rec.recall, marker="o")
            ax.set(xlabel="Op #", ylabel="Recall", title="Recall")
        else:
            ax.text(0.5,0.5,"No recall",ha="center")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_plots.png")
        plt.close()

    # ---------------------------------------------------------------------
    def _time_breakdown_plot(self, totals: Dict[str, float], title: str):
        plt.figure(figsize=(6,4))
        bars = [totals.get("query",0), totals.get("insert",0), totals.get("delete",0), sum(totals.values())]
        labels = ["Query","Insert","Delete","Total"]
        plt.bar(labels, bars)
        plt.ylabel("Cumulative ms")
        plt.title(f"Time breakdown – {title}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_breakdown.png", dpi=150)
        plt.close()


if __name__ == "__main__":

    # parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate a dynamic workload for Wikidata.")
    parser.add_argument(
        "--workload_dir",
        type=str,
        default="data/wikidata/workload",
        help="Directory to save the workload.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=1000,
        help="Batch size for queries between updates.",
    )

    # Initialize the workload generator
    args = parser.parse_args()
    workload_generator = WikidataWorkloadGenerator(
        workload_dir=args.workload_dir,
        num_updates=TOTAL_NUM_UPDATES,
        query_batch_size=args.query_batch_size,
        metric=metric,
        with_patterns=True,
        overwrite=True,
    )
    # Generate the workload
    workload_generator.generate_workload()