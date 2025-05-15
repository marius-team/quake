import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quake
import torch

from quake import SearchParams
from quake.index_wrappers.quake import QuakeWrapper
from quake.utils import compute_recall, knn, to_path


def run_query(
    index,
    queries: torch.Tensor,
    search_k: int,
    search_params: dict,
    gt_ids: torch.Tensor,
):
    """
    Run queries on the index and compute the recall.
    """
    # ... (existing implementation)
    recalls = []
    timing_infos = []
    for i, query in enumerate(queries):
        query = query.unsqueeze(0)
        gt_id = gt_ids[i].unsqueeze(0)
        pred_ids, pred_dists, timing_info = index.search(query, **search_params)
        recall = compute_recall(pred_ids, gt_id, search_k).item()
        recalls.append(recall)
        timing_infos.append(timing_info)
    recalls = torch.tensor(recalls)
    return recalls, timing_infos


class VectorSampler(ABC):
    """Abstract class for sampling vectors."""

    @abstractmethod
    def sample(self, size: int):
        """Sample vectors for an operation."""


class UniformSampler(VectorSampler):
    """Uniformly sample vectors."""

    def __init__(self):
        pass

    def sample(self, sample_pool: torch.Tensor, size: int, update_ranks: bool = True):
        randperm = torch.randperm(sample_pool.shape[0])
        sample_ids = sample_pool[randperm[:size]]
        return sample_ids


class StratifiedClusterSampler(VectorSampler):
    """
    Sample vectors from clusters.

    This sampler uses cluster assignments and centroid distances
    to sample in a stratified fashion.
    """

    def __init__(self, assignments: torch.Tensor, centroids: torch.Tensor):
        self.assignments = assignments
        self.centroids = centroids
        self.cluster_size = centroids.shape[0]
        non_empty_clusters = torch.unique(assignments)
        self.root_cluster = non_empty_clusters[torch.randint(0, non_empty_clusters.shape[0], (1,))]
        self.update_ranks(self.root_cluster)

    def update_ranks(self, root_cluster: int):
        print("Updating cluster ranks: root cluster", root_cluster)
        self.root_cluster = root_cluster
        nearest_cluster_ids, _ = knn(self.centroids[root_cluster], self.centroids, -1, "l2")
        self.cluster_ranks = nearest_cluster_ids.flatten()

    def sample(self, sample_pool: torch.Tensor, size: int, update_ranks: bool = True):
        # Get the cluster assignments for all indices in the sample pool.
        sample_assignments = self.assignments[sample_pool]

        # Identify which clusters are present in the sample pool.
        present_clusters = set(sample_assignments.tolist())

        # Filter self.cluster_ranks to only include clusters that are present.
        cluster_order = [c for c in self.cluster_ranks.tolist() if c in present_clusters]

        sampled_indices = []
        num_collected = 0

        # Loop over clusters in the order defined by the filtered ranks.
        for cluster in cluster_order:
            print("Sampling from cluster", cluster)
            # Find indices in sample_pool that belong to this cluster.
            cluster_mask = (sample_assignments == cluster).nonzero(as_tuple=True)[0]
            if cluster_mask.numel() == 0:
                continue

            # Determine how many samples to draw from this cluster.
            n_to_sample = min(size - num_collected, cluster_mask.numel())

            # Randomly sample from the indices in this cluster.
            perm = torch.randperm(cluster_mask.numel())
            chosen = cluster_mask[perm[:n_to_sample]]
            sampled_indices.append(sample_pool[chosen])

            num_collected += n_to_sample
            if num_collected >= size:
                break

        # Concatenate the sampled indices.
        result = torch.cat(sampled_indices) if sampled_indices else torch.tensor([], dtype=torch.long)

        # Optionally update ranks with the last cluster that contributed samples.
        if update_ranks and cluster_order:
            self.update_ranks(cluster_order[1])

        # Remove duplicates, if any.
        result = torch.unique(result)
        return result


class DynamicWorkloadGenerator:
    """
    Generates a dynamic workload from a static vector search dataset.

    Workflow:
      1. Cluster the base vectors.
      2. Initialize the workload with an initial resident set.
      3. Generate operations (insert, delete, query) according to given ratios.
      4. Save each operation and a runbook that includes a summary.
    """

    def __init__(
        self,
        workload_dir: Union[str, Path],
        base_vectors: np.ndarray,
        metric: str,
        insert_ratio: float,
        delete_ratio: float,
        query_ratio: float,
        update_batch_size: int,
        query_batch_size: int,
        number_of_operations: int,
        initial_size: int,
        cluster_size: int,
        cluster_sample_distribution: str,
        queries: np.ndarray,
        query_cluster_sample_distribution: str = "uniform",
        seed: int = 1738,
        initial_clustering_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ):
        # (Initialization code unchanged)
        self.workload_dir = to_path(workload_dir)
        self.base_vectors = base_vectors
        self.metric = metric.lower()
        self.insert_ratio = insert_ratio
        self.delete_ratio = delete_ratio
        self.query_ratio = query_ratio
        self.update_batch_size = update_batch_size
        self.query_batch_size = query_batch_size
        self.number_of_operations = number_of_operations
        self.initial_size = initial_size
        self.cluster_size = cluster_size
        self.cluster_sample_distribution = cluster_sample_distribution
        self.query_cluster_sample_distribution = query_cluster_sample_distribution
        self.queries = queries
        self.seed = seed
        self.initial_clustering_path = to_path(initial_clustering_path) if initial_clustering_path else None
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.validate_parameters()
        self.workload_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir = self.workload_dir / "operations"
        self.operations_dir.mkdir(parents=True, exist_ok=True)
        self.resident_set = torch.zeros(base_vectors.shape[0], dtype=torch.bool)
        self.all_ids = torch.arange(base_vectors.shape[0])
        self.assignments = None
        self.runbook = {}
        self.clustered_index = None
        self.cluster_ranks = None
        self.sampler = None

        self.resident_history = []
        self.query_history = []

    def workload_exists(self):
        return (self.workload_dir / "runbook.json").exists()

    def validate_parameters(self):
        assert self.metric in ["l2", "ip"]
        assert 0 <= self.insert_ratio <= 1
        assert 0 <= self.delete_ratio <= 1
        assert 0 <= self.query_ratio <= 1
        assert self.insert_ratio + self.delete_ratio + self.query_ratio == 1
        assert self.update_batch_size > 0
        assert self.query_batch_size > 0
        assert self.number_of_operations > 0
        assert self.initial_size > 0
        assert self.cluster_size > 0
        assert self.cluster_sample_distribution in ["uniform", "skewed", "skewed_fixed"]

    def initialize_clustered_index(self):
        if self.initial_clustering_path is not None:
            index_dir = self.initial_clustering_path
        else:
            index_dir = self.workload_dir / "clustered_index.bin"
        if index_dir.exists():
            index = QuakeWrapper()
            index.load(index_dir)
            n_clusters = index.index.nlist()
        else:
            n_clusters = self.base_vectors.shape[0] // self.cluster_size
            index = QuakeWrapper()
            index.build(
                self.base_vectors, nc=n_clusters, metric=self.metric, ids=torch.arange(self.base_vectors.shape[0])
            )
            index.save(str(self.workload_dir / "clustered_index.bin"))

        search_params = SearchParams()
        search_params.k = 1
        search_params.batched_scan = True
        self.assignments = index.index.parent.search(self.base_vectors, search_params).ids.flatten()

        return index

    def sample(self, size: int, operation_type: str):
        if operation_type == "insert":
            sample_pool = self.all_ids[~self.resident_set]
        elif operation_type == "delete":
            sample_pool = self.all_ids[self.resident_set]
        elif operation_type == "query":
            sample_pool = (
                torch.arange(self.queries.shape[0]) if self.queries is not None else self.all_ids[~self.resident_set]
            )
        else:
            raise ValueError(f"Invalid operation type {operation_type}.")
        if sample_pool.shape[0] == 0:
            return torch.tensor([], dtype=torch.long)
        if operation_type in ["insert", "delete"]:
            sample_ids = self.sampler.sample(sample_pool, size)
        else:
            # update_ranks = (self.query_cluster_sample_distribution != "skewed_fixed")
            update_ranks = True
            sample_ids = self.query_sampler.sample(sample_pool, size, update_ranks=update_ranks)
        return sample_ids

    def initialize_workload(self):
        if self.cluster_sample_distribution in ["skewed", "skewed_fixed"]:
            self.sampler = StratifiedClusterSampler(self.assignments, self.clustered_index.centroids())
        elif self.cluster_sample_distribution == "uniform":
            self.sampler = UniformSampler()
        else:
            raise ValueError(f"Invalid cluster sample distribution {self.cluster_sample_distribution}.")
        if self.query_cluster_sample_distribution in ["skewed", "skewed_fixed"]:
            query_assignments = knn(self.queries, self.clustered_index.centroids(), 1, "l2")[0].flatten()
            self.query_sampler = StratifiedClusterSampler(query_assignments, self.clustered_index.centroids())
        elif self.query_cluster_sample_distribution == "uniform":
            self.query_sampler = UniformSampler()
        else:
            raise ValueError(f"Invalid query cluster sample distribution {self.query_cluster_sample_distribution}.")
        initial_indices = self.sample(self.initial_size, operation_type="insert")
        self.resident_set[initial_indices] = True
        torch.save(initial_indices, self.workload_dir / "initial_indices.pt")
        if self.queries is not None:
            torch.save(self.queries, self.workload_dir / "query_vectors.pt")
        torch.save(self.base_vectors, self.workload_dir / "base_vectors.pt")
        self.runbook["parameters"] = {
            "sample_queries": self.queries is None,
            "n_base_vectors": self.base_vectors.shape[0],
            "vector_dimension": self.base_vectors.shape[1],
            "metric": self.metric,
            "insert_ratio": self.insert_ratio,
            "delete_ratio": self.delete_ratio,
            "query_ratio": self.query_ratio,
            "update_batch_size": self.update_batch_size,
            "query_batch_size": self.query_batch_size,
            "number_of_operations": self.number_of_operations,
            "initial_size": self.initial_size,
            "cluster_size": self.cluster_size,
            "cluster_sample_distribution": self.cluster_sample_distribution,
            "query_cluster_sample_distribution": self.query_cluster_sample_distribution,
            "seed": self.seed,
        }
        self.runbook["initialize"] = {"size": self.initial_size}
        self.runbook["operations"] = {}

    def generate_workload(self):
        """
        Generate the workload and print a summary at the end.
        """
        self.clustered_index = self.initialize_clustered_index()
        self.initialize_workload()
        n_inserts = n_deletes = n_queries = 0
        n_operations = 0

        initial_uniques, initial_counts = torch.unique(self.assignments, return_counts=True)
        all_sizes = torch.zeros(initial_uniques.shape[0])
        all_sizes[initial_uniques] = initial_counts.float()
        for i in range(self.number_of_operations):
            operation_type = np.random.choice(
                ["insert", "delete", "query"], p=[self.insert_ratio, self.delete_ratio, self.query_ratio]
            )
            if operation_type == "insert":
                sample_size = self.update_batch_size
                resident = True
                n_inserts += 1
            elif operation_type == "delete":
                sample_size = self.update_batch_size
                resident = False
                n_deletes += 1
            elif operation_type == "query":
                sample_size = self.query_batch_size
                resident = False
                n_queries += 1
            else:
                raise ValueError(f"Invalid operation type {operation_type}.")
            sample_ids = self.sample(sample_size, operation_type)
            if sample_ids.shape[0] == 0:
                break
            n_operations = i + 1
            if operation_type in ["insert", "delete"]:
                self.resident_set[sample_ids] = resident
            n_resident = self.resident_set.sum().item()
            if n_resident < 5 * self.update_batch_size:
                print(f"Below minimum resident set size: {n_resident}")
                break
            entry = {"type": operation_type, "sample_size": sample_ids.shape[0], "n_resident": n_resident}
            torch.save(sample_ids, self.operations_dir / f"{i}.pt")
            if operation_type == "query":
                queries = self.queries[sample_ids] if self.queries is not None else self.base_vectors[sample_ids]
                start_time = time.time()
                resident_ids = self.all_ids[self.resident_set]
                curr_vectors = self.base_vectors[resident_ids]
                ids, dists = knn(queries, curr_vectors, 100, self.metric)
                ids = resident_ids[ids]
                gt_time = time.time() - start_time
                entry["gt_time"] = gt_time
                torch.save(ids, self.operations_dir / f"{i}_gt_ids.pt")
                torch.save(dists, self.operations_dir / f"{i}_gt_dists.pt")
            print("Operation", i, entry)
            self.runbook["operations"][i] = entry

            # Determine the number of clusters. Assuming clusters are labeled from 0 to max_cluster.
            n_clusters = int(self.assignments.max().item()) + 1
            fractions = np.zeros(n_clusters)

            # get resident assignments
            resident_assignments = self.assignments[self.resident_set]

            uniques, counts = torch.unique(resident_assignments, return_counts=True)
            fractions[uniques] = counts.float() / all_sizes[uniques]
            # Append the vector of fractions for this operation.
            self.resident_history.append(fractions)

        self.runbook["summary"] = {
            "n_inserts": n_inserts,
            "n_deletes": n_deletes,
            "n_queries": n_queries,
            "n_operations": n_operations,
        }

        # Convert the history to a NumPy array with shape (n_clusters, n_operations)
        heatmap_array = np.array(self.resident_history).T
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.imshow(heatmap_array, cmap="viridis", aspect="auto")
        ax.set_xlabel("Operation Number")
        ax.set_ylabel("Cluster ID")
        cbar = fig.colorbar(cax)
        cbar.set_label("Resident Fraction")
        plt.tight_layout()
        plt.savefig(self.workload_dir / "resident_history.png")

        print("\nWorkload Generation Summary:")
        print(f"Total Operations: {n_operations}")
        print(f"Inserts: {n_inserts}, Deletes: {n_deletes}, Queries: {n_queries}")
        print(f"Final resident set size: {self.resident_set.sum().item()}")
        with open(self.workload_dir / "runbook.json", "w") as f:
            json.dump(self.runbook, f, indent=4)
        # ----------------------------------------------------------------------------


class WorkloadEvaluator:
    # ---------------------------------------------------------------------
    def __init__(self,
                 workload_dir: Union[str, Path],
                 output_dir:  Union[str, Path],
                 base_vectors_path: Optional[Union[str, Path]] = None):

        self.workload_dir  = to_path(workload_dir)
        self.output_dir    = to_path(output_dir)
        self.runbook_path  = self.workload_dir / "runbook.json"
        self.ops_dir       = self.workload_dir / "operations"
        self.init_ids_path = self.workload_dir / "initial_indices.pt"
        self.base_vec_path = (to_path(base_vectors_path)
                              if base_vectors_path
                              else self.workload_dir / "base_vectors.pt")

    # ---------------------------------------------------------------------
    def _init_index(self, name: str, wrapper, build_params: Dict,
                    m_params: Optional[Dict]):

        idx_dir  = self.workload_dir / "init_indexes"
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_file = idx_dir / f"{name}.index"

        vecs      = torch.load(self.base_vec_path,  weights_only=True).float()
        init_ids  = torch.load(self.init_ids_path,  weights_only=True).long()
        vecs_init = vecs[init_ids]

        if not idx_file.exists():
            print(f"[{name}] building base index …")
            wrapper.build(vecs_init, ids=init_ids, **build_params)
            wrapper.save(idx_file)
            print(f"[{name}] stored → {idx_file}")
        else:
            wrapper.load(idx_file,
                         num_workers=build_params.get("num_workers", 0))
            print(f"[{name}] loaded ← {idx_file}")

        if isinstance(wrapper, QuakeWrapper) and m_params:
            mp = quake.MaintenancePolicyParams()
            for k, v in m_params.items():
                setattr(mp, k, v)
            wrapper.index.initialize_maintenance_policy(mp)
            print(f"[{name}] maintenance policy: {m_params}")

        return wrapper

    # ---------------------------------------------------------------------
    def evaluate_workload(self, *,
                          name: str,
                          index,
                          build_params: Dict,
                          search_params: Dict,
                          do_maintenance: bool = False,
                          m_params: Optional[Dict] = None,
                          batch: bool = False) -> List[Dict]:

        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = self._init_index(name, index, build_params, m_params)

        runbook   = json.load(open(self.runbook_path))
        base_vecs = torch.load(self.base_vec_path, weights_only=True).float()
        queries   = (base_vecs if runbook["parameters"]["sample_queries"]
                     else torch.load(self.workload_dir / "query_vectors.pt",
                                     weights_only=True).float())

        results = []
        totals  = dict(query=0., insert=0., delete=0., maintain=0.)

        print(f"─ Evaluating workload on {name} ({len(runbook['operations'])} ops) ─")
        for op_id, op in runbook["operations"].items():
            op_no   = int(op_id)
            op_type = op["type"]
            ids     = torch.load(self.ops_dir / f"{op_id}.pt",
                                 weights_only=True)

            print(f"[{name}] op {op_no:4d} | {op_type:<6} | {len(ids):6d} ids", end="", flush=True)

            # ----- perform the operation ----------------------------------
            if op_type == "insert":
                t0 = time.perf_counter()
                index.add(base_vecs[ids], ids=ids)
                latency_ms = (time.perf_counter() - t0) * 1e3
                recall = None
            elif op_type == "delete":
                t0 = time.perf_counter()
                index.remove(ids)
                latency_ms = (time.perf_counter() - t0) * 1e3
                recall = None
            elif op_type == "query":
                qs = queries[ids]
                t0 = time.perf_counter()
                t_infos = []
                if batch:
                    sr = index.search(qs, **search_params)
                    pred_ids = sr.ids
                else:
                    p = []
                    for q in qs:
                        sr = index.search(q.unsqueeze(0), **search_params)
                        p.append(sr.ids)
                        t_infos.append(sr.timing_info)
                    pred_ids = torch.cat(p)
                    # pred_ids = torch.cat([
                    #     index.search(q.unsqueeze(0), **search_params).ids
                    #     for q in qs
                    # ])
                latency_ms = (time.perf_counter() - t0) * 1e3
                gt_ids = torch.load(self.ops_dir / f"{op_id}_gt_ids.pt",
                                    weights_only=True)

                total_parent_time = 0
                total_time = 0
                total_boundary_time = 0
                total_aps_time = 0
                total_scan_time = 0

                for t_info in t_infos:
                    total_parent_time += t_info.parent_info.total_time_ns / 1e6
                    total_time += t_info.total_time_ns / 1e6
                    total_boundary_time += t_info.boundary_distance_time_ns / 1e6
                    total_aps_time += t_info.aps_time_ns / 1e6
                    total_scan_time += t_info.scan_time_ns / 1e6

                print(f" | parent {total_parent_time:.2f} ms"
                      f" | total {total_time:.2f} ms"
                      f" | boundary {total_boundary_time:.2f} ms"
                      f" | aps {total_aps_time:.2f} ms"
                      f" | scan {total_scan_time:.2f} ms")
                recall = compute_recall(pred_ids, gt_ids,
                                        search_params["k"]).mean().item()
                op["recall"] = recall
            else:
                raise ValueError(op_type)

            totals[op_type] += latency_ms

            # ----- maintenance -------------------------------------------
            maint_ms = nsplits = ndeletes = 0
            if do_maintenance:
                mi = index.maintenance()
                maint_ms = mi.total_time_us / 1e3
                nsplits  = mi.n_splits
                ndeletes = mi.n_deletes
                totals["maintain"] += maint_ms

            print(f" | lat {latency_ms:8.2f} ms"
                  f" | maint {maint_ms:7.2f} ms"
                  f" | splits {nsplits:4d}"
                  f" | dels {ndeletes:4d}"
                  f"{' | rec {:.3f}'.format(recall) if recall is not None else ''}")

            # ----- store row ---------------------------------------------
            row = {
                "operation_number"   : op_no,
                "operation_type"     : op_type,
                "latency_ms"         : latency_ms,
                "maintenance_time_ms": maint_ms,
                "n_splits"           : nsplits,
                "n_deletes"          : ndeletes,
                "recall"             : recall,
                "n_resident"         : op.get("n_resident"),
            }
            row.update(index.index_state())   # keeps existing behaviour
            row.update(search_params)         # 〃
            results.append(row)

        # ---- original four-panel figure (unchanged) ---------------------
        df = pd.DataFrame(results)
        self._four_panel_plot(df)

        # ---- cumulative-time bar chart ---------------------------------
        self._time_breakdown_plot(totals, name)

        # ---- CSV --------------------------------------------------------
        df.to_csv(self.output_dir / "results.csv", index=False)
        print(f"Results → {self.output_dir / 'results.csv'}")
        return results

    # ---------------------------------------------------------------------
    def _four_panel_plot(self, df: pd.DataFrame):
        """
        Re-implementation of the original four-panel plot – byte-for-byte
        identical file name (*evaluation_plots.png*).
        """
        lat_insert = df[df.operation_type == "insert"]
        lat_delete = df[df.operation_type == "delete"]
        lat_query  = df[df.operation_type == "query" ]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # A: latency
        ax = axs[0, 0]
        if not lat_insert.empty:
            ax.plot(lat_insert.operation_number, lat_insert.latency_ms,
                    label="Insert", marker="o")
        if not lat_delete.empty:
            ax.plot(lat_delete.operation_number, lat_delete.latency_ms,
                    label="Delete", marker="s")
        if not lat_query.empty:
            ax.plot(lat_query.operation_number,  lat_query.latency_ms,
                    label="Query",  marker="^")
        ax.set(xlabel="Operation Number", ylabel="Latency (ms)",
               title="Operation Latency"); ax.legend()

        # B: partitions
        part = df[df.n_list.notna()]
        ax = axs[0, 1]
        if not part.empty:
            ax.plot(part.operation_number, part.n_list, marker="o")
            ax.set(xlabel="Operation Number", ylabel="Number of Partitions",
                   title="Partitions per Operation")
        else:
            ax.text(0.5, 0.5, "No partition info",
                    ha="center", va="center"); ax.axis("off")

        # C: resident set size
        res = df[df.n_resident.notna()]
        ax = axs[1, 0]
        if not res.empty:
            ax.plot(res.operation_number, res.n_resident, marker="o")
            ax.set(xlabel="Operation Number", ylabel="Resident Vectors",
                   title="Resident Set Size")
        else:
            ax.text(0.5, 0.5, "No resident set info",
                    ha="center", va="center"); ax.axis("off")

        # D: recall
        rec = df[(df.operation_type == "query") & df.recall.notna()]
        ax = axs[1, 1]
        if not rec.empty:
            ax.plot(rec.operation_number, rec.recall, marker="o")
            ax.set(xlabel="Operation Number", ylabel="Query Recall",
                   title="Query Recall")
        else:
            ax.text(0.5, 0.5, "No query recall info",
                    ha="center", va="center"); ax.axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_plots.png")
        plt.close()

    # ---------------------------------------------------------------------
    def _time_breakdown_plot(self, totals: Dict[str, float], title: str):
        plt.figure(figsize=(6, 4))
        bars = [totals["query"], totals["insert"],
                totals["delete"], totals["maintain"],
                sum(totals.values())]
        plt.bar(["Query", "Insert", "Delete", "Maintain", "Total"], bars)
        plt.ylabel("Cumulative time (ms)")
        plt.title(f"Time budget – {title}")
        plt.tight_layout()
        plt.savefig(self.output_dir / "time_breakdown.png", dpi=150)
        plt.close()