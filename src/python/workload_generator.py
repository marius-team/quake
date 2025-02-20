import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from quake.utils import compute_recall, knn, to_path
from quake.index_wrappers.quake import QuakeWrapper
from quake import MaintenancePolicyParams
import hashlib
import matplotlib.pyplot as plt

def get_index_class(index_name):
    if index_name == 'Quake':
        from quake.index_wrappers.quake import QuakeWrapper as IndexClass
    elif index_name == 'HNSW':
        from quake.index_wrappers.faiss_hnsw import FaissHNSW as IndexClass
    elif index_name == 'IVF':
        from quake.index_wrappers.faiss_ivf import FaissIVF as IndexClass
    elif index_name == "DiskANN":
        from quake.index_wrappers.diskann import DiskANNDynamic as IndexClass
    else:
        raise ValueError(f"Unknown index type: {index_name}")
    return IndexClass


# Generate a unique method ID based on index configuration and experiment parameters
def generate_method_id(index_cfg, experiment_params):
    """
    Generate a unique method ID based on index configuration and experiment parameters.
    """
    method_dict = {
        'index_name': index_cfg.name,
        'experiment_params': experiment_params
    }
    method_str = json.dumps(method_dict, sort_keys=True)
    method_id = hashlib.md5(method_str.encode()).hexdigest()
    return method_id


def run_query(
        index,
        queries: torch.Tensor,
        search_k: int,
        search_params: dict,
        gt_ids: torch.Tensor,
        single_query: bool = True,
        tune_nprobe: bool = False,
        nprobes: Optional[torch.Tensor] = None,
):
    """
    Run queries on the index and compute the recall.

    [Docstring shortened for brevity]
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
        self.root_cluster = root_cluster
        nearest_cluster_ids, _ = knn(self.centroids[root_cluster], self.centroids, -1, "l2")
        self.cluster_ranks = nearest_cluster_ids.flatten()

    def sample(self, sample_pool: torch.Tensor, size: int, update_ranks: bool = True):
        sample_assignments = self.assignments[sample_pool]
        non_empty_clusters = torch.unique(sample_assignments)
        mask = torch.zeros(self.cluster_size, dtype=torch.bool)
        mask[non_empty_clusters] = True
        cluster_order = self.cluster_ranks[mask[self.cluster_ranks]]
        cluster_samples = []
        curr_sample_size = 0
        for cluster_id in cluster_order:
            mask = sample_assignments == cluster_id
            cluster_sample_pool = sample_pool[mask]
            if cluster_sample_pool.shape[0] == 0:
                continue
            cluster_sample_size = min(size - curr_sample_size, cluster_sample_pool.shape[0])
            cluster_sample = cluster_sample_pool[torch.randperm(cluster_sample_pool.shape[0])[:cluster_sample_size]]
            cluster_samples.append(cluster_sample)
            curr_sample_size += cluster_sample_size
            if curr_sample_size >= size:
                break
        sample_ids = torch.cat(cluster_samples)
        if update_ranks:
            self.update_ranks(cluster_id)
        sample_ids = torch.unique(sample_ids)
        return sample_ids


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
        assert self.cluster_sample_distribution in ["uniform", "skewed"]

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
            index.build(self.base_vectors,
                        nc=n_clusters,
                        metric=self.metric,
                        ids=torch.arange(self.base_vectors.shape[0]))
            index.save(str(self.workload_dir / "clustered_index.bin"))
        self.assignments = index.cluster_ids()
        print("Cluster assignments shape:", self.assignments.shape)
        return index

    def sample(self, size: int, operation_type: str):
        if operation_type == "insert":
            sample_pool = self.all_ids[~self.resident_set]
        elif operation_type == "delete":
            sample_pool = self.all_ids[self.resident_set]
        elif operation_type == "query":
            sample_pool = torch.arange(self.queries.shape[0]) if self.queries is not None else self.all_ids[~self.resident_set]
        else:
            raise ValueError(f"Invalid operation type {operation_type}.")
        if sample_pool.shape[0] == 0:
            return torch.tensor([], dtype=torch.long)
        if operation_type in ["insert", "delete"]:
            sample_ids = self.sampler.sample(sample_pool, size)
        else:
            update_ranks = (self.query_cluster_sample_distribution != "skewed_fixed")
            sample_ids = self.query_sampler.sample(sample_pool, size, update_ranks=update_ranks)
        return sample_ids

    def initialize_workload(self):
        if self.cluster_sample_distribution == "skewed":
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
        for i in range(self.number_of_operations):
            operation_type = np.random.choice(["insert", "delete", "query"],
                                              p=[self.insert_ratio, self.delete_ratio, self.query_ratio])
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
        self.runbook["summary"] = {
            "n_inserts": n_inserts,
            "n_deletes": n_deletes,
            "n_queries": n_queries,
            "n_operations": n_operations,
        }
        print("\nWorkload Generation Summary:")
        print(f"Total Operations: {n_operations}")
        print(f"Inserts: {n_inserts}, Deletes: {n_deletes}, Queries: {n_queries}")
        print(f"Final resident set size: {self.resident_set.sum().item()}")
        with open(self.workload_dir / "runbook.json", "w") as f:
            json.dump(self.runbook, f, indent=4)


class WorkloadEvaluator:
    """
    Evaluates a generated workload on a given index and produces summary statistics and plots.
    """
    def __init__(
            self,
            workload_dir: Union[str, Path],
            index_cfg: dict,
            output_dir: Union[str, Path],
            base_vectors_path: Optional[Union[str, Path]] = None,
    ):
        self.workload_dir = to_path(workload_dir)
        self.index_cfg = index_cfg
        self.index = None
        self.output_dir = to_path(output_dir)
        self.runbook_path = self.workload_dir / "runbook.json"
        self.operations_dir = self.workload_dir / "operations"
        self.initial_indices_path = self.workload_dir / "initial_indices.pt"
        self.base_vectors_path = to_path(base_vectors_path) if base_vectors_path else self.workload_dir / "base_vectors.pt"
        self.runbook = None

    def build_or_load_index(self, index_cfg):
        index_dir = self.workload_dir / "init_indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_name = index_cfg['name']
        index_path = index_dir / f"{index_name}.index"
        IndexClass = get_index_class(index_name)
        vectors = torch.load(self.base_vectors_path, weights_only=True).to(torch.float32)
        initial_indices = torch.load(self.initial_indices_path, weights_only=True).to(torch.int64)
        vectors = vectors[initial_indices]
        if index_name == 'DiskANN':
            index = IndexClass()
            build_params = index_cfg.get('build_params', {})
            index.build(vectors, ids=initial_indices, **build_params)
            print(f"DiskANN index built")
        else:
            if not index_path.exists():
                index = IndexClass()
                build_params = index_cfg.get('build_params', {})
                print(f"Building index {index_name} with parameters: {build_params}")
                index.build(vectors, ids=initial_indices, **build_params)
                index.save(index_path)
                print(f"Index {index_name} built and saved to {index_path}")
            index = IndexClass()
            index.load(index_path)
            print(f"Index {index_name} loaded from {index_path}")
        return index

    def evaluate_workload(self, search_params, do_maintenance=False):
        """
        Evaluate the workload on the index. At the end a summary is printed and a multi-panel plot is saved.
        """
        # validate search_params
        assert 'k' in search_params, "search_params must contain 'k' for number of neighbors"

        # --- Load Workload and Index ---
        base_vectors = torch.load(self.base_vectors_path, weights_only=True).to(torch.float32)
        initial_indices = torch.load(self.initial_indices_path, weights_only=True).to(torch.int64)
        self.runbook = json.load(open(self.runbook_path, "r"))
        metric = self.runbook["parameters"]["metric"]
        query_vectors = (base_vectors if self.runbook["parameters"]["sample_queries"]
                         else torch.load(self.workload_dir / "query_vectors.pt", weights_only=True))
        query_vectors = query_vectors.to(torch.float32)
        start_time = time.time()
        self.index = self.build_or_load_index(self.index_cfg)
        init_time = time.time() - start_time
        self.runbook["initialize"]["time"] = init_time

        # --- Run Operations ---
        curr_ids = initial_indices
        curr_vectors = base_vectors[curr_ids]
        results = []
        for operation_id, operation in self.runbook["operations"].items():
            print(f"Operation {operation_id}/{len(self.runbook['operations'])}...")
            operation_id_int = int(operation_id)
            operation_type = operation["type"]
            operation_ids = torch.load(self.operations_dir / f"{operation_id}.pt", weights_only=True)
            if operation_type == "insert":
                curr_ids = torch.cat([curr_ids, operation_ids])
                curr_vectors = torch.cat([curr_vectors, base_vectors[operation_ids]])
                start_time = time.time()
                self.index.add(base_vectors[operation_ids], ids=operation_ids, num_threads=16)
                op_time = time.time() - start_time
                mean_recall = None
            elif operation_type == "delete":
                start_time = time.time()
                self.index.remove(operation_ids)
                op_time = time.time() - start_time
                mean_recall = None
            elif operation_type == "query":
                gt_ids = torch.load(self.operations_dir / f"{operation_id}_gt_ids.pt", weights_only=True)
                gt_dist = torch.load(self.operations_dir / f"{operation_id}_gt_dists.pt", weights_only=True)
                queries = query_vectors[operation_ids]
                Is, Ds, timing_infos = [], [], []
                start_time = time.time()
                for query in queries:
                    query = query.unsqueeze(0)
                    search_result = self.index.search(query, **search_params)
                    Is.append(search_result.ids)
                    Ds.append(search_result.distances)
                    timing_infos.append(search_result.timing_info)
                op_time = time.time() - start_time
                pred_ids = torch.cat(Is)
                recalls = compute_recall(pred_ids, gt_ids, search_params['k'])
                mean_recall = recalls.mean().item()
                self.runbook["operations"][operation_id]["recall"] = mean_recall
                total_time = sum([ti.total_time_ns for ti in timing_infos])
                mean_time = total_time / len(timing_infos)
                print(f"Query Time: {mean_time:.2f} ns, Recall: {mean_recall:.2f}")

            if do_maintenance:
                self.index.maintenance()

            n_resident = operation.get("n_resident", None)
            n_partitions = self.index.index.nlist() if hasattr(self.index, "index") and hasattr(self.index.index, "nlist") else None
            result = {
                'operation_number': operation_id_int,
                'operation_type': operation_type,
                'latency_ms': op_time * 1000,
                'recall': mean_recall,
                'n_resident': n_resident,
                'n_partitions': n_partitions,
                'index_name': self.index_cfg['name'],
            }
            result.update(self.index_cfg.get('build_params', {}))
            result.update(search_params)
            results.append(result)

        # --- Print Evaluation Summary ---
        # Gather per-operation metrics
        op_nums = [r['operation_number'] for r in results]
        latencies_insert = [r['latency_ms'] for r in results if r['operation_type']=='insert']
        op_nums_insert = [r['operation_number'] for r in results if r['operation_type']=='insert']
        latencies_delete = [r['latency_ms'] for r in results if r['operation_type']=='delete']
        op_nums_delete = [r['operation_number'] for r in results if r['operation_type']=='delete']
        latencies_query = [r['latency_ms'] for r in results if r['operation_type']=='query']
        op_nums_query = [r['operation_number'] for r in results if r['operation_type']=='query']
        query_recalls = [r['recall'] for r in results if r['operation_type']=='query' and r['recall'] is not None]
        n_vectors = [r['n_resident'] for r in results if r['n_resident'] is not None]

        avg_latency_insert = np.mean(latencies_insert) if latencies_insert else None
        avg_latency_delete = np.mean(latencies_delete) if latencies_delete else None
        avg_latency_query = np.mean(latencies_query) if latencies_query else None
        avg_query_recall = np.mean(query_recalls) if query_recalls else None

        print("\nWorkload Evaluation Summary:")
        if avg_latency_insert is not None:
            print(f"Average Insert Latency: {avg_latency_insert:.2f} ms")
        if avg_latency_delete is not None:
            print(f"Average Delete Latency: {avg_latency_delete:.2f} ms")
        if avg_latency_query is not None:
            print(f"Average Query Latency: {avg_latency_query:.2f} ms")
        if avg_query_recall is not None:
            print(f"Average Query Recall: {avg_query_recall:.2f}")

        # --- Generate Multi-Panel Plot ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Plot A: Latency per operation type
        ax = axs[0, 0]
        if op_nums_insert:
            ax.plot(op_nums_insert, latencies_insert, label='Insert', marker='o')
        if op_nums_delete:
            ax.plot(op_nums_delete, latencies_delete, label='Delete', marker='s')
        if op_nums_query:
            ax.plot(op_nums_query, latencies_query, label='Query', marker='^')
        ax.set_xlabel('Operation Number')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Operation Latency')
        ax.legend()
        # Plot B: Number of partitions per operation (if available)
        ax = axs[0, 1]
        partitions = [r['n_partitions'] for r in results if r['n_partitions'] is not None]
        op_nums_part = [r['operation_number'] for r in results if r['n_partitions'] is not None]
        if partitions:
            ax.plot(op_nums_part, partitions, marker='o')
            ax.set_xlabel('Operation Number')
            ax.set_ylabel('Number of Partitions')
            ax.set_title('Partitions per Operation')
        else:
            ax.text(0.5, 0.5, 'No partition info', ha='center', va='center')
            ax.axis('off')
        # Plot C: Resident set size per operation
        ax = axs[1, 0]
        if n_vectors:
            op_nums_vect = [r['operation_number'] for r in results if r['n_resident'] is not None]
            ax.plot(op_nums_vect, n_vectors, marker='o')
            ax.set_xlabel('Operation Number')
            ax.set_ylabel('Resident Vectors')
            ax.set_title('Resident Set Size')
        else:
            ax.text(0.5, 0.5, 'No resident set info', ha='center', va='center')
            ax.axis('off')
        # Plot D: Query recall per query operation
        ax = axs[1, 1]
        if op_nums_query and query_recalls:
            ax.plot(op_nums_query, query_recalls, marker='o')
            ax.set_xlabel('Operation Number')
            ax.set_ylabel('Query Recall')
            ax.set_title('Query Recall')
        else:
            ax.text(0.5, 0.5, 'No query recall info', ha='center', va='center')
            ax.axis('off')
        plt.tight_layout()
        plot_path = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_path)
        print(f"Saved evaluation plots to {plot_path}")
        plt.close()

        return results