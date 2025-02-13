import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


from quake.utils import compute_recall, knn, to_path
from quake import MaintenancePolicyParams
from quake.index_wrappers.dynamic_ivf import DynamicIVF
import hashlib


def get_index_class(index_name):
    if index_name == 'DynamicIVF':
        from quake.index_wrappers.dynamic_ivf import DynamicIVF as IndexClass
    elif index_name == 'HNSW':
        from quake.index_wrappers.faiss_hnsw import FaissHNSW as IndexClass
    elif index_name == 'IVF':
        from quake.index_wrappers.faiss_wrapper import FaissWrapper as IndexClass
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

    If `tune_nprobe` is True, the function dynamically adjusts the `nprobe` parameter
    to find the minimum `nprobe` that achieves at least the target recall using binary search.

    Parameters:
    - index: The index to run the queries on.
    - queries (torch.Tensor): The queries to run.
    - search_k (int): The number of nearest neighbors to search for.
    - initial_nprobe (int): The initial number of clusters to search (used when `tune_nprobe` is False).
    - metric (str): The distance metric to use.
    - recall_target (float): The target recall to reach.
    - gt_ids (torch.Tensor): The ground truth IDs for the queries.
    - single_query (bool): Whether to run queries one by one or all at once.
    - tune_nprobe (bool): Whether to tune `nprobe` to meet the recall target.
    - nprobes (torch.Tensor): The `nprobe` values to use for each query (may be used when `tune_nprobe` is False).

    Returns:
    - recalls (torch.Tensor): The per-query recall values.
    - timing_infos (list): Timing information per query.
    """
    ids = [0] * len(queries)
    dists = [0] * len(queries)
    recalls = [0] * len(queries)
    timing_infos = []

    for i, query in enumerate(queries):
        query = query.unsqueeze(0)
        gt_id = gt_ids[i].unsqueeze(0)
        pred_ids, pred_dists, timing_info = index.search(query, search_k, **search_params)
        recall = compute_recall(pred_ids, gt_id, search_k).item()
        recalls.append(recall)
        timing_infos.append(timing_info)

    # if tune_nprobe:
    #     if single_query:
    #         # Adjust nprobe per query individually
    #         recalls = []
    #         timing_infos = []
    #         for i, query in enumerate(queries):
    #             print(f"Running query {i + 1}/{len(queries)} for nprobe tuning...")
    #             query = query.unsqueeze(0)  # Add batch dimension
    #             gt_id = gt_ids[i].unsqueeze(0)
    #
    #             nprobe_min = 1
    #             nprobe_max = nlist
    #             found_nprobe = None
    #             best_recall = 0.0
    #             best_timing_info = None
    #             max_iterations = 20
    #             iteration = 0
    #
    #             while nprobe_min <= nprobe_max and iteration < max_iterations:
    #                 nprobe = (nprobe_min + nprobe_max) // 2
    #
    #                 # Perform the search
    #                 pred_ids, pred_dists, timing_info = index.search(query, search_k, nprobe)
    #
    #                 # Compute recall
    #                 recall = compute_recall(pred_ids, gt_id, search_k).item()
    #
    #                 if recall >= recall_target:
    #                     # Found a valid nprobe; try to find a smaller one
    #                     found_nprobe = nprobe
    #                     best_recall = recall
    #                     best_timing_info = timing_info
    #                     nprobe_max = nprobe - 1
    #                 else:
    #                     # Current nprobe too small; search in the upper half
    #                     if recall > best_recall:
    #                         best_recall = recall
    #                         best_timing_info = timing_info
    #                     nprobe_min = nprobe + 1
    #
    #                 iteration += 1
    #
    #             if found_nprobe is None:
    #                 print(f"Query {i + 1}: Could not achieve recall target {recall_target}. Best recall: {best_recall}")
    #             else:
    #                 print(f"Query {i + 1}: Achieved recall target {recall_target} with nprobe={found_nprobe}")
    #
    #             recalls.append(best_recall)
    #             timing_infos.append(best_timing_info)
    #     else:
    #         # Adjust nprobe globally for all queries
    #         nprobe_min = 1
    #         nprobe_max = nlist
    #         found_nprobe = None
    #         best_recall = 0.0
    #         best_timing_info = None
    #         max_iterations = 20
    #         iteration = 0
    #
    #         while nprobe_min <= nprobe_max and iteration < max_iterations:
    #             nprobe = (nprobe_min + nprobe_max) // 2
    #             print(f"Global tuning iteration {iteration + 1}: Testing nprobe={nprobe}")
    #
    #             start_time = time.time()
    #
    #             # Perform the search
    #             pred_ids, pred_dists, timing_info = index.search(queries, search_k, nprobe)
    #
    #             # Compute recall
    #             recalls = compute_recall(pred_ids, gt_ids, search_k)
    #             current_recall = recalls.mean().item()
    #
    #             if current_recall >= recall_target:
    #                 # Found a valid nprobe; try to find a smaller one
    #                 found_nprobe = nprobe
    #                 best_recall = current_recall
    #                 best_timing_info = timing_info
    #                 nprobe_max = nprobe - 1
    #             else:
    #                 # Current nprobe too small; search in the upper half
    #                 if current_recall > best_recall:
    #                     best_recall = current_recall
    #                     best_timing_info = timing_info
    #                 nprobe_min = nprobe + 1
    #
    #             iteration += 1
    #
    #         if found_nprobe is None:
    #             print(f"Could not achieve recall target {recall_target}. Best recall: {best_recall}")
    #         else:
    #             print(f"Achieved recall target {recall_target} with nprobe={found_nprobe}")
    #
    #         # Use the best found recall and timing info
    #         recalls = recalls if single_query else compute_recall(pred_ids, gt_ids, search_k).tolist()
    #         timing_infos = [best_timing_info] if not single_query else timing_infos
    #
    # else:
    #     # Do not tune nprobe; use the initial_nprobe provided or the nprobes provided
    #     if single_query:
    #         recalls = []
    #         timing_infos = []
    #
    #         if nprobes is None:
    #             for i, query in enumerate(queries):
    #                 # print(f"Running query {i+1}/{len(queries)} with fixed nprobe...")
    #                 query = query.unsqueeze(0)  # Add batch dimension
    #                 gt_id = gt_ids[i].unsqueeze(0)
    #
    #                 pred_ids, pred_dists, timing_info = index.search(query, search_k, initial_nprobe)
    #                 recall = compute_recall(pred_ids, gt_id, search_k).item()
    #                 # print(f"Recall: {recall}")
    #
    #                 recalls.append(recall)
    #                 timing_infos.append(timing_info)
    #         else:
    #             for i, query in enumerate(queries):
    #                 print(f"Running query {i + 1}/{len(queries)} with fixed nprobe={nprobes[i]}...")
    #                 query = query.unsqueeze(0)
    #                 gt_id = gt_ids[i].unsqueeze(0)
    #
    #                 pred_ids, pred_dists, timing_info = index.search(query, search_k, nprobes[i])
    #                 recall = compute_recall(pred_ids, gt_id, search_k).item()
    #                 recalls.append(recall)
    #                 timing_infos.append(timing_info)
    #     else:
    #         pred_ids, pred_dists, timing_info = index.search(queries, search_k, initial_nprobe)
    #         recalls = compute_recall(pred_ids, gt_ids, search_k).tolist()
    #         timing_infos = [timing_info]

    # Convert recalls to tensor if not already
    if isinstance(recalls, list):
        recalls = torch.tensor(recalls)

    return recalls, timing_infos


class VectorSampler(ABC):
    """
    Abstract class for sampling vectors.
    """

    @abstractmethod
    def sample(self, size: int):
        """
        Sample vectors for an operation.

        :param size: The number of vectors to sample.
        """


class UniformSampler(VectorSampler):
    """
    Sample vectors uniformly.
    """

    def __init__(self):
        pass

    def sample(self, sample_pool: torch.Tensor, size: int, update_ranks: bool = True):
        """
        Sample vectors for an operation.

        :param sample_pool: The pool of vector ids to sample from.
        :param size: The number of vectors to sample.
        """
        randperm = torch.randperm(sample_pool.shape[0])
        sample_ids = sample_pool[randperm[:size]]
        return sample_ids


class StratifiedClusterSampler(VectorSampler):
    """
    Sample vectors from clusters.
    """

    def __init__(self, assignments: torch.Tensor, centroids: torch.Tensor):
        """
        Initialize the sampler with pool of vector ids to sample from.

        :param assignments: The cluster assignments of the vectors.
        :param centroids: The centroids of the clusters.
        """
        self.assignments = assignments
        self.centroids = centroids

        self.cluster_size = centroids.shape[0]

        # choose non-empty root cluster
        non_empty_clusters = torch.unique(assignments)

        self.root_cluster = non_empty_clusters[torch.randint(0, non_empty_clusters.shape[0], (1,))]
        self.update_ranks(self.root_cluster)

    def update_ranks(self, root_cluster: int):
        """
        Update the cluster ranks based on the root cluster.

        :param root_cluster: The root cluster to update the ranks based on.
        """
        self.root_cluster = root_cluster
        nearest_cluster_ids, _ = knn(self.centroids[root_cluster], self.centroids, -1, "l2")
        self.cluster_ranks = nearest_cluster_ids.flatten()

    def sample(self, sample_pool: torch.Tensor, size: int, update_ranks: bool = True):
        """
        Sample vectors for an operation.

        :param sample_pool: The pool of vector ids to sample from.
        :param size: The number of vectors to sample.
        :param update_ranks: Whether to update the cluster ranks after sampling.
        :return: The sampled vector ids.
        """
        sample_assignments = self.assignments[sample_pool]

        assert sample_assignments.shape[0] == sample_pool.shape[0]

        # remove empty clusters from the cluster ranks
        non_empty_clusters = torch.unique(sample_assignments)
        mask = torch.zeros(self.cluster_size, dtype=torch.bool)
        mask[non_empty_clusters] = True
        cluster_order = self.cluster_ranks[mask[self.cluster_ranks]]

        cluster_samples = []
        curr_sample_size = 0

        n_cluster_sampled = 0
        for cluster_id in cluster_order:
            mask = sample_assignments == cluster_id
            cluster_sample_pool = sample_pool[mask]

            assert cluster_sample_pool.shape[0] > 0

            cluster_sample_size = min(size - curr_sample_size, cluster_sample_pool.shape[0])

            cluster_sample = cluster_sample_pool[torch.randperm(cluster_sample_pool.shape[0])[:cluster_sample_size]]
            assert cluster_sample.shape[0] == cluster_sample_size

            cluster_samples.append(cluster_sample)
            curr_sample_size += cluster_sample_size
            n_cluster_sampled += 1

            if curr_sample_size >= size:
                break

        sample_ids = torch.cat(cluster_samples)

        # update the cluster ranks based on the last sampled cluster
        if update_ranks:
            self.update_ranks(cluster_id)

        # only take the unique samples
        sample_ids = torch.unique(sample_ids)

        return sample_ids


class DynamicWorkloadGenerator:
    """
    Takes in a static vector search dataset and generates a dynamic workload from it by sampling vectors
    for queries and updates.

    The workflow is as follows:
    1. The base_vectors are clustered using k-means.
    2. The workload is initialized with a set of vectors from the base_vectors.
    3. Operations are generated according to the specified ratios and parameters by sampling from the clusters.
    4. The workload is saved to a directory as a set of operations with a corresponding runbook.
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
        """
        Initialize the workload generator with the given parameters.

        :param workload_dir: The directory to save the workload to.
        :param base_vectors: The static vector dataset.
        :param metric: The distance metric to use.
        :param insert_ratio: The ratio of insert to delete operations.
        :param delete_ratio: The ratio of delete to insert operations.
        :param query_ratio: The ratio of query to update operations.
        :param update_batch_size: The number of vectors to modify in a single update operation.
        :param query_batch_size: The number of vectors to query in a single query operation.
        :param number_of_operations: The number of operations to generate.
        :param initial_size: The initial size to start the workload with.
        :param cluster_size: The size of the clusters to sample vectors from.
        :param cluster_sample_distribution: The distribution to use when sampling vectors for queries and updates.
        :param queries: The queries to use in the workload. If None, queries are sampled from the base_vectors.
        :param query_cluster_sample_distribution: The distribution to use when sampling vectors for queries.
        :param seed: The random seed to use.
        :param initial_clustering_path: The path to the initial clustering to use. Avoids re-clustering the base vectors.
        :param overwrite: Whether to overwrite the workload directory if it already exists.
        """
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

        if initial_clustering_path is not None:
            self.initial_clustering_path = to_path(initial_clustering_path)
        else:
            self.initial_clustering_path = None

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.validate_parameters()
        self.workload_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir = self.workload_dir / "operations"
        self.operations_dir.mkdir(parents=True, exist_ok=True)

        # set generator state
        self.resident_set = torch.zeros(
            base_vectors.shape[0], dtype=torch.bool
        )  # bitmask representing the vectors in the index
        self.all_ids = torch.arange(base_vectors.shape[0])
        self.assignments = None
        self.runbook = {}
        self.clustered_index = None
        self.cluster_ranks = None
        self.sampler = None

    def workload_exists(self):
        """
        Check if the workload runbook exists.
        """
        return (self.workload_dir / "runbook.json").exists()

    def validate_parameters(self):
        """
        Validate the parameters of the workload generator.
        """
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
        """
        Cluster the base vectors using faiss (k-means).
        """

        if self.initial_clustering_path is not None:
            index_dir = self.initial_clustering_path
        else:
            index_dir = self.workload_dir / "clustered_index.bin"

        if index_dir.exists():
            index = DynamicIVF()
            index.load(index_dir)
            n_clusters = index.index.nlist()
        else:
            # cluster the base vectors using faiss
            n_clusters = self.base_vectors.shape[0] // self.cluster_size

            index = DynamicIVF()
            index.build(
                self.base_vectors, nc=n_clusters, metric=self.metric, ids=torch.arange(self.base_vectors.shape[0])
            )
            index.save(str(self.workload_dir / "clustered_index.bin"))

        self.assignments = index.cluster_ids()

        print(self.assignments.shape)
        return index

    def sample(self, size: int, operation_type: str):
        """
        Sample vectors from the clustered index.

        :param size: The number of vectors to sample.
        :param operation_type: The type of operation to sample for.
        """
        if operation_type == "insert":
            # sample from the non-resident set
            sample_pool = self.all_ids[~self.resident_set]
        elif operation_type == "delete":
            sample_pool = self.all_ids[self.resident_set]
        elif operation_type == "query":
            if self.queries is None:
                sample_pool = self.all_ids[~self.resident_set]
            else:
                sample_pool = torch.arange(self.queries.shape[0])
        else:
            raise ValueError(f"Invalid operation type {operation_type}.")

        # if the sample_pool is empty, return an empty tensor
        if sample_pool.shape[0] == 0:
            return torch.tensor([], dtype=torch.long)

        if operation_type in ["insert", "delete"]:
            sample_ids = self.sampler.sample(sample_pool, size)
        else:
            # uniform sampler for queries
            update_ranks = True
            if self.query_cluster_sample_distribution == "skewed_fixed":
                update_ranks = False
            sample_ids = self.query_sampler.sample(sample_pool, size, update_ranks=update_ranks)

        return sample_ids

    def initialize_workload(self):
        """
        Initialize the workload with the initial size.
        """

        if self.cluster_sample_distribution == "skewed":
            self.sampler = StratifiedClusterSampler(self.assignments, self.clustered_index.centroids())
        elif self.cluster_sample_distribution == "uniform":
            self.sampler = UniformSampler()
        else:
            raise ValueError(f"Invalid cluster sample distribution {self.cluster_sample_distribution}.")

        if self.query_cluster_sample_distribution == "skewed":
            query_assignments = knn(self.queries, self.clustered_index.centroids(), 1, "l2")[0].flatten()
            self.query_sampler = StratifiedClusterSampler(query_assignments, self.clustered_index.centroids())
        elif self.query_cluster_sample_distribution == "skewed_fixed":
            query_assignments = knn(self.queries, self.clustered_index.centroids(), 1, "l2")[0].flatten()
            self.query_sampler = StratifiedClusterSampler(query_assignments, self.clustered_index.centroids())
        elif self.query_cluster_sample_distribution == "uniform":
            self.query_sampler = UniformSampler()
        else:
            raise ValueError(f"Invalid query cluster sample distribution {self.query_cluster_sample_distribution}.")

        # sample vectors for the initial size
        initial_indices = self.sample(self.initial_size, operation_type="insert")

        # mark the vectors as resident
        self.resident_set[initial_indices] = True

        # save the ids
        torch.save(initial_indices, self.workload_dir / "initial_indices.pt")

        # save the queries
        if self.queries is not None:
            torch.save(self.queries, self.workload_dir / "query_vectors.pt")

        # save the base vectors
        torch.save(self.base_vectors, self.workload_dir / "base_vectors.pt")

        # create the runbook entry
        entry = {
            "size": self.initial_size,
        }

        # save generator parameters
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
        self.runbook["initialize"] = entry
        self.runbook["operations"] = {}

    def generate_workload(self):
        """
        Generate the workload and save it to the directory.
        """

        # cluster the base vectors
        self.clustered_index = self.initialize_clustered_index()

        # initialize the workload with the initial size
        self.initialize_workload()

        # generate operations
        n_inserts = 0
        n_deletes = 0
        n_queries = 0
        n_operations = 0
        for i in range(self.number_of_operations):
            # decide the operation type such that the ratios are maintained

            # get the probabilities of each operation type
            operation_type = np.random.choice(
                ["insert", "delete", "query"], p=[self.insert_ratio, self.delete_ratio, self.query_ratio]
            )

            # sample vectors for the operation
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
            if operation_type == "insert" or operation_type == "delete":
                self.resident_set[sample_ids] = resident
            n_resident = self.resident_set.sum().item()

            if n_resident < 5 * self.update_batch_size:
                print(f"Below minimum resident set size: {n_resident}")
                break

            # create the operation entry
            entry = {"type": operation_type, "sample_size": sample_ids.shape[0], "n_resident": n_resident}

            # save the operation ids to the directory
            torch.save(sample_ids, self.operations_dir / f"{i}.pt")

            # if the operation is a query, compute the ground truth based on the resident set of vectors
            if operation_type == "query":
                if self.queries is None:
                    queries = self.base_vectors[sample_ids]
                else:
                    queries = self.queries[sample_ids]

                # compute the ground truth generation time
                start_time = time.time()

                resident_ids = self.all_ids[self.resident_set]
                curr_vectors = self.base_vectors[resident_ids]
                ids, dists = knn(queries, curr_vectors, 100, self.metric)

                ids = resident_ids[ids]

                gt_time = time.time() - start_time

                entry["gt_time"] = gt_time

                # save the ground truth to the directory
                torch.save(ids, self.operations_dir / f"{i}_gt_ids.pt")
                torch.save(dists, self.operations_dir / f"{i}_gt_dists.pt")

            print("Operation", i, entry)

            # add the operation to the operations list
            self.runbook["operations"][i] = entry

        self.runbook["summary"] = {
            "n_inserts": n_inserts,
            "n_deletes": n_deletes,
            "n_queries": n_queries,
            "n_operations": n_operations,
        }

        # save the runbook to the directory
        print("Saving runbook to", self.workload_dir / "runbook.json")
        with open(self.workload_dir / "runbook.json", "w") as f:
            json.dump(self.runbook, f, indent=4)


class WorkloadEvaluator:
    """
    Takes in a workload and evaluates it on a given index.
    """

    def __init__(
            self,
            workload_dir: Union[str, Path],
            index_cfg: dict,
            output_dir: Union[str, Path],
            base_vectors_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the workload evaluator with the given parameters.

        :param workload_dir: The directory containing the workload.
        :param index: The index to evaluate the workload on.
        :param output_dir: The directory to save the evaluation results to.
        :param base_vectors_path: The path to the base vectors. If None, the base vectors are loaded from the workload directory.
        """
        self.workload_dir = to_path(workload_dir)
        self.index_cfg = index_cfg
        self.index = None
        self.output_dir = to_path(output_dir)
        self.runbook_path = self.workload_dir / "runbook.json"
        self.operations_dir = self.workload_dir / "operations"
        self.initial_indices_path = self.workload_dir / "initial_indices.pt"
        self.base_vectors_path = base_vectors_path
        self.runbook = None
        self.initial_indices = None

        if self.base_vectors_path is None:
            self.base_vectors_path = self.workload_dir / "base_vectors.pt"


    def build_or_load_index(self, index_cfg, build_params, experiment_params={}):
        index_dir = self.workload_dir / "init_indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_name = index_cfg.name
        index_path = index_dir / f"{index_name}.index"

        IndexClass = get_index_class(index_name)

        # Load dataset
        vectors = torch.load(self.base_vectors_path).to(torch.float32)
        initial_indices = torch.load(self.initial_indices_path).to(torch.int64)
        vectors = vectors[initial_indices]

        if index_name == 'DiskANN':
            # diskann does not support saving/loading indexes currently
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

            # Load existing index
            index = IndexClass()

            # Set worker parameters during index loading for DynamicIVF
            if index_name == 'DynamicIVF':
                # Extract worker parameters
                n_workers = experiment_params.get('n_workers', 1)
                use_numa = experiment_params.get('use_numa', False)
                same_core = experiment_params.get('same_core', False)
                use_centroid_workers = experiment_params.get('use_centroid_workers', False)
                if use_numa is False and same_core is True:
                    return None  # Skip invalid configuration

                index.load(
                    index_path,
                    n_workers=n_workers,
                    use_numa=use_numa,
                    same_core=same_core,
                    use_centroid_workers=use_centroid_workers,
                )
                print(f"DynamicIVF index loaded with n_workers={n_workers}, use_numa={use_numa}, same_core={same_core}, use_centroid_workers={use_centroid_workers}")
            else:
                # For other indexes, load without worker parameters
                index.load(index_path)
                print(f"Index {index_name} loaded from {index_path}")

        return index


    def evaluate_workload(
            self, search_k: int, build_params, search_params, maintenance_params={}, experiment_params={}
    ):
        """
        Evaluate the workload on the given index.

        :param search_k: The number of nearest neighbors to search for.
        :param index_maintenance: The maintenance strategy to use.
        :param build_params: The parameters to use when building the index.
        """
        base_vectors = torch.load(self.base_vectors_path).to(torch.float32)
        initial_indices = torch.load(self.initial_indices_path).to(torch.int64)
        self.runbook = json.load(open(self.runbook_path, "r"))
        metric = self.runbook["parameters"]["metric"]

        if self.runbook["parameters"]["sample_queries"] is True:
            query_vectors = base_vectors
        else:
            query_vectors = torch.load(self.workload_dir / "query_vectors.pt")

        query_vectors = query_vectors.to(torch.float32)

        # load the initial indices
        start_time = time.time()
        self.index = self.build_or_load_index(self.index_cfg, build_params, experiment_params)
        init_time = time.time() - start_time

        self.runbook["initialize"]["time"] = init_time

        if isinstance(self.index, DynamicIVF) and len(maintenance_params.keys()) > 0:
            policy = MaintenancePolicyParams()

            # check if policy name is in the maintenance_params
            if maintenance_params.maintenance_policy:
                policy.maintenance_policy = maintenance_params.maintenance_policy
            else:
                policy.maintenance_policy = "query_cost"

            if policy.maintenance_policy == "query_cost":
                policy.alpha = maintenance_params.alpha
                policy.window_size = maintenance_params.window_size
                policy.refinement_radius = maintenance_params.refinement_radius
                policy.refinement_iterations = maintenance_params.refinement_iterations
                policy.delete_threshold_ns = maintenance_params.delete_threshold_ns
                policy.split_threshold_ns = maintenance_params.split_threshold_ns
            elif policy.maintenance_policy == "lire":
                policy.target_partition_size = maintenance_params.target_partition_size

            self.index.index.set_maintenance_policy_params(policy)


        curr_ids = initial_indices
        curr_vectors = base_vectors[curr_ids]

        results = []

        # evaluate the operations
        for operation_id, operation in self.runbook["operations"].items():
            print(f"Operation {operation_id}/{len(self.runbook['operations'])}...")

            operation_id = int(operation_id)
            operation_type = operation["type"]

            operation_ids = torch.load(self.operations_dir / f"{operation_id}.pt")
            # check that there are no duplicates
            unique_operation_ids = torch.unique(operation_ids)
            assert unique_operation_ids.shape[0] == operation_ids.shape[0]

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
                gt_ids = torch.load(self.operations_dir / f"{operation_id}_gt_ids.pt")
                gt_dist = torch.load(self.operations_dir / f"{operation_id}_gt_dists.pt")

                # compute ground truth to debug
                queries = query_vectors[operation_ids]

                Is = [0] * len(queries)
                Ds = [0] * len(queries)
                timing_infos = [0] * len(queries)
                start_time = time.time()

                for i, query in enumerate(queries):
                    query = query.unsqueeze(0)
                    I, D, timing_info = self.index.search(query, search_k, **search_params)

                    Is[i] = I
                    Ds[i] = D
                    timing_infos[i] = timing_info

                op_time = time.time() - start_time

                pred_ids = torch.cat(Is)
                recalls = compute_recall(pred_ids, gt_ids, search_k)
                mean_recall = recalls.mean().item()
                self.runbook["operations"][str(operation_id)]["recall"] = mean_recall

                # get the quantizer time and the partition scan time
                total_time_partition_scan = 0
                total_time_quantizer = 0
                total_time_metadata = 0
                total_time = 0
                for timing_info in timing_infos:
                    total_time_partition_scan += (timing_info.partition_scan_time_us)
                    total_time_quantizer += (timing_info.quantizer_search_time_us)
                    total_time += (timing_info.total_time_us)
                    total_time_metadata += (timing_info.metadata_update_time_us)

                mean_time_partition_scan = total_time_partition_scan / len(timing_infos)
                mean_time_quantizer = total_time_quantizer / len(timing_infos)
                mean_time = total_time / len(timing_infos)
                mean_time_metadata = total_time_metadata / len(timing_infos)

                # compute the recall
                mean_recall = recalls.mean().item()

                print(f"Query Time: {mean_time:.2f} us Mean Partition Scan Time: {mean_time_partition_scan:.2f} us, Mean Quantizer Time: {mean_time_quantizer:.2f} us, Mean Metadata Time: {mean_time_metadata:.2f} us, Recall: {mean_recall:.2f}")

            self.runbook["operations"][str(operation_id)]["time"] = op_time

            # Record results
            result = {
                'operation_number': operation_id,
                'operation_type': operation_type,
                'latency_ms': op_time * 1000,
                'recall': mean_recall,
                'index_name': self.index_cfg.name,
            }

            # add experiment parameters to the result
            result.update(experiment_params)

            # add build parameters to the result
            result.update(build_params)

            # add search parameters to the result
            result.update(search_params)

            if isinstance(self.index, DynamicIVF) and len(maintenance_params.keys()) > 0:
                maintenance_info = self.index.index.maintenance()  # Ensure this method exists
                result.update({
                    'nlist': self.index.index.nlist(),  # Ensure this method exists
                    'maintenance': {
                        "n_splits": maintenance_info.n_splits,
                        "n_deletes": maintenance_info.n_deletes,
                        "delete_time_us": maintenance_info.delete_time_us,
                        "delete_refine_time_us": maintenance_info.delete_refine_time_us,
                        "split_time_us": maintenance_info.split_time_us,
                        "split_refine_time_us": maintenance_info.split_refine_time_us,
                        "total_time_us": maintenance_info.total_time_us,
                    }
                })

            results.append(result)

        return results