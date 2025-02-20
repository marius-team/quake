from typing import Optional, Tuple, Union, List

import quake
import torch
from quake import QuakeIndex

from quake.index_wrappers.wrapper import IndexWrapper


class DynamicIVF(IndexWrapper):
    index: QuakeIndex
    assignments: Union[torch.Tensor, None]

    def __init__(self):
        self.index = None
        self.index_type = None
        self.assignments = None

    def n_total(self) -> int:
        """
        Return the number of vectors in the index.

        :return: The number of vectors in the index.
        """
        return self.index.ntotal()

    def d(self) -> int:
        """
        Return the dimension of the vectors in the index.

        :return: The dimension of the vectors in the index.
        """
        return self.index.d

    def index_ready(self) -> bool:
        """
        Returns if the workers have been initialized in the index

        :return: A boolean indicating whether the workers have been initialized or nor
        """
        return self.index.index_ready()

    def build(self, vectors: torch.Tensor, nc: int, metric: str = "l2", ids: Optional[torch.Tensor] = None,
              n_workers: int = 1, m: int = -1, code_size: int = 8):
        """
        Build the index with the given vectors and arguments.

        :param vectors: The vectors to build the index with.
        :param nc: The number of centroids (ivf).
        :param metric: The distance metric to use, optional. Default is "l2".
        :param ids: The ids of the vectors.
        """
        assert vectors.ndim == 2
        assert nc > 0

        vec_dim = vectors.shape[1]
        metric = metric.lower()
        print(
            f"Building index with {vectors.shape[0]} vectors of dimension {vec_dim} and {nc} centroids, with metric {metric}.")
        # self.index = QuakeIndex(vec_dim, nc, metric_str_to_faiss(metric), n_workers, m, code_size)

        build_params = quake.IndexBuildParams()
        build_params.metric = metric
        build_params.nlist = nc
        build_params.num_workers = n_workers

        self.index = QuakeIndex()

        if ids is None:
            ids = torch.arange(vectors.shape[0], dtype=torch.int64)

        self.index.build(vectors, ids.to(torch.int64), build_params)

        print("Index built successfully.")

    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None, num_threads: int = 0):
        """
        Add vectors to the index.

        :param vectors: The vectors to add to the index.
        :param ids: The ids of the vectors to add to the index.
        """
        assert self.index is not None
        assert vectors.ndim == 2

        if ids is None:
            curr_id = self.n_total()
            ids = torch.arange(curr_id, curr_id + vectors.shape[0], dtype=torch.int64)

        self.index.add(vectors, ids)

    def set_worker_counts(self, all_workers: List[int], same_core: bool, use_numa_optimizations: bool):
        curr_index = self.index
        for curr_worker_count in reversed(all_workers):
            curr_index.reset_workers(curr_worker_count, same_core, use_numa_optimizations)
            curr_index = curr_index.parent

    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from the index.

        :param indices: The indices of the vectors to remove.
        """
        assert self.index is not None
        assert ids.ndim == 1
        self.index.remove(ids)

    def search(self, query: torch.Tensor, k: int, nprobe: int = 1, recall_target: float = .9, k_factor=4.0, use_precomputed = True) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors of the query vectors.

        :param query: The query vectors.
        :param k: The number of nearest neighbors to find.
        :param nprobe: The number of centroids to visit during search. Default is 1.

        :return: The distances and indices of the k-nearest neighbors.
        """
        search_params = quake.SearchParams()
        search_params.nprobe = nprobe
        search_params.recall_target = recall_target
        search_params.use_precomputed = use_precomputed
        search_params.k = k
        return self.index.search(query, search_params)

    def save(self, filename: str):
        """
        Save the index to a file.

        :param filename: The name of the file to save the index to.
        """
        self.index.save(str(filename))

    def load(self, filename: str, n_workers: int = 1, use_numa: bool = False, verbose: bool = False,
             verify_numa: bool = False, same_core: bool = True, use_centroid_workers: bool = False, use_adaptive_n_probe : bool = False):
        """
        Load the index from a file.

        :param filename: The name of the file to load the index from.
        """
        print(
            f"Loading index from {filename}, with {n_workers} workers, use_numa={use_numa}, verbose={verbose}, verify_numa={verify_numa}, same_core={same_core}, use_centroid_workers={use_centroid_workers}")
        self.index = QuakeIndex()
        self.index.load(str(filename), True)
    
    def set_timeout_values(self, max_query_latency : int = -1, flush_gap_time : int = -1):
        """
        Set the timeout values for the index

        :param max_query_latency: The end to end query time we are trying to hit (in uS)
        :param flush_gap_time: How often we flush the shared buffer where results are being written (in uS)
        """
        self.index.set_timeout_values(max_query_latency, flush_gap_time)

    def centroids(self) -> torch.Tensor:
        """
        Return the centroids of the index.

        :return: The centroids of the index
        """
        return self.index.centroids()

    def cluster_ids(self) -> torch.Tensor:
        """
        Return the cluster assignments of the vectors in the index.

        :return: The cluster ids of the index
        """
        return self.index.parent.get_ids()

    def metric(self) -> str:
        """
        Return the metric of the index.

        :return: The metric of the index.
        """

        return self.index.metric()
