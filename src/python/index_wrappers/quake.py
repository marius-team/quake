from typing import Optional, Tuple, Union, List

import quake
import torch
from quake import QuakeIndex

from quake.index_wrappers.wrapper import IndexWrapper


class QuakeWrapper(IndexWrapper):
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

    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from the index.

        :param indices: The indices of the vectors to remove.
        """
        assert self.index is not None
        assert ids.ndim == 1
        self.index.remove(ids)

    def search(self, query: torch.Tensor, k: int, nprobe: int = 1, recall_target: float = -1, k_factor=4.0, use_precomputed = True) -> Tuple[
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

    def maintenance(self):
        """
        Perform maintenance on the index.
        :return: maintenance results
        """
        return self.index.maintenance()

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
