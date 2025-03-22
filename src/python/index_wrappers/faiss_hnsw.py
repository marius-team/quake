import time
from typing import Optional, Tuple, Union

import faiss
import torch

from quake import SearchTimingInfo
from quake.index_wrappers.faiss_ivf import metric_str_to_faiss
from quake.index_wrappers.wrapper import IndexWrapper
from quake.utils import to_numpy, to_torch


class FaissHNSW(IndexWrapper):
    """
    Wrapper class for faiss hnsw indexes.
    """

    index: faiss.IndexHNSW

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        """
        Return the number of vectors in the index.

        :return: The number of vectors in the index.
        """
        return self.index.ntotal

    def d(self) -> int:
        """
        Return the dimension of the vectors in the index.

        :return: The dimension of the vectors in the index.
        """
        return self.index.d

    def build(
        self,
        vectors: torch.Tensor,
        m: int = 32,
        ef_construction: int = 40,
        metric: str = "l2",
        ids: Optional[torch.Tensor] = None,
    ):
        """
        Build the index with the given vectors and arguments.

        :param vectors: The vectors to build the index with.
        :param m: The number of neighbors for the HNSW graph, optional. Default is 32.
        :param ef_construction: The number of neighbors to explore during construction, optional. Default is 40.
        """
        assert vectors.ndim == 2
        assert m > 0

        metric = metric_str_to_faiss(metric)

        vectors = to_numpy(vectors)
        d = vectors.shape[1]
        self.index = faiss.IndexHNSWFlat(d, m, metric)
        self.index.hnsw.efConstruction = ef_construction

        self.index.add(vectors)

    def search(self, query: torch.Tensor, k: int, ef_search: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors using the provided search arguments.

        :param query: The query vectors.
        :param k: The number of neighbors to find.
        :param ef_search: The number of neighbors to explore during search, optional. Default is 16.
        :return: The indices and distances of the k-nearest neighbors.
        """
        assert query.ndim == 2
        assert k > 0

        self.index.hnsw.efSearch = ef_search

        query = to_numpy(query)
        # print(ef_search)
        timing_info = SearchTimingInfo()

        start = time.time()
        distances, indices = self.index.search(query, k)
        end = time.time()

        timing_info.total_time_us = int((end - start) * 1e6)
        distances = to_torch(distances)
        indices = to_torch(indices)

        return indices, distances, timing_info

    def save(self, filename: str):
        """
        Save the index to a file.

        :param filename: The name of the file to save the index to.
        """
        faiss.write_index(self.index, str(filename))

    def load(self, filename: str):
        """
        Load the index from a file.

        :param filename: The name of the file to load the index from.
        """
        self.index = faiss.read_index(str(filename))

    # Can't instantiate abstract class FaissHNSW with abstract method centroids
    def centroids(self) -> Union[torch.Tensor, None]:
        return super().centroids()

    def add(self, vectors: torch.Tensor, num_threads: int = 0):
        """
        Add vectors to the index.
        HNSW does only support sequential adds.

        :param vectors: The vectors to add.
        """
        assert vectors.ndim == 2

        vectors = to_numpy(vectors)
        self.index.add(vectors)

    # Faiss HNSW does not support removal of vectors
    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from the index.
        """
        # throw a runtime error
        raise RuntimeError("Faiss HNSW does not support removal of vectors.")
