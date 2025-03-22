# from pathlib import Path
import time
from typing import Tuple

import diskannpy as dap
import numpy as np
import torch

from quake import SearchTimingInfo
from quake.index_wrappers.wrapper import IndexWrapper
from quake.utils import to_numpy, to_torch


class DiskANNDynamic(IndexWrapper):
    """
    Wrapper class for diskann dynamic indexes.
    """

    index: dap.DynamicMemoryIndex

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.get_current_count()

    def d(self) -> int:
        return self.index.get_dimensions()

    def build(
        self,
        vectors: torch.Tensor,
        ids: torch.Tensor = None,
        metric: str = "l2",
        max_vectors: int = 1_000_000,
        complexity: int = 32,
        graph_degree: int = 16,
        num_threads: int = 0,
    ):
        """
        Build the index with the given vectors and arguments.

        :param vectors: The vectors to build the index with.
        :type vectors: torch.Tensor
        :param metric: The distance metric to use, optional. Default is "l2".
        :type metric: str
        :param max_vectors: The maximum number of vectors the index can hold, optional. Default is 1_000_000.
        :type max_vectors: int
        :param complexity: The complexity of the index, optional. Default is 32.
        :type complexity: int
        :param graph_degree: The degree of the graph, optional. Default is 16.
        :type graph_degree: int
        :param num_threads: The number of threads to use, optional. Default is 0.
        :type num_threads: int
        """
        assert vectors.ndim == 2

        vectors = to_numpy(vectors)
        dtype = vectors.dtype
        d = vectors.shape[1]

        self.index = dap.DynamicMemoryIndex(
            distance_metric=metric,
            vector_dtype=dtype,
            dimensions=d,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            num_threads=num_threads,
        )

        if ids is None:
            ids = np.arange(vectors.shape[0]).astype(np.uint32)

        ids = to_numpy(ids).astype(np.uint32)
        ids = ids + 1  # ids cannot be 0, which is reserved for invalid ids
        self.index.batch_insert(vectors, ids)

    def search(
        self, query: torch.Tensor, k: int, complexity: int = 16, num_threads: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors using the provided search arguments.

        :param query: The query vectors.
        :type query: torch.Tensor
        :param k: The number of nearest neighbors to search for.
        :type k: int
        :param complexity: The search complexity, optional. Default is 16.
        :type complexity: int
        :param num_threads: The number of threads to use, optional. Default is 16.
        :type num_threads: int

        :return: The indices and distances of the k-nearest neighbors.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        assert self.index is not None
        assert query.ndim == 2

        query = to_numpy(query)
        timing_info = SearchTimingInfo()

        start = time.time()
        indices, distances = self.index.batch_search(
            query, k_neighbors=k, complexity=complexity, num_threads=num_threads
        )
        end = time.time()
        timing_info.total_time_us = int((end - start) * 1e6)
        indices = to_torch(indices.astype(np.int64))
        indices = indices - 1  # ids are 1-indexed
        distances = to_torch(distances)

        return indices, distances, timing_info

    def save(self, path: str):
        """
        Save the index to the given path.

        :param path: The path to save the index to.
        :type path: str
        """
        # assert self.index is not None
        # if not Path(path).exists():
        #     Path(path).mkdir(exist_ok=True)
        # self.index.save(path)

        # TODO: save/load index
        raise RuntimeError("DiskANNDynamic.save() not implemented")

    def load(
        self,
        path: str,
        max_vectors: int = 1_000_000,
        complexity: int = 32,
        graph_degree: int = 16,
        num_threads: int = 0,
    ):
        """
        Load the index from the given path.

        :param path: The path to load the index from.
        :type path: str
        """
        # self.index = dap.DynamicMemoryIndex.from_file(
        #     index_directory=path,
        #     max_vectors=max_vectors,
        #     complexity=complexity,
        #     graph_degree=graph_degree,
        #     num_threads=num_threads,
        # )

        # TODO: save/load index
        raise RuntimeError("DiskANNDynamic.save() not implemented")

    def add(self, vectors: torch.Tensor, ids: torch.Tensor, num_threads: int = 0):
        """
        Add vectors to the index.

        :param vectors: The vectors to add to the index.
        :type vectors: torch.Tensor
        """
        assert self.index is not None
        assert vectors.ndim == 2

        vectors = to_numpy(vectors)
        ids = to_numpy(ids).astype(np.uint32)
        ids = ids + 1

        self.index.batch_insert(vectors, ids, num_threads)

    def remove(self, ids: torch.Tensor, lazy: bool = True):
        """
        Remove vectors from the index.

        :param ids: The ids of the vectors to remove.
        :type ids: torch.Tensor
        :param lazy: Whether to remove the vectors lazily, optional. Default is False.
        :type lazy: bool
        """
        assert self.index is not None

        ids = to_numpy(ids).astype(np.uint32)
        ids = ids + 1

        for id in ids:
            self.index.mark_deleted(id)

        if not lazy:
            self.index.consolidate_delete()

    def centroids(self) -> torch.Tensor | None:
        return super().centroids()
