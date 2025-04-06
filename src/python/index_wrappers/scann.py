import os
import re
from typing import Optional, Tuple

import numpy as np
import scann
import torch

from quake.index_wrappers.wrapper import IndexWrapper
from quake.utils import to_numpy, to_torch


class Scann(IndexWrapper):
    index: scann.scann_ops_pybind.ScannSearcher

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.size()

    def d(self) -> int:
        config = self.index.config()
        match = re.search(r"input_dim:\s*(\d+)", config)
        return int(match.group(1))

    def build(
        self,
        vectors: torch.Tensor,
        num_neighbors: int = 10,
        metric: str = "squared_l2",
        num_leaves: int = 4096,
        num_leaves_to_search: int = 100,
        training_sample_size: int = -1,
        dimensions_per_block: int = 2,
        anisotropic_quantization_threshold=float("nan"),
        min_cluster_size=100,
        reordering_num: int = 100,
        ids: Optional[torch.Tensor] = None,
        mode: str = "no_autopilot",
    ):
        """
        Build the index with the given vectors and arguments.

        :param vectors: The vectors to build the index with.
        :param num_neighbors: The number of neighbors to return.
        :param metric: The distance measure to use. Options are "dot_product" and "squared_l2".
        :param num_leaves: The number of leaves in the search tree.
        :param num_leaves_to_search: The number of leaves to search.
        :param training_sample_size: The size of the training sample.
        :param dimensions_per_block: The number of dimensions per block.
        :param anisotropic_quantization_threshold: The anisotropic quantization threshold.
        :param min_cluster_size: The minimum cluster size.
        :param reordering_num: The reordering number.
        :param ids: The ids of the vectors.
        :param mode: The mode to use. Options are "no_autopilot", "online", and "online_incremental".
        """
        assert vectors.ndim == 2

        if training_sample_size == -1:
            training_sample_size = vectors.shape[0]

        if ids is None:
            ids = np.arange(vectors.shape[0]).tolist()
        else:
            assert ids.ndim == 1
            ids = to_numpy(ids).tolist()

        vectors = to_numpy(vectors)

        if mode != "no_autopilot":
            if mode == "online":
                searcher = (
                    scann.scann_ops_pybind.builder(vectors, num_neighbors, metric)
                    .autopilot(
                        mode=scann.scann_ops_pybind.scann_builder.IncrementalMode.ONLINE,
                    )
                    .build(docids=ids)
                )
                self.index = searcher
                return
            elif mode == "online_incremental":
                searcher = (
                    scann.scann_ops_pybind.builder(vectors, num_neighbors, metric)
                    .autopilot(
                        mode=scann.scann_ops_pybind.scann_builder.IncrementalMode.ONLINE_INCREMENTAL,
                    )
                    .score_brute_force()
                    .build(docids=ids)
                )
                self.index = searcher
                return
            else:
                raise ValueError(f"Invalid mode: {mode}")

        searcher = (
            scann.scann_ops_pybind.builder(vectors, num_neighbors, metric)
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=training_sample_size,
                incremental_threshold=1000,
            )
            .score_brute_force()
            .build(docids=ids)
        )

        self.index = searcher

    def search(
        self,
        query: torch.Tensor,
        k: int,
        leaves_to_search: int = 100,
        pre_reorder_num_neighbors: int = 20_000,
        n_threads: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors using the provided search arguments.

        :param query: The query vectors.
        :param k: The number of neighbors to find.
        :param leaves_to_search: The number of leaves to search.
        :param pre_reorder_num_neighbors: The number of neighbors to reorder.
        :param n_threads: The number of threads to use.
        """
        assert query.ndim == 2

        if not n_threads == 0:
            self.index.set_num_threads(n_threads)

        query = to_numpy(query)
        indices, distances = self.index.search_batched(
            query,
            final_num_neighbors=k,
            leaves_to_search=leaves_to_search,
            pre_reorder_num_neighbors=pre_reorder_num_neighbors,
        )
        indices = np.array(indices)

        return to_torch(indices), to_torch(distances)

    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None):
        """
        Add vectors to the index

        :param vectors: The vectors to add to the index.
        :param ids: The ids of the vectors to add to the index.
        """
        assert self.index is not None
        assert vectors.ndim == 2

        if ids is None:
            curr_id = self.n_total()
            ids = np.arange(curr_id, curr_id + vectors.shape[0], dtype=np.int64)

        ids = to_numpy(ids).tolist()
        vectors = to_numpy(vectors)
        self.index.upsert(ids, vectors)

    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from the index.

        :param ids: The ids of the vectors to remove.
        """
        assert self.index is not None
        assert ids.ndim == 1

        ids = to_numpy(ids).tolist()
        self.index.delete(ids)

    def save(self, filename: str):
        """
        Save the index to a file.

        :param filename: The name of the file to save the index to.
        """
        os.makedirs(filename, exist_ok=True)
        self.index.serialize(str(filename))

    def load(self, filename: str):
        """
        Load the index from a file.

        :param filename: The name of the file to load the index from.
        """
        self.index = scann.scann_ops_pybind.load_searcher(str(filename))

    def centroids(self) -> torch.Tensor | None:
        return super().centroids()
