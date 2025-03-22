from typing import Optional, Tuple

import numpy as np
import svs
import torch

from quake.index_wrappers.wrapper import IndexWrapper
from quake.utils import to_numpy, to_path, to_torch


class Vamana(IndexWrapper):
    index: svs.DynamicVamana

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.size

    def d(self) -> int:
        return self.index.dimensions

    def build(
        self,
        vectors: torch.Tensor,
        ids: Optional[torch.Tensor] = None,
        metric: str = "l2",
        num_threads: int = 1,
        alpha: float = 0.95,
        graph_max_degree: int = 128,
        window_size: int = 128,
        max_candidate_pool_size: int = 128,
        prune_to: int = 128,
    ):
        parameters = svs.VamanaBuildParameters(
            alpha=alpha,
            graph_max_degree=graph_max_degree,
            window_size=window_size,
            max_candidate_pool_size=max_candidate_pool_size,
        )

        if metric == "l2":
            distance = svs.DistanceType.L2
        elif metric == "ip":
            distance = svs.DistanceType.MIP

        if ids is not None:
            ids = to_numpy(ids).astype(np.uint64)
        else:
            ids = np.arange(len(vectors)).astype(np.uint64)

        vectors = to_numpy(vectors).astype(np.float32)
        self.index = svs.DynamicVamana.build(
            parameters=parameters, data=vectors, ids=ids, distance_type=distance, num_threads=num_threads
        )
        self.index.search_window_size = 128

    def search(self, queries: torch.Tensor, k: int) -> Tuple[np.ndarray, np.ndarray]:
        indices, distances = self.index.search(queries=to_numpy(queries).astype(np.float32), n_neighbors=k)
        return to_torch(indices.astype(np.int64)), to_torch(distances)

    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None):
        if ids is None:
            ids = torch.arange(self.n_total(), self.n_total() + len(vectors))
        self.index.add(points=to_numpy(vectors).astype(np.float32), ids=to_numpy(ids).astype(np.uint64))

    def remove(self, ids: torch.Tensor):
        self.index.delete(ids=to_numpy(ids).astype(np.uint64))

    def save(self, directory: str):
        config_dir = to_path(directory) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        graph_dir = to_path(directory) / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        data_dir = to_path(directory) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.index.save(str(config_dir), str(graph_dir), str(data_dir))

    def load(self, directory: str, metrics: str = "ip"):
        config_dir = to_path(directory) / "config"
        graph_dir = to_path(directory) / "graph"
        data_dir = to_path(directory) / "data"

        graph_loader = svs.GraphLoader(str(graph_dir))
        data_loader = svs.VectorDataLoader(str(data_dir))

        if metrics == "l2":
            distance = svs.DistanceType.L2
        elif metrics == "ip":
            distance = svs.DistanceType.MIP

        self.index = svs.DynamicVamana(
            config_path=str(config_dir), graph_loader=graph_loader, data_loader=data_loader, distance=distance
        )

        self.index.search_window_size = 100
        self.index.num_threads = 16

    def centroids(self) -> torch.Tensor | None:
        return super().centroids()
