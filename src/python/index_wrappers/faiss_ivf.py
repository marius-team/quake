import time
from enum import Enum
from typing import Optional, Tuple, Union

import faiss
import torch

from quake import SearchResult, SearchTimingInfo
from quake.index_wrappers.wrapper import IndexWrapper
from quake.utils import to_torch


def metric_str_to_faiss(metric: str) -> int:
    """
    Convert the metric to the corresponding faiss metric.
    :param metric: The metric to convert.
    :return: The faiss metric.
    """
    metric = metric.lower()
    if metric == "l2":
        return faiss.METRIC_L2
    elif metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"Unknown metric: {metric}")


def faiss_metric_to_str(metric: int) -> str:
    """
    Convert the faiss metric to the corresponding string.
    :param metric: The faiss metric to convert.
    :return: The string metric.
    """
    if metric == faiss.METRIC_L2:
        return "l2"
    elif metric == faiss.METRIC_INNER_PRODUCT:
        return "ip"
    else:
        raise ValueError(f"Unknown metric: {metric}")


class IndexType(Enum):
    """
    Enum class for the different types of indexes.
    """

    FLAT = 0
    PQ = 1
    IVF = 2
    IVFPQ = 3


class FaissIVF(IndexWrapper):
    """
    Wrapper class for faiss indexes.
    """

    index: faiss.Index
    index_type: IndexType

    def __init__(self):
        self.index = None
        self.index_type = None

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

    def index_state(self) -> dict:
        """
        Return the state of the index.

        - `n_list`: The number of centroids in the index.
        - `n_total': The number of vectors in the index.
        - `metric`: The distance metric used in the index.

        :return: The state of the index as a dictionary.
        """
        return {
            "n_list": self.index.nlist,
            "n_total": self.index.ntotal,
            "metric": faiss_metric_to_str(self.index.metric_type),
        }

    def maintenance(self):
        return

    def build(
        self,
        vectors: torch.Tensor,
        nc: int,
        m: int = 0,
        b: int = 0,
        metric: str = "l2",
        ids: Optional[torch.Tensor] = None,
    ):
        """
        Build the index with the given vectors and arguments.
        When nc is 0, the index is a flat index or PQ index.
        When nc is not 0, the index is an IVF or IVFPQ index.
        (PQ is presence only if m and b are not 0)

        :param vectors: The vectors to build the index with.
        :param nc: The number of centroids (ivf).
        :param m: The number of subquantizers (pq), optional. Default is 0.
        :param b: The number of bytes per code (pq), optional. Default is 0.
        :param metric: The distance metric to use, optional. Default is "l2".

        :raises AssertionError: If any of the constraints on nc, m, or b are violated.
        """
        assert vectors.ndim == 2
        assert nc >= 0 and m >= 0 and b >= 0
        # ensure m and b are both zero or none-zero at the same time
        assert not (m == 0) ^ (b == 0)

        vec_dim = vectors.shape[1]
        metric = metric.lower()

        if nc == 0:
            if m == 0:
                if metric == "l2":
                    self.index = faiss.IndexFlatL2(vec_dim)
                elif metric == "ip":
                    self.index = faiss.IndexFlatIP(vec_dim)
                else:
                    raise ValueError(f"Invalid metric: {metric}")
                self.index_type = IndexType.FLAT
            else:
                q = faiss.IndexPQ(vec_dim, m, b)
                self.index = faiss.IndexRefineFlat(q)
                self.index_type = IndexType.PQ
        else:
            if metric == "l2":
                quantizer = faiss.IndexFlatL2(vec_dim)
            elif metric == "ip":
                quantizer = faiss.IndexFlatIP(vec_dim)
            else:
                raise ValueError(f"Invalid metric: {metric}")

            if m == 0:
                self.index = faiss.IndexIVFFlat(quantizer, vec_dim, nc, metric_str_to_faiss(metric))
                self.index_type = IndexType.IVF
            else:
                q = faiss.IndexIVFPQ(quantizer, vec_dim, nc, m, b)
                self.index = faiss.IndexRefineFlat(q)
                self.index_type = IndexType.IVFPQ

        if not self.index.is_trained:
            print("Training the index...")
            self.index.train(vectors)

        assert self.index.is_trained
        print("Adding vectors to the index...")
        if ids is None:
            self.index.add(vectors)
        else:
            self.index.add_with_ids(vectors, ids.to(torch.int64))
        print("Index built successfully.")

    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None, num_threads: int = 0):
        """
        Add vectors to the index.

        :param vectors: The vectors to add to the index.
        :param ids: The ids of the vectors to add to the index.

        :raises AssertionError: If the vectors are not 2-dimensional.
        """
        assert vectors.ndim == 2

        if ids is None:
            self.index.add(vectors)
        else:
            self.index.add_with_ids(vectors, ids.to(torch.int64))

        return None

    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from the index.
        :param ids: The ids of the vectors to remove.
        """
        self.index.remove_ids(ids)

        return None

    def search(
        self, query: torch.Tensor, k: int, nprobe: int = 1, rf: int = 1, batched_scan: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors of the query vectors.

        :param query: The query vectors.
        :param k: The number of nearest neighbors to find.
        :param nprobe: The number of centroids to visit during search. Default is 1. Only used for IVF indexes.
        :param rf: The re-ranking factor. Search for rf * k results to re-rank. Default is 1. Only used for PQ indexes.

        :return: The distances and indices of the k-nearest neighbors.
        """
        # try to set the nprobe value if the index is an IVF index
        try:
            ivf = faiss.extract_index_ivf(self.index)
            ivf.nprobe = nprobe
        except RuntimeError:
            pass

        # try to set the re-ranking factor if the index is a PQ index
        try:
            self.index.k_factor = rf
        except AttributeError:
            pass

        timing_info = SearchTimingInfo()
        start = time.time()
        distances, indices = self.index.search(query, k)
        end = time.time()

        timing_info.total_time_ns = int((end - start) * 1e9)

        search_result = SearchResult()
        search_result.ids = to_torch(indices)
        search_result.distances = to_torch(distances)
        search_result.timing_info = timing_info

        return search_result

        # The code below only works for faiss 1.18.0

        # match self.index_type:
        #     case IndexType.FLAT:
        #         params = faiss.SearchParameters()
        #     case IndexType.PQ:
        #         params = faiss.IndexRefineSearchParameters(k_factor=rf)
        #     case IndexType.IVF:
        #         params = faiss.SearchParametersIVF(nprobe=nprobe)
        #     case IndexType.IVFPQ:
        #         params = faiss.IndexRefineSearchParameters(
        #             k_factor=4, base_index_params=faiss.SearchParametersIVF(nprobe=nprobe)
        #         )
        # return self.index.search(query, k, params=params)

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

    def centroids(self) -> Union[torch.Tensor, None]:
        """
        Return the centroids of the index if it is an IVF index. Otherwise, return None.

        :return: The centroids of the index or None if the index is not an IVF index.
        :rtype: Union[torch.Tensor, None]
        """
        try:
            centroids = self.index.quantizer.reconstruct_n(0, self.index.nlist)
            return to_torch(centroids)
        except AttributeError:
            return None
