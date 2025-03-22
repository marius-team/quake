import abc
from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch


def get_index_class(index_name):
    if index_name == "Quake":
        from quake.index_wrappers.quake import QuakeWrapper as IndexClass
    elif index_name == "HNSW":
        from quake.index_wrappers.faiss_hnsw import FaissHNSW as IndexClass
    elif index_name == "IVF":
        from quake.index_wrappers.faiss_ivf import FaissIVF as IndexClass
    elif index_name == "DiskANN":
        from quake.index_wrappers.diskann import DiskANNDynamic as IndexClass
    else:
        raise ValueError(f"Unknown index type: {index_name}")
    return IndexClass


class IndexWrapper(abc.ABC):
    """
    Wrapper interface of various index implementations (faiss, leviathan, etc.)
    """

    @abstractmethod
    def build(self, vectors: torch.Tensor, *args, ids: Optional[torch.Tensor] = None):
        """Build the index with the provided build arguments"""
        raise NotImplementedError("Subclasses must implement build method")

    @abstractmethod
    def search(self, query: torch.Tensor, k: int, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the k-nearest neighbors using the provided search arguments"""
        raise NotImplementedError("Subclasses must implement search method")

    @abstractmethod
    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None):
        """Add vectors to the index"""
        raise NotImplementedError("Subclasses must implement add method")

    @abstractmethod
    def remove(self, ids: torch.Tensor):
        """Remove vectors from the index"""
        raise NotImplementedError("Subclasses must implement remove method")

    @abstractmethod
    def save(self, directory: str):
        """Save the index to the given directory"""
        raise NotImplementedError("Subclasses must implement save method")

    @abstractmethod
    def load(self, directory: str):
        """Load the index from the given directory"""
        raise NotImplementedError("Subclasses must implement load method")

    @abstractmethod
    def centroids(self) -> Union[torch.Tensor, None]:
        """Return the centroids the index"""
        raise NotImplementedError("Subclasses must implement centroids method")

    @abstractmethod
    def n_total(self) -> int:
        """Return the number of vectors in the index"""
        raise NotImplementedError("Subclasses must implement n_total method")

    @abstractmethod
    def maintenance(self):
        """Perform any necessary maintenance operations on the index"""
        return None

    @abstractmethod
    def d(self) -> int:
        """Return the dimension of vectors in the index"""
        raise NotImplementedError("Subclasses must implement d method")

    @abstractmethod
    def index_state(self) -> dict:
        """Return the state of the index"""
        raise NotImplementedError("Subclasses must implement index_state method")
