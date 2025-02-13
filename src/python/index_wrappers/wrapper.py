import abc
from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch


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
    def d(self) -> int:
        """Return the dimension of vectors in the index"""
        raise NotImplementedError("Subclasses must implement d method")
