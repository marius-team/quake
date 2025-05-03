import abc
from abc import abstractmethod
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from quake.utils import download_url, extract_file, fvecs_to_tensor, ivecs_to_tensor, to_path, knn

DEFAULT_DOWNLOAD_DIR = Path("data/")


class Dataset(abc.ABC):
    url: str = None
    download_dir: Path = None
    metric: str = "l2"

    @abstractmethod
    def is_downloaded(self) -> bool:
        pass

    @abstractmethod
    def download(self, overwrite: bool = False):
        pass

    @abstractmethod
    def load_vectors(self) -> Union[np.ndarray, torch.Tensor]:
        pass

    @abstractmethod
    def load_queries(self) -> Union[np.ndarray, torch.Tensor]:
        pass

    @abstractmethod
    def load_ground_truth(self) -> Union[np.ndarray, torch.Tensor]:
        pass

    def load(self) -> List[Union[np.ndarray, torch.Tensor]]:
        return [self.load_vectors(), self.load_queries(), self.load_ground_truth()]


class Sift1m(Dataset):
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

    def __init__(self, download_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR):
        self.download_dir = to_path(download_dir)
        self.data_dir = self.download_dir / "sift"
        self.downloaded = False
        self.metric = "l2"

    def is_downloaded(self) -> bool:
        print(self.data_dir.absolute())
        base = self.data_dir / "sift_base.fvecs"
        query = self.data_dir / "sift_query.fvecs"
        gt = self.data_dir / "sift_groundtruth.ivecs"
        return base.exists() and query.exists() and gt.exists()

    def download(self, overwrite: bool = False):
        if not self.is_downloaded() or overwrite:
            download_file = download_url(self.url, output_dir=self.download_dir, overwrite=overwrite)
            extract_file(download_file, remove_input=False)
            self.downloaded = True

    def load_vectors(self) -> Union[np.ndarray, torch.Tensor]:
        return fvecs_to_tensor(self.data_dir / "sift_base.fvecs")

    def load_queries(self) -> Union[np.ndarray, torch.Tensor]:
        return fvecs_to_tensor(self.data_dir / "sift_query.fvecs")

    def load_ground_truth(self) -> Union[np.ndarray, torch.Tensor]:
        return ivecs_to_tensor(self.data_dir / "sift_groundtruth.ivecs")

class UniformDataset(Dataset):
    def __init__(self, num_vectors: int, dim: int, download_dir: Union[str, Path] = DEFAULT_DOWNLOAD_DIR):
        self.num_vectors = num_vectors
        self.dim = dim
        self.download_dir = to_path(download_dir)

    def is_downloaded(self) -> bool:
        # check if files exist
        vecs = self.download_dir / "vectors.pt"
        queries = self.download_dir / "queries.pt"
        gt = self.download_dir / "ground_truth.pt"

        return vecs.exists() and queries.exists() and gt.exists()

    def download(self, overwrite: bool = False):

        vecs = np.random.uniform(0, 1, (self.num_vectors, self.dim)).astype(np.float32)
        queries = np.random.uniform(0, 1, (1000, self.dim)).astype(np.float32)
        gt = knn(queries, vecs, k=100)[0]

        # vecs = torch.randn(self.num_vectors, self.dim)
        # queries = torch.randn(1000, self.dim)
        # gt = knn(queries, vecs, k=100)[0]


        np.save(self.download_dir / "vectors.npy", vecs)
        np.save(self.download_dir / "queries.npy", queries)
        np.save(self.download_dir / "ground_truth.npy", gt)

    def load_vectors(self) -> Union[np.ndarray, torch.Tensor]:
        # return torch.randn(self.num_vectors, self.dim)
        # Load the vectors from disk
        return torch.from_numpy(np.load(self.download_dir / "vectors.npy"))

    def load_queries(self) -> Union[np.ndarray, torch.Tensor]:
        # return torch.randn(self.num_vectors, self.dim)
        # Load the queries from disk
        return torch.from_numpy(np.load(self.download_dir / "queries.npy"))

    def load_ground_truth(self) -> Union[np.ndarray, torch.Tensor]:
        # return torch.randint(0, self.num_vectors, (self.num_vectors, 100))
        # Load the ground truth from disk
        return torch.from_numpy(np.load(self.download_dir / "ground_truth.npy"))


def load_dataset(
    name: str, download_dir: str = DEFAULT_DOWNLOAD_DIR, overwrite_download: bool = False
) -> List[Union[np.ndarray, torch.Tensor]]:
    if name.lower() == "sift1m":
        dataset = Sift1m(download_dir=download_dir)
    elif name.lower() == "uniform":
        dataset = UniformDataset(num_vectors=100000, dim=8, download_dir=download_dir)
    else:
        raise RuntimeError("Unimplemented dataset")

    dataset.download(overwrite=overwrite_download)
    return dataset.load()
