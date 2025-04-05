import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert a string to a Path object.
    :param path: input path. Can be a string or a Path object. If it is a Path object, it will be returned as is.
    :return: Path object.
    """
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    else:
        raise ValueError("Input path must be a string or a Path object.")


def to_torch(tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor.
    :param tensor: input tensor. Can be a numpy array or a torch tensor. If a torch tensor, it will be returned as is.
    :return: torch tensor.
    """
    if isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor)
    elif isinstance(tensor, torch.Tensor):
        return tensor
    else:
        raise ValueError("Input tensor must be a numpy array or a torch tensor.")


def to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    :param tensor: input tensor. Can be a numpy array or a torch tensor. If a numpy array, it will be returned as is.
    :return: numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError("Input tensor must be a numpy array or a torch tensor.")


def is_metric_descending(metric: str) -> bool:
    """
    Check if the metric is descending.
    :param metric: distance metric to use. Can be 'l2' or 'ip'.
    :return: True if the metric is descending, False otherwise.
    """
    metric = metric.upper()

    if metric == "L2":
        return False
    elif metric == "IP":
        return True
    else:
        raise ValueError("Invalid metric. Must be 'l2' or 'ip'.")


def download_url(url, output_dir, overwrite):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    url_components = urlparse(url)
    filename = Path(url_components.path + url_components.query).name
    filepath = output_dir / filename

    if filepath.is_file() and not overwrite:
        print(f"File already exists: {filepath}")
    else:
        try:
            print(f"Downloading {filename} to {filepath}")
            urlretrieve(url, str(filepath))
        except OSError:
            raise RuntimeError(f"Failed to download {filename}")

    return filepath


def extract_file(filepath, remove_input=True):
    try:
        if tarfile.is_tarfile(str(filepath)):
            if str(filepath).endswith(".gzip") or str(filepath).endswith(".gz"):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".tar.gz") or str(filepath).endswith(".tgz"):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".tar"):
                with tarfile.open(filepath, "r:") as tar:
                    tar.extractall(path=filepath.parent)
            elif str(filepath).endswith(".bz2"):
                with tarfile.open(filepath, "r:bz2") as tar:
                    tar.extractall(path=filepath.parent)
            else:
                try:
                    with tarfile.open(filepath, "r:gz") as tar:
                        tar.extractall(path=filepath.parent)
                except tarfile.TarError:
                    raise RuntimeError(
                        "Unrecognized file format, may need to perform extraction manually with a custom dataset."
                    )
        elif zipfile.is_zipfile(str(filepath)):
            with ZipFile(filepath, "r") as zip:
                zip.extractall(filepath.parent)
        else:
            try:
                with filepath.with_suffix("").open("wb") as output_f, gzip.GzipFile(filepath) as gzip_f:
                    shutil.copyfileobj(gzip_f, output_f)
            except gzip.BadGzipFile:
                raise RuntimeError("Undefined file format.")
    except EOFError:
        raise RuntimeError("Dataset file isn't complete. Try downloading again.")

    if filepath.exists() and remove_input:
        filepath.unlink()

    return filepath.parent


def fvecs_to_tensor(filename):
    d = np.fromfile(filename, dtype=np.dtype("<u4"))[0]
    buffer = np.fromfile(filename, dtype=np.dtype("<f4"))
    buffer = buffer.reshape(-1, d + 1).astype(np.float32)
    return torch.from_numpy(buffer[:, 1:])


def ivecs_to_tensor(filename):
    d = np.fromfile(filename, dtype=np.dtype("<u4"))[0]
    buffer = np.fromfile(filename, dtype=np.dtype("<u4"))
    buffer = buffer.reshape(-1, d + 1).astype(np.int32)
    return torch.from_numpy(buffer[:, 1:])


def fbin_to_tensor(filename):
    n = np.fromfile(filename, dtype=np.int32, count=2)[0]
    d = np.fromfile(filename, dtype=np.int32, count=2)[1]
    numpy_array = np.fromfile(filename, dtype=np.float32)[2:].reshape(n, d)
    return torch.from_numpy(numpy_array)


def ibin_to_tensor(filename, header_size=0):
    n = np.fromfile(filename, dtype=np.int32, count=2)[0]
    d = np.fromfile(filename, dtype=np.int32, count=2)[1]
    numpy_array = np.fromfile(filename, dtype=np.int32, offset=8).reshape(2 * n, d)[:n]
    return torch.from_numpy(numpy_array)


def compute_recall(ids: torch.Tensor, gt_ids: torch.Tensor, k: int) -> torch.Tensor:
    ids = to_torch(ids)
    gt_ids = to_torch(gt_ids)

    ids = ids[:, :k]
    gt_ids = gt_ids[:, :k]

    num_queries = ids.size(0)

    assert ids.size() == gt_ids.size(), print(ids.shape, gt_ids.shape)

    recall = torch.zeros(num_queries)
    for i in range(num_queries):
        recall[i] = len(set(ids[i].tolist()).intersection(set(gt_ids[i].tolist()))) / k

    return recall


def compute_distance(x: torch.Tensor, y: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """
    Compute the distance between two tensors.
    :param x: input tensor.
    :param y: input tensor.
    :param metric: distance metric to use. Can be 'ip' or 'l2'.
    :return: the distance between the two tensors.
    """
    if metric.upper() == "IP":
        return torch.matmul(x, y.T)
    elif metric.upper() == "L2":
        return torch.cdist(x, y)


def knn(
    queries: torch.Tensor, vectors: torch.Tensor, k: int = 1, metric: str = "l2"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the k-nearest neighbors of the queries in the vectors.
    :param queries: input queries. Can be a 1D or 2D tensor.
    :param vectors: input vectors. Must be a 2D tensor. Last dimension must match the last dimension of queries.
    :param k: number of nearest neighbors to return.
    :param metric: distance metric to use. Can be 'ip' or 'l2'.
    :return: the indices and distances of the k-nearest neighbors of the queries in the vectors.
    """
    queries = to_torch(queries)
    vectors = to_torch(vectors)

    # check shape of the input tensors
    assert queries.dim() == 1 or queries.dim() == 2
    assert vectors.dim() == 2

    # if queries is a 1D tensor, convert it to a 2D tensor
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)

    assert queries.size(1) == vectors.size(1)

    num_queries = queries.size(0)

    distances = compute_distance(queries, vectors, metric)

    if k == -1:
        indices = torch.argsort(distances, dim=1, descending=is_metric_descending(metric))
        values = distances[torch.arange(num_queries).unsqueeze(1), indices]
    else:
        topk = torch.topk(distances, k, largest=is_metric_descending(metric))
        indices, values = topk.indices, topk.values

    return indices, values
