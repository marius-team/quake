import time as t

import faiss
import torch

from quake import IndexBuildParams, QuakeIndex, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall

if __name__ == "__main__":
    print("Testing bindings")

    search_params = SearchParams()
    search_params.k = 10
    search_params.nprobe = 10
    vectors, queries, gt = load_dataset("sift1m")
    ids = torch.arange(vectors.size(0))

    nq = 1000
    queries = queries[:nq].contiguous()
    gt = gt[:nq]

    exhaustive_index = QuakeIndex()
    build_params = IndexBuildParams()
    exhaustive_index.build(vectors, ids, build_params)
    start = t.time()
    search_result = exhaustive_index.search(queries, search_params)
    end = t.time()
    recall = compute_recall(search_result.ids, gt, search_params.k)
    print("Exhaustive search", recall.mean(), "Time", end - start)

    ivf_index = QuakeIndex()
    build_params = IndexBuildParams()
    build_params.nlist = 1000
    ivf_index.build(vectors, ids, build_params)
    start = t.time()
    search_result = ivf_index.search(queries, search_params)
    end = t.time()
    recall = compute_recall(search_result.ids, gt, search_params.k)
    print("IVF search", recall.mean(), "Time", end - start)
    print(search_result.timing_info.partition_scan_time_us, search_result.timing_info.quantizer_search_time_us)

    faiss_ivf_index = faiss.index_factory(vectors.size(1), "IVF1000,Flat")
    faiss_ivf_index.train(vectors.numpy())
    faiss_ivf_index.add(vectors.numpy())
    start = t.time()
    faiss_ivf_index.nprobe = search_params.nprobe
    dists, ids = faiss_ivf_index.search(queries.numpy(), search_params.k)
    end = t.time()
    recall = compute_recall(torch.from_numpy(ids), gt, search_params.k)
    print("Faiss IVF search", recall.mean(), "Time", end - start)
