import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import faiss

import test.experiments.osdi2025.experiment_utils as common_utils
from quake.utils import compute_recall
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
from quake.index_wrappers.quake import QuakeWrapper  # type: ignore

# optional wrappers
try: from quake.index_wrappers.scann import Scann
except ImportError: Scann = None
try: from quake.index_wrappers.diskann import DiskANNDynamic
except ImportError: DiskANNDynamic = None
try: from quake.index_wrappers.vamana import Vamana
except ImportError: Vamana = None

INDEX_CLASSES = {
    "Quake": QuakeWrapper,
    "IVF": FaissIVF,
    "HNSW": FaissHNSW,
    "SCANN": Scann,
    "DiskANN": DiskANNDynamic,
    "SVS": Vamana,
}

logger = logging.getLogger("numa_query_batch_orchestrator")

def _set_faiss_threads(val: str, tag: str):
    t = int(val)
    faiss.omp_set_num_threads(t)
    print(f"[{tag}] faiss.omp_set_num_threads({t})")

def task_build_index(idx_cfg, ds_cfg, run_params, idx_file: Path):
    name = idx_cfg["name"]
    print(f"[{name} BUILD] PID={os.getpid()} -> {idx_file}")
    bp = dict(idx_cfg.get("build_params", {}))
    omp_build = str(bp.pop("omp_num_threads_build", "1"))
    _set_faiss_threads(omp_build, f"{name} BUILD")

    # only need 1 query vector to build
    base_vecs, _, _ = common_utils.load_data(ds_cfg["name"], nq_override=1)
    Cls = INDEX_CLASSES[idx_cfg["index"]]
    idx = common_utils.prepare_wrapper_index(
        Cls, idx_file, base_vecs, bp, run_params["force_rebuild_indices"]
    )
    del base_vecs
    if idx is None:
        return {"error": "build failed"}
    return {"status": "success", "index_file_path": str(idx_file)}

def task_search_index(idx_cfg, ds_cfg, run_params, idx_file: Path, batch_size: int):
    name = idx_cfg["name"]
    itype = idx_cfg["index"]
    print(f"[{name} SEARCH bs={batch_size}] Loading index")
    bp = dict(idx_cfg.get("build_params", {}))
    sp = dict(idx_cfg.get("search_params", {}))
    omp_search = str(sp.pop("omp_num_threads_search", "1"))
    _set_faiss_threads(omp_search, f"{name} SEARCH")

    # load all nq queries & gt
    _, all_qvecs, all_gt = common_utils.load_data(ds_cfg["name"], nq_override=ds_cfg["num_queries"])
    nq = all_qvecs.shape[0]
    Cls = INDEX_CLASSES[itype]
    inst = Cls()
    load_kwargs = {
        k: bp[k] for k in ("num_workers","use_numa","num_merge_workers","parent") if k in bp
    }
    inst.load(str(idx_file), **load_kwargs)

    # warmup on a single chunk
    warmup_chunk = all_qvecs[:batch_size]
    for _ in range(run_params["num_warmup"]):
        _ = inst.search(warmup_chunk, run_params["k_val"], **sp)

    trial_total_latencies = []
    trial_recalls = []

    for t in range(run_params["num_trials"]):
        # iterate over chunks, summing latency
        total_ns = 0
        ids_list = []  # collect results per chunk
        for i in range(0, nq, batch_size):
            chunk = all_qvecs[i : min(i+batch_size, nq)]
            res = inst.search(chunk, run_params["k_val"], **sp)
            ti = getattr(res, "timing_info", None)
            ns = getattr(ti, "total_time_ns", None) or getattr(ti, "child_total_time_ns", 0)
            total_ns += ns
            ids_list.append(res.ids)

        total_ms = total_ns / 1e6
        trial_total_latencies.append(total_ms)

        # concatenate all ids and compute recall over nq
        all_ids = np.vstack(ids_list)  # shape (nq, k)
        rec = float(compute_recall(all_ids, all_gt, run_params["k_val"]).mean())
        trial_recalls.append(rec)

    mean_lat = float(np.mean(trial_total_latencies))
    std_lat  = float(np.std(trial_total_latencies))
    mean_rec = float(np.mean(trial_recalls))
    std_rec  = float(np.std(trial_recalls))

    return {
        "index": name,
        "query_batch_size": batch_size,
        "nq": nq,
        "mean_total_latency_ms": mean_lat,
        "std_total_latency_ms": std_lat,
        "mean_recall": mean_rec,
        "std_recall": std_rec
    }

def run_experiment(cfg_path, out_dir):
    logging.basicConfig(level=logging.INFO)
    cfg = common_utils.load_config(cfg_path)
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)

    ds_cfg = cfg["dataset"]
    batch_sizes = ds_cfg["query_batch_sizes"]
    run_params = {
        "num_trials": cfg.get("trials",3),
        "num_warmup": cfg.get("warmup",1),
        "k_val":      ds_cfg["k"],
        "force_rebuild_indices": cfg.get("force_rebuild",False)
    }

    all_rows = []
    # build once per index
    idx_store = out/"indices"
    idx_store.mkdir(exist_ok=True)
    for idx_cfg in cfg["indexes"]:
        idx_path = Path(idx_cfg.get("index_file", idx_store/f"{idx_cfg['name']}.bin"))
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        br = task_build_index(idx_cfg, ds_cfg, run_params, idx_path)
        if br.get("error"):
            logger.error(f"Build failed for {idx_cfg['name']}: {br['error']}")

    # now search for each batch_size
    for bs in batch_sizes:
        logger.info(f"--- Running batch size = {bs} ---")
        for idx_cfg in cfg["indexes"]:
            idx_path = Path(idx_cfg.get("index_file", idx_store/f"{idx_cfg['name']}.bin"))
            row = task_search_index(idx_cfg, ds_cfg, run_params, idx_path, bs)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # compute QPS = nq*1000 / mean_total_latency_ms
    df["QPS"] = df["nq"] * 1000.0 / df["mean_total_latency_ms"]
    df.to_csv(out/cfg["output"]["results_csv"], index=False)
    logger.info(f"Wrote results to {cfg['output']['results_csv']}")

    # plot only points with recall ≥ 0.9
    # plot_df = df[df["mean_recall"] >= 0.9]
    plt.figure(figsize=(8,6))
    for name, grp in df.groupby("index"):
        grp = grp.sort_values("query_batch_size")
        plt.errorbar(
            grp["query_batch_size"],
            grp["QPS"],
            yerr=None,
            marker="o",
            label=name
        )
    plt.xscale("log")
    plt.xlabel("Query Batch Size")
    plt.ylabel("QPS (@ Recall ≥ 0.9)")
    plt.title(f"QPS vs Batch Size ({ds_cfg['name']}, k={ds_cfg['k']})")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(out/f"qps_vs_batch_{ds_cfg['name']}.png")
    plt.close()
    logger.info("Saved QPS vs Batch Size plot to " + str(out/f"qps_vs_batch_{ds_cfg['name']}.png"))