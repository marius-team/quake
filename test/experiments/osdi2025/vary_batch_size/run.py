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

    if idx_file.exists() and not run_params["force_rebuild_indices"]:
        return {"status": "skipped", "index_file_path": str(idx_file)}

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
    print(f"[{name} SEARCH bs={batch_size}] PID={os.getpid()} Loading index")
    bp = dict(idx_cfg.get("build_params", {}))
    sp = dict(idx_cfg.get("search_params", {}))
    omp_search = str(sp.pop("omp_num_threads_search", "1"))
    _set_faiss_threads(omp_search, f"{name} SEARCH")

    # load all nq queries & gt
    _, all_qvecs, all_gt = common_utils.load_data(
        ds_cfg["name"], nq_override=ds_cfg["num_queries"]
    )
    nq = all_qvecs.shape[0]

    Cls = INDEX_CLASSES[itype]
    inst = Cls()
    load_kwargs = {
        k: bp[k]
        for k in ("num_workers", "use_numa", "num_merge_workers", "parent")
        if k in bp
    }
    inst.load(str(idx_file), **load_kwargs)

    # warmup on one chunk
    warmup = all_qvecs[:batch_size]
    for _ in range(run_params["num_warmup"]):
        _ = inst.search(warmup, run_params["k_val"], **sp)

    trial_latencies = []
    trial_recalls = []

    for _ in range(run_params["num_trials"]):
        total_ns = 0
        ids_list = []
        for i in range(0, nq, batch_size):
            chunk = all_qvecs[i : min(i + batch_size, nq)]
            res = inst.search(chunk, run_params["k_val"], **sp)
            ti = getattr(res, "timing_info", None)
            ns = getattr(ti, "total_time_ns", None) or getattr(ti, "child_total_time_ns", 0)
            total_ns += ns
            ids_list.append(res.ids)

        total_ms = total_ns / 1e6
        trial_latencies.append(total_ms)

        all_ids = np.vstack(ids_list)
        rec = float(compute_recall(all_ids, all_gt, run_params["k_val"]).mean())
        trial_recalls.append(rec)

    return {
        "mean_total_latency_ms": float(np.mean(trial_latencies)),
        "std_total_latency_ms":   float(np.std(trial_latencies)),
        "mean_recall":            float(np.mean(trial_recalls)),
        "std_recall":             float(np.std(trial_recalls)),
        "nq":                     nq,
        "query_batch_size":       batch_size,
    }

def run_experiment(cfg_path, out_dir):
    logging.basicConfig(level=logging.INFO)
    cfg = common_utils.load_config(cfg_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["dataset"]
    batch_sizes = ds_cfg["query_batch_sizes"]
    run_params = {
        "num_trials":            cfg.get("trials", 3),
        "num_warmup":            cfg.get("warmup", 1),
        "k_val":                 ds_cfg["k"],
        "force_rebuild_indices": cfg.get("force_rebuild", False),
        "force_overwrite":       cfg.get("overwrite", False),
        "enable_glances":        cfg.get("enable_glances_monitoring", False),
    }

    # prepare directories
    process_logs_dir = out / "process_logs"; process_logs_dir.mkdir(exist_ok=True)
    index_store_dir  = out / "indices";      index_store_dir.mkdir(exist_ok=True)

    # load existing results if present
    results_csv = out / cfg["output"]["results_csv"]
    print(results_csv)
    if results_csv.exists() and not run_params["force_overwrite"]:
        existing_df = pd.read_csv(results_csv)
        done_pairs = set(zip(existing_df["index"], existing_df["query_batch_size"]))
        print(done_pairs)
    else:
        print("No existing results found")
        existing_df = None
        done_pairs = set()

    all_rows = []

    # === BUILD TASKS ===
    for idx_cfg in cfg["indexes"]:
        name = idx_cfg["name"]
        idx_path = Path(idx_cfg.get("index_file", index_store_dir/f"{name}.bin"))
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        build_log = process_logs_dir / f"{name}_build.log"
        build_env = idx_cfg.get("build_env_vars", {})
        timeout   = idx_cfg.get("process_timeout", cfg.get("default_process_timeout", 7200))

        res = common_utils.run_operation_in_process(
            task_build_index,
            (idx_cfg, ds_cfg, run_params, idx_path),
            env_vars=build_env,
            log_file_path=str(build_log),
            process_name=f"BuildTask_{name}",
            timeout_seconds=timeout,
            enable_glances=run_params["enable_glances"]
        )
        data = res.get("data", {})
        if res.get("status")!="success" or data.get("error"):
            logger.error(f"[BUILD FAILED] {name}: {data.get('error','unknown')}")
        else:
            logger.info(f"[BUILD OK] {name} → {data['index_file_path']}")

    # === SEARCH TASKS ===
    for bs in batch_sizes:
        for idx_cfg in cfg["indexes"]:
            name = idx_cfg["name"]
            if (name, bs) in done_pairs:
                logger.info(f"Skipping existing result for {name}, batch={bs}")
                continue

            idx_path = Path(idx_cfg.get("index_file", index_store_dir/f"{name}.bin"))
            search_log = process_logs_dir / f"{name}_search_bs{bs}.log"
            search_env = idx_cfg.get("search_env_vars", {})
            timeout    = idx_cfg.get("process_timeout", cfg.get("default_process_timeout", 7200))

            res = common_utils.run_operation_in_process(
                task_search_index,
                (idx_cfg, ds_cfg, run_params, idx_path, bs),
                env_vars=search_env,
                log_file_path=str(search_log),
                process_name=f"SearchTask_{name}_bs{bs}",
                timeout_seconds=timeout,
                enable_glances=run_params["enable_glances"]
            )

            if res.get("status")=="success" and not res.get("data",{}).get("error"):
                row = {"index": name, **res["data"]}
                logger.info(f"[SEARCH OK] {name}, batch={bs}: {row}")
            else:
                err = res.get("data",{}).get("error","proc failed")
                row = {"index": name, "query_batch_size": bs, "error": err}
                logger.error(f"[SEARCH ERR] {name}, batch={bs}: {err}")

            all_rows.append(row)

    # === MERGE & SAVE ===
    new_df = pd.DataFrame(all_rows)
    if existing_df is not None and not run_params["force_overwrite"]:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    # compute QPS
    df["QPS"] = df["nq"] * 1000.0 / df["mean_total_latency_ms"]
    df.to_csv(results_csv, index=False)
    logger.info(f"Wrote results to {results_csv}")

    # === PLOT ===
    plt.figure(figsize=(8,6))
    for name, grp in df.groupby("index"):
        grp = grp.sort_values("query_batch_size")
        plt.errorbar(
            grp["query_batch_size"],
            grp["QPS"],
            marker="o",
            label=name
        )
    plt.xscale("log", base=2)
    plt.xlabel("Query Batch Size")
    plt.ylabel("QPS")
    plt.title(f"QPS vs Batch Size ({ds_cfg['name']}, k={ds_cfg['k']})")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout(rect=[0,0,0.8,1])
    plot_path = out / f"qps_vs_batch_{ds_cfg['name']}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved plot to {plot_path}")