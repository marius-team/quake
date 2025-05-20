#!/usr/bin/env python3
"""
NUMA Batch-Query Latency and Recall Benchmark (with detailed child/parent timers)

Benchmarks a suite of ANN indexes (Quake, Faiss-IVF, SCANN, HNSW, DiskANN, SVS, etc.)
for batch-query latency, per-phase breakdown, and recall@k under configurable parameters.
Results are cached and only recomputed if 'overwrite' is set.
Outputs per-index CSVs with detailed timers, a unified CSV, and plots.

Usage:
    python numa_latency_batch_experiment.py nna_latency_experiment.yaml output_dir
"""
import sys
from pathlib import Path
import yaml
import numpy as np
torch = None
import torch
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from quake.utils import compute_recall
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
from quake.index_wrappers.quake import QuakeWrapper

# Optional wrappers
try:
    from quake.index_wrappers.scann import Scann
except ImportError:
    Scann = None
try:
    from quake.index_wrappers.diskann import DiskANNDynamic
except ImportError:
    DiskANNDynamic = None
try:
    from quake.index_wrappers.vamana import Vamana
except ImportError:
    Vamana = None

INDEX_CLASSES = {
    "Quake":   QuakeWrapper,
    "IVF":     FaissIVF,
    "SCANN":   Scann,
    "HNSW":    FaissHNSW,
    "DiskANN": DiskANNDynamic,
    "SVS":     Vamana,
}


def build_and_save_index(index_class, build_params, base_vecs, index_file):
    """Build and save an index to disk."""
    idx = index_class()
    idx.build(base_vecs, **build_params)
    idx.save(str(index_file))


def extract_ids(res):
    """Normalize returned prediction IDs to a NumPy array."""
    if hasattr(res, "ids"):
        arr = res.ids
    elif hasattr(res, "I"):
        arr = res.I
    elif hasattr(res, "indices"):
        arr = res.indices
    else:
        raise RuntimeError("Cannot extract predicted indices from search result.")
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    return np.asarray(arr)


def run_experiment(cfg_path: str, output_dir: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ds      = cfg["dataset"]
    queries_n = ds["num_queries"]
    k       = ds["k"]
    trials  = cfg.get("trials", 5)
    warmup  = cfg.get("warmup", 10)
    indexes_cfg = cfg["indexes"]
    csv_name = cfg.get("output", {}).get("results_csv", "numa_results.csv")
    overwrite = cfg.get("overwrite", False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset '{ds['name']}'...")
    base_vecs, queries, gt = load_dataset(ds["name"])
    queries = queries[:queries_n]
    gt = gt[:queries_n] if gt is not None else None

    # --- Build shared Quake base index once ---
    quake_base_params = None
    quake_base_index_file = out_dir / "Quake_base_index.bin"
    for idx_cfg in indexes_cfg:
        if idx_cfg["index"] == "Quake":
            quake_base_params = dict(idx_cfg.get("build_params", {}))
            quake_base_params.pop("num_workers", None)
            quake_base_params.pop("use_numa", None)
            break
    if quake_base_params and not quake_base_index_file.exists():
        print(f"[BUILD] Quake (shared) base index ...")
        build_and_save_index(QuakeWrapper, quake_base_params, base_vecs, quake_base_index_file)
        print(f"[SAVE] Quake base index saved at {quake_base_index_file}")

    all_rows = []

    for idx_cfg in indexes_cfg:
        name         = idx_cfg["name"]
        idx_type     = idx_cfg["index"]
        build_params = dict(idx_cfg.get("build_params", {}))
        search_params= dict(idx_cfg.get("search_params", {}))
        IndexClass   = INDEX_CLASSES.get(idx_type)
        if IndexClass is None:
            print(f"[WARN] Unknown index type: {idx_type}. Skipping.")
            continue

        idx_csv = out_dir / f"{name}_results.csv"
        idx_file = quake_base_index_file if idx_type == "Quake" else out_dir / f"{name}_index.bin"

        if idx_csv.exists() and not overwrite:
            print(f"[SKIP] {name}: Results exist. Use overwrite: true to rerun.")
            all_rows.append(pd.read_csv(idx_csv).iloc[0].to_dict())
            continue

        # Build non-Quake if necessary
        if idx_type != "Quake" and not idx_file.exists():
            print(f"[BUILD] {name} index...")
            build_and_save_index(IndexClass, build_params, base_vecs, idx_file)
            print(f"[SAVE] Index saved at {idx_file}")

        # Load index
        if idx_type == "Quake":
            load_kwargs = {k: build_params[k] for k in ("use_numa","num_workers") if k in build_params}
            idx = QuakeWrapper(); idx.load(str(idx_file), **load_kwargs)
        elif idx_type == "DiskANN":
            idx = IndexClass(); build_params.pop("metric", None); idx.load(str(idx_file), **build_params)
        else:
            idx = IndexClass(); idx.load(str(idx_file))

        # Warmup (batch)
        for _ in range(min(warmup,1)):
            idx.search(queries, k, **search_params)

        # Prepare per-phase accumulators
        child_buf_init_trials = []
        child_copy_query_trials = []
        child_enqueue_trials  = []
        child_wait_trials     = []
        child_agg_trials      = []
        child_total_trials    = []
        parent_buf_init_trials= []
        parent_copy_query_trials = []
        parent_enqueue_trials = []
        parent_wait_trials    = []
        parent_agg_trials     = []
        parent_total_trials   = []
        recall_trials         = []

        # Benchmark Trials (batch)
        for t in range(trials):
            res = idx.search(queries, k, **search_params)
            ti  = res.timing_info

            # Child (worker_scan) timings
            child_buf_init_trials.append(ti.buffer_init_time_ns / 1e6)
            child_copy_query_trials.append(ti.copy_query_time_ns / 1e6)
            child_enqueue_trials .append(ti.job_enqueue_time_ns   / 1e6)
            child_wait_trials    .append(ti.job_wait_time_ns      / 1e6)
            child_agg_trials     .append(ti.result_aggregate_time_ns / 1e6)
            child_total_trials   .append(ti.total_time_ns         / 1e6)

            # Parent timings
            pi = getattr(ti, "parent_info", None)
            if pi:
                parent_buf_init_trials.append(pi.buffer_init_time_ns / 1e6)
                parent_copy_query_trials.append(pi.copy_query_time_ns / 1e6)
                parent_enqueue_trials .append(pi.job_enqueue_time_ns   / 1e6)
                parent_wait_trials    .append(pi.job_wait_time_ns      / 1e6)
                parent_agg_trials     .append(pi.result_aggregate_time_ns / 1e6)
                parent_total_trials   .append(pi.total_time_ns         / 1e6)
            else:
                parent_buf_init_trials.append(0.0)
                parent_copy_query_trials.append(0.0)
                parent_enqueue_trials .append(0.0)
                parent_wait_trials    .append(0.0)
                parent_agg_trials     .append(0.0)
                parent_total_trials   .append(0.0)

            # Recall for this trial
            if gt is not None:
                preds_arr = extract_ids(res)
                recall    = float(compute_recall(preds_arr, gt, k).mean())
                recall_trials.append(recall)
                print(f" [{name} trial {t+1}/{trials}] child_total={child_total_trials[-1]:.2f} ms, parent_total={parent_total_trials[-1]:.2f} ms | recall@{k} {recall:.4f}")
            else:
                print(f" [{name} trial {t+1}/{trials}] child_total={child_total_trials[-1]:.2f} ms, parent_total={parent_total_trials[-1]:.2f} ms")

        # Aggregate metrics
        mean_lat   = float(np.mean(child_total_trials))
        std_lat    = float(np.std(child_total_trials))
        mean_recall= float(np.mean(recall_trials)) if recall_trials else np.nan
        n_workers  = build_params.get("num_workers", None)

        row = {
            "index":               name,
            "n_workers":           n_workers,
            f"recall_at_{k}":      mean_recall,
            "mean_latency_ms":     mean_lat,
            "std_latency_ms":      std_lat,
            "child_buffer_init_ms":float(np.mean(child_buf_init_trials)),
            "child_copy_query_ms": float(np.mean(child_copy_query_trials)),
            "child_enqueue_ms":    float(np.mean(child_enqueue_trials)),
            "child_wait_ms":       float(np.mean(child_wait_trials)),
            "child_aggregate_ms":  float(np.mean(child_agg_trials)),
            "child_total_ms":      float(np.mean(child_total_trials)),
            "parent_buffer_init_ms":float(np.mean(parent_buf_init_trials)),
            "parent_copy_query_ms": float(np.mean(parent_copy_query_trials)),
            "parent_enqueue_ms":   float(np.mean(parent_enqueue_trials)),
            "parent_wait_ms":      float(np.mean(parent_wait_trials)),
            "parent_aggregate_ms": float(np.mean(parent_agg_trials)),
            "parent_total_ms":     float(np.mean(parent_total_trials)),
        }
        all_rows.append(row)
        pd.DataFrame([row]).to_csv(idx_csv, index=False)

    # Save unified CSV
    df = pd.DataFrame(all_rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\n[RESULT] Results written to {out_csv}")

    # Plot batch total latency (child only)
    plt.figure()
    for idx_name in df["index"].unique():
        sub = df[df["index"]==idx_name]
        label = f"{idx_name}" if pd.isna(sub["n_workers"]).all() else f"{idx_name} (n_workers={int(sub['n_workers'].iloc[0])})"
        plt.errorbar(
            sub["n_workers"], sub["child_total_ms"],
            yerr=sub["child_total_ms"].std(),
            marker="o", capsize=5, label=label
        )
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel("Mean Child Batch Latency (ms)")
    plt.title("Batch Query Child Total Latency")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{out_csv.stem}_batch_latency.png")
    print(f"[PLOT] Saved to {out_dir / f'{out_csv.stem}_batch_latency.png'}")

    # Plot recall
    if f"recall_at_{k}" in df.columns:
        plt.figure()
        for idx_name in df["index"].unique():
            sub = df[df["index"]==idx_name]
            label = f"{idx_name}" if pd.isna(sub["n_workers"]).all() else f"{idx_name} (n_workers={int(sub['n_workers'].iloc[0])})"
            plt.plot(sub["n_workers"], sub[f"recall_at_{k}"], marker="o", label=label)
        plt.xscale("symlog", base=2)
        plt.xlabel("Num Workers")
        plt.ylabel(f"Recall@{k}")
        plt.title(f"Recall@{k} (batch)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"{out_csv.stem}_batch_recall.png")
        print(f"[PLOT] Recall plot saved to {out_dir / f'{out_csv.stem}_batch_recall.png'}")