#!/usr/bin/env python3
"""
NUMA Single-Query Latency Benchmark

Benchmarks a suite of approximate nearest neighbor (ANN) indexes (Quake, Faiss-IVF, SCANN, HNSW, DiskANN, SVS, etc.)
for single-query latency under configurable build and search parameters. Results are cached and only recomputed if
the 'overwrite' flag is set in the configuration. Outputs a unified CSV and a latency plot per index.

Usage:
    python numa_latency_experiment.py numa_latency_experiment.yaml output_dir
"""

import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
from quake.index_wrappers.quake import QuakeWrapper

try:
    from quake.index_wrappers.scann import Scann
except ImportError:
    raise ImportError("SCANN wrapper not available. Please install the required package.")
try:
    from quake.index_wrappers.diskann import DiskANNDynamic
except ImportError:
    raise ImportError("DiskANN wrapper not available. Please install the required package.")
try:
    from quake.index_wrappers.vamana import Vamana
except ImportError:
    raise ImportError("SVS wrapper not available. Please install the required package.")

INDEX_CLASSES = {
    "Quake": QuakeWrapper,
    "IVF": FaissIVF,
    "SCANN": Scann,
    "HNSW": FaissHNSW,
    "DiskANN": DiskANNDynamic,
    "SVS": Vamana,
}

def build_and_save_index(index_class, build_params, base_vecs, index_file):
    idx = index_class()
    params = dict(build_params)
    if "nc" not in params and "num_leaves" not in params:
        params["nc"] = int(np.sqrt(base_vecs.shape[0]))
    idx.build(base_vecs, **params)
    idx.save(str(index_file))

def run_experiment(cfg_path: str, output_dir: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ds           = cfg["dataset"]
    queries_n    = ds["num_queries"]
    k            = ds["k"]
    trials       = cfg.get("trials", 5)
    warmup       = cfg.get("warmup", 10)
    indexes_cfg  = cfg["indexes"]
    csv_name     = cfg.get("output", {}).get("results_csv", "numa_results.csv")
    overwrite    = cfg.get("overwrite", False)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset '{ds['name']}'...")
    base_vecs, queries, _ = load_dataset(ds["name"])
    queries = queries[:queries_n]

    all_rows = []

    for idx_cfg in indexes_cfg:
        idx_name      = idx_cfg["name"]
        idx_type      = idx_cfg["index"]
        build_params  = idx_cfg.get("build_params", {})
        search_params = idx_cfg.get("search_params", {})
        IndexClass    = INDEX_CLASSES.get(idx_type)
        if IndexClass is None:
            print(f"[WARN] Unknown index type: {idx_type}. Skipping.")
            continue

        idx_csv = out_dir / f"{idx_name}_results.csv"
        idx_file = out_dir / f"{idx_name}_index.bin"

        if idx_csv.exists() and not overwrite:
            print(f"[SKIP] {idx_name}: Results exist. Use overwrite: true to rerun.")
            # For plotting, load mean/std for this index
            df_idx = pd.read_csv(idx_csv)
            all_rows.append(df_idx.iloc[0].to_dict())
            continue

        if not idx_file.exists():
            print(f"[BUILD] {idx_name} index...")
            build_and_save_index(IndexClass, build_params, base_vecs, idx_file)
            print(f"[SAVE] Index saved at {idx_file}")

        print(f"[RUN] {idx_name} ...")

        # Prepare load arguments per index type
        if idx_type == "Quake":
            load_kwargs = {}
            if "use_numa" in build_params:
                load_kwargs["use_numa"] = build_params["use_numa"]
            if "num_workers" in build_params:
                load_kwargs["num_workers"] = build_params["num_workers"]
            idx = IndexClass()
            idx.load(str(idx_file), **load_kwargs)
        else:
            idx = IndexClass()
            idx.load(str(idx_file))

        # Dispatch search argument keys by index type
        if idx_type == "SCANN":
            search_args = {"leaves_to_search": search_params.get("leaves_to_search", 100)}
        elif idx_type == "HNSW":
            search_args = {"ef_search": search_params.get("ef_search", 32)}
        elif idx_type == "DiskANN":
            search_args = {"complexity": search_params.get("complexity", 32)}
        elif idx_type == "SVS":
            search_args = {"search_window_size": search_params.get("search_window_size", 32)}
        else:  # IVF, Quake, others
            search_args = {"nprobe": search_params.get("nprobe", 100)}

        # Warmup
        for i in range(min(warmup, len(queries))):
            q = queries[i].unsqueeze(0).float()
            idx.search(q, k, batched_scan=False, **search_args)

        # Benchmark Trials
        trial_means = []
        for t in range(trials):
            lats = []
            for q_vec in queries:
                q = q_vec.unsqueeze(0).float()
                res = idx.search(
                    q, k,
                    batched_scan=False,
                    **search_args
                )
                ti = getattr(res, "timing_info", None)
                if ti and hasattr(ti, "total_time_ns"):
                    lats.append(ti.total_time_ns / 1e6)
                elif hasattr(res, "latency_ms"):
                    lats.append(res.latency_ms)
                else:
                    raise RuntimeError(f"No timing info found in search result for {idx_name}")
            mean_t = float(np.mean(lats))
            trial_means.append(mean_t)
            print(f" [trial {t+1}/{trials}] {mean_t:.2f} ms")

        mean_lat = float(np.mean(trial_means))
        std_lat  = float(np.std(trial_means))
        n_workers_val = build_params.get("num_workers", None)
        row = {
            "index": idx_name,
            "n_workers": n_workers_val,
            "mean_latency_ms": mean_lat,
            "std_latency_ms":  std_lat,
        }
        all_rows.append(row)
        pd.DataFrame([row]).to_csv(idx_csv, index=False)

    # Save Unified CSV
    df = pd.DataFrame(all_rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\n[RESULT] Results written to {out_csv}")

    # Plot
    plt.figure()
    for idx_name in df["index"].unique():
        subset = df[df["index"] == idx_name]
        label = f"{idx_name}" if subset["n_workers"].isnull().all() else f"{idx_name} (n_workers={subset['n_workers'].iloc[0]})"
        plt.errorbar(
            subset["n_workers"] if "n_workers" in subset else subset.index,
            subset["mean_latency_ms"],
            yerr=subset["std_latency_ms"],
            marker="o",
            capsize=5,
            label=label
        )
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel("Mean Latency (ms)")
    plt.title("Single-Query Latency (per-index)")
    plt.legend()
    plot_file = out_dir / f"{out_csv.stem}_latency.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"[PLOT] Saved to {plot_file}")
