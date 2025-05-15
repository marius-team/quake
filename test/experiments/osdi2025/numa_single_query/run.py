#!/usr/bin/env python3
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake import QuakeWrapper

INDEX_CLASSES = {
    "Quake": QuakeWrapper,
    "IVF": FaissIVF,
    # Extend as needed
}

def build_and_save_index(index_class, build_params, base_vecs, index_file):
    idx = index_class()
    params = dict(build_params)
    if "nc" not in params:
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

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{ds['name']}'...")
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
            print(f"Unknown index type: {idx_type}")
            continue

        idx_file = out_dir / f"{idx_name}_index.bin"
        if not idx_file.exists():
            print(f"Building {idx_name} index...")
            build_and_save_index(IndexClass, build_params, base_vecs, idx_file)
            print(f"Index saved at {idx_file}")

        print(f"\n--- {idx_name} ---")

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

        nprobe = search_params.get("nprobe", 100)

        # warm-up
        for i in range(min(warmup, len(queries))):
            q = queries[i].unsqueeze(0).float()
            idx.search(q, k, batched_scan=False, nprobe=nprobe)

        trial_means = []
        for t in range(trials):
            lats = []
            for q_vec in queries:
                q = q_vec.unsqueeze(0).float()
                res = idx.search(
                    q, k,
                    batched_scan=False,
                    nprobe=nprobe
                )
                ti = getattr(res, "timing_info", None)
                if ti and hasattr(ti, "total_time_ns"):
                    lats.append(ti.total_time_ns / 1e6)
                elif hasattr(res, "latency_ms"):
                    lats.append(res.latency_ms)
                else:
                    raise RuntimeError("No timing info found in search result")
            mean_t = float(np.mean(lats))
            trial_means.append(mean_t)
            print(f" Trial {t+1}/{trials}: {mean_t:.2f} ms")

        mean_lat = float(np.mean(trial_means))
        std_lat  = float(np.std(trial_means))
        # Pull out n_workers for grouping if present, else use name
        n_workers_val = build_params.get("num_workers", None)
        all_rows.append({
            "index": idx_name,
            "n_workers": n_workers_val,
            "mean_latency_ms": mean_lat,
            "std_latency_ms":  std_lat,
        })

    # Save CSV
    df = pd.DataFrame(all_rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\nResults written to {out_csv}")

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
    plt.title("Single‑Query Latency (per-index)")
    plt.legend()
    plot_file = out_dir / f"{out_csv.stem}_latency.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")