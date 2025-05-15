#!/usr/bin/env python
"""
NUMA single‑query latency experiment for Quake.
Reads a config.yaml, writes results.csv to output_dir, then plots
thread‑vs‑latency with error bars for NUMA/no‑NUMA.
"""

import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper

def run_experiment(cfg_path: str, output_dir: str):
    # 1) Load config
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ds           = cfg["dataset"]
    n_workers    = cfg["n_workers"]
    nprobe       = ds["nprobe"]
    k            = ds["k"]
    num_q        = ds["num_queries"]
    trials       = cfg.get("trials", 5)      # default 5
    warmup       = cfg.get("warmup", 10)     # default 10
    csv_name     = cfg.get("output", {}).get("results_csv", "numa_results.csv")

    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load data
    print(f"Loading dataset '{ds['name']}'...")
    base_vecs, queries, _ = load_dataset(ds["name"])
    queries = queries[:num_q]

    # 3) Build index once
    idx_file = out_dir / "quake_index.bin"
    if not idx_file.exists():
        print("Building Quake index...")
        idx = QuakeWrapper()
        nc = int(np.sqrt(base_vecs.shape[0]))
        idx.build(base_vecs, nc=nc, metric="l2", num_workers=0)
        idx.save(str(idx_file))
        print(f"Index saved at {idx_file}")


    # 4) Benchmark: warmup + trials
    rows = []
    for use_numa in [False, True]:
        mode = "NUMA" if use_numa else "no‑NUMA"
        print(f"\n--- {mode} benchmarking ---")

        for n_worker in n_workers:

            idx_mode = QuakeWrapper()
            idx_mode.load(str(idx_file), use_numa=use_numa, same_core=True, num_workers=n_worker)

            print(f"Threads: {n_worker}")
            # warm‑up
            for i in range(min(warmup, len(queries))):
                q = queries[i].unsqueeze(0).float()
                idx_mode.search(q, k, batched_scan=False, nprobe=nprobe, n_threads=n_worker)

            trial_means = []
            for t in range(trials):
                lats = []
                for q_vec in queries:
                    q = q_vec.unsqueeze(0).float()
                    res = idx_mode.search(
                        q, k,
                        batched_scan=False,
                        nprobe=nprobe,
                    )
                    ti = res.timing_info
                    lats.append(ti.total_time_ns / 1e6)
                mean_t = float(np.mean(lats))
                trial_means.append(mean_t)
                print(f" Trial {t+1}/{trials}: {mean_t:.2f} ms")

            mean_lat = float(np.mean(trial_means))
            std_lat  = float(np.std(trial_means))
            rows.append({
                "numa_enabled": use_numa,
                "n_worker":      n_worker,
                "mean_latency_ms": mean_lat,
                "std_latency_ms":  std_lat,
            })

    # 5) Save CSV
    df = pd.DataFrame(rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\nResults written to {out_csv}")

    # 6) Plot with error bars
    plt.figure()
    for flag in df["numa_enabled"].unique():
        subset = df[df["numa_enabled"] == flag]
        label = "NUMA" if flag else "no‑NUMA"
        plt.errorbar(
            subset["n_worker"],
            subset["mean_latency_ms"],
            yerr=subset["std_latency_ms"],
            marker="o",
            capsize=5,
            label=label
        )
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel("Mean Latency (ms)")
    plt.title("NUMA Single‑Query Latency")
    plt.legend()
    plot_file = out_dir / f"{out_csv.stem}_latency.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")