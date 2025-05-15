#!/usr/bin/env python3
"""
NUMA multi‐query QPS experiment for Quake.

Sweeps NUMA on/off, worker counts, batched_scan on/off, and batch sizes;
records mean/std QPS and produces a simplified batched_scan QPS plot.
"""

import yaml
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper

def run_experiment(cfg_path: str, output_dir: str):
    cfg         = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)
    csv_name    = cfg.get("output", {}).get("results_csv", "results.csv")
    out_csv = out / csv_name

    # check if the results already exist
    if not Path(out_csv).exists():

        # Load config
        ds          = cfg["dataset"]
        n_workers   = cfg["n_workers"]
        batched_opts= cfg.get("batch_options", [False, True])
        batch_sizes = cfg.get("batch_sizes", [ds["num_queries"]])
        num_q       = ds["num_queries"]
        k           = ds["k"]
        nprobe      = ds["nprobe"]
        trials      = cfg.get("trials", 5)
        warmup      = cfg.get("warmup", 2)


        # Load data
        print(f"Loading {ds['name']} ({num_q} queries)…")
        base_vecs, queries_data, _ = load_dataset(ds["name"])
        queries = (
            torch.from_numpy(queries_data[:num_q]).float()
            if isinstance(queries_data, np.ndarray)
            else queries_data[:num_q].float()
        )

        # Build index if needed
        idx_file = out / "quake_index.bin"
        if not idx_file.exists():
            print("Building index…")
            builder = QuakeWrapper()
            vecs = (
                torch.from_numpy(base_vecs).float()
                if isinstance(base_vecs, np.ndarray)
                else base_vecs.float()
            )
            nc = int(np.sqrt(vecs.shape[0]))
            builder.build(vecs, nc=nc, metric="l2", num_workers=0)
            builder.save(str(idx_file))
            print("Index saved.")

        records = []
        for use_numa in [False]:
            for nw in n_workers:
                for bs in batched_opts:
                    for batch_size in batch_sizes:
                        print(f"NUMA={use_numa}, workers={nw}, batched={bs}, batch={batch_size}")
                        idxm = QuakeWrapper()
                        idxm.load(str(idx_file), use_numa=use_numa, same_core=True, num_workers=nw)
                        n_threads = 16 if nw == 0 else 1

                        # Warmup
                        for _ in range(min(warmup,1)):
                            if bs:
                                for i in range(0, num_q, batch_size):
                                    idxm.search(queries[i:i+batch_size], k,
                                                batched_scan=bs,
                                                nprobe=nprobe,
                                                n_threads=n_threads)
                            else:
                                idxm.search(queries, k,
                                            batched_scan=bs,
                                            nprobe=nprobe,
                                            n_threads=n_threads)

                        # Trials → QPS
                        qps_vals = []
                        for t in range(trials):
                            start = time.time()
                            if bs:
                                for i in range(0, num_q, batch_size):
                                    idxm.search(queries[i:i+batch_size], k,
                                                batched_scan=bs,
                                                nprobe=nprobe,
                                                n_threads=n_threads)
                            else:
                                idxm.search(queries, k,
                                            batched_scan=bs,
                                            nprobe=nprobe,
                                            n_threads=n_threads)
                            elapsed = time.time() - start
                            qps_vals.append(num_q / elapsed)
                            print(f"  Trial {t+1}/{trials}: {qps_vals[-1]:.1f} QPS")

                        records.append({
                            "numa_enabled": use_numa,
                            "n_worker":     nw,
                            "batched_scan": bs,
                            "batch_size":   batch_size,
                            "mean_qps":     np.mean(qps_vals),
                            "std_qps":      np.std(qps_vals),
                        })

        # Save CSV
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        print(f"Results → {out_csv}")

    df = pd.read_csv(out_csv)

    # Simplified batched_scan QPS plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for bsz in sorted(df["batch_size"].unique()):
        for bs in (False, True):
            grp = df[(df["batch_size"] == bsz) & (df["batched_scan"] == bs)]
            if grp.empty:
                continue
            linestyle = '-' if not bs else '--'
            label = f"batch={bsz}" + ("" if not bs else " batched")
            ax.plot(
                grp["n_worker"],
                grp["mean_qps"],
                marker='o',
                linestyle=linestyle,
                label=label
            )
    ax.set_xscale("symlog", base=2)
    ax.set_yscale("log",    base=2)
    ax.set_xlabel("Workers")
    ax.set_ylabel("QPS")
    ax.legend(title="Config", fontsize="small")

    fig.suptitle("QPS vs. Workers  (solid=regular, dashed=batched)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plot_path = out / f"{out_csv.stem}_combined_qps.png"
    fig.savefig(plot_path)
    print(f"Combined plot saved to {plot_path}")

    # 7) QPS vs. batch size for 1 and 16 workers, solid=unbatched, dashed=batched
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))

    # choose which workers to show
    workers = [1, 16]
    # matplotlib will auto‑cycle colors for each worker
    for w in workers:
        for bs, ls in [(False, "-"), (True, "--")]:
            sub = df[(df["n_worker"] == w) & (df["batched_scan"] == bs)]
            if sub.empty:
                continue
            sub = sub.sort_values("batch_size")
            label = f"{w} worker{'s' if w>1 else ''}" + (" (batched)" if bs else "")
            ax2.errorbar(
                sub["batch_size"],
                sub["mean_qps"],
                yerr=sub["std_qps"],
                marker="o",
                linestyle=ls,
                capsize=4,
                label=label
            )

    ax2.set_xscale("symlog", base=2)
    ax2.set_yscale("log",    base=2)
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("QPS")
    ax2.set_title("QPS vs. Batch Size\n(solid=unbatched, dashed=batched)")
    ax2.grid(True, which="both", ls=":")
    ax2.legend(title="Configuration", fontsize="small", ncol=1)
    fig2.tight_layout()
    plot2_path = out / f"{out_csv.stem}_qps_vs_batchsize_full.png"
    fig2.savefig(plot2_path)
    print(f"Full batch‑size QPS plot saved to {plot2_path}")