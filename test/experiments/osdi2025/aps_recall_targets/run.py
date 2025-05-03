#!/usr/bin/env python3
"""
Adaptive‐Partition‐Scanning Recall‐Target Runner
Supports three methods: Oracle, FixedNProbe, APS
"""

import time
import yaml
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall

logger = logging.getLogger(__name__)

def run_experiment(cfg_path: str, output_dir: str):
    # 1) Load config + setup output dir
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    # 2) Load data
    data_dir = cfg["dataset"].get("path", "")
    vecs, queries, gt = load_dataset(cfg["dataset"]["name"], data_dir)


    nq = cfg["experiment"]["nq"]
    queries, gt = queries[:nq], gt[:nq]

    # 3) Build or load index
    build_params = IndexBuildParams()
    build_params.nlist = cfg["index"]["nlist"]
    build_params.metric = cfg["index"]["metric"]
    build_params.num_workers = cfg["experiment"]["n_workers"]

    idx_dir  = Path(cfg["paths"]["index_dir"]); idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / f"{cfg['dataset']['name']}_ivf{build_params.nlist}.index"

    idx = QuakeIndex()
    idx.build(vecs, torch.arange(len(vecs)), build_params)
    logger.info(f"Loaded index → {idx_path} (workers={cfg['experiment']['n_workers']})")

    # 4) Run each method × recall_target
    records = []
    k = cfg["experiment"]["k"]
    max_nlist = build_params.nlist

    for method in cfg["methods"]:
        for rt in cfg["experiment"]["recall_targets"]:
            logger.info(f"Method={method} RecallTarget={rt}")
            per_query = []  # collect (nprobe, recall, time_ms) tuples

            if method == "Oracle":
                # per‐query binary search
                for i, q in enumerate(queries):
                    lo, hi, best = 1, max_nlist, max_nlist
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        sp = SearchParams()
                        sp.nprobe = mid
                        sp.k = k
                        sp.recall_target = -1
                        res = idx.search(q.unsqueeze(0), sp)
                        rec = compute_recall(res.ids, gt[i].unsqueeze(0), k).item()
                        if rec >= rt:
                            best = mid
                            hi = mid - 1
                        else:
                            lo = mid + 1

                    # time at best
                    t0 = time.time()
                    sp = SearchParams()
                    sp.nprobe = best
                    sp.k = k
                    sp.recall_target = -1
                    res = idx.search(q.unsqueeze(0), sp)
                    rec = compute_recall(res.ids, gt[i].unsqueeze(0), k).item()
                    elapsed = (time.time() - t0) * 1000.0
                    per_query.append((best, rec, elapsed))

            elif method == "FixedNProbe":
                # find one global nprobe
                lo, hi, best = 1, max_nlist, max_nlist
                while lo <= hi:
                    mid = (lo + hi) // 2
                    sp = SearchParams(nprobe=mid, k=k, recall_target=-1)
                    ids = []
                    for q in queries:
                        ids.append(idx.search(q.unsqueeze(0), sp).ids)
                    avg_rec = compute_recall(torch.cat(ids,0), gt, k).mean().item()
                    if avg_rec >= rt:
                        best = mid
                        hi = mid - 1
                    else:
                        lo = mid + 1

                # then time each query at best
                for q in queries:
                    t0 = time.time()
                    sp = SearchParams(nprobe=best, k=k, recall_target=-1)
                    res = idx.search(q.unsqueeze(0), sp)
                    elapsed = (time.time() - t0) * 1000.0
                    rec = compute_recall(res.ids, gt[0:1], k).item()  # single recall
                    per_query.append((best, rec, elapsed))

            elif method == "APS":
                # one‐shot APS per query
                for q, true in zip(queries, gt):
                    sp = SearchParams()
                    sp.nprobe = -1
                    sp.k = k
                    sp.recall_target = rt
                    sp.recompute_threshold = cfg["experiment"]["recompute_ratio"]
                    sp.use_precomputed = cfg["experiment"]["use_precompute"]
                    sp.initial_search_fraction = cfg["experiment"]["initial_search_fraction"]
                    t0 = time.time()
                    res = idx.search(q.unsqueeze(0), sp)
                    elapsed = (time.time() - t0) * 1000.0
                    rec = compute_recall(res.ids, true.unsqueeze(0), k).item()
                    print(f"recall: {rec:.4f} (target={rt})")
                    per_query.append((res.timing_info.partitions_scanned, rec, elapsed))
            else:
                raise ValueError(f"Unknown method: {method}")

            # 5) Aggregate
            arr = np.array(per_query, dtype=float)  # shape (nq,3)
            records.append({
                "Method":       method,
                "RecallTarget": rt,
                "mean_nprobe":  arr[:,0].mean(),
                "std_nprobe":   arr[:,0].std(),
                "mean_recall":  arr[:,1].mean(),
                "std_recall":   arr[:,1].std(),
                "mean_time_ms": arr[:,2].mean(),
                "std_time_ms":  arr[:,2].std(),
            })

    # 6) Save CSV
    df = pd.DataFrame(records)
    out_csv = out / "aps_recall_targets_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Summary → {out_csv}")

    # 7) Plot: recall-target vs time, one line per method
    fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True)
    for m in df["Method"].unique():
        grp = df[df["Method"]==m]
        axes[0].errorbar(grp["RecallTarget"], grp["mean_time_ms"],
                     yerr=grp["std_time_ms"], marker="o", capsize=4, label=m)
        axes[1].errorbar(grp["RecallTarget"], grp["mean_recall"],
                     yerr=grp["std_recall"], marker="o", capsize=4, label=m)
        axes[2].errorbar(grp["RecallTarget"], grp["mean_nprobe"],
                     yerr=grp["std_nprobe"], marker="o", capsize=4, label=m)
    axes[0].set_xlabel("Recall Target")
    axes[0].set_xlabel("Mean Query Time (ms)")
    axes[0].set_title("APS‑Experiment: Recall Target vs Time")
    axes[0].legend()

    recall_targets = cfg["experiment"]["recall_targets"]

    axes[1].plot(recall_targets, recall_targets, "k--", label="Target")
    axes[1].set_xlabel("Recall Target")
    axes[1].set_ylabel("Mean Recall")
    axes[1].set_title("APS‑Experiment: Recall Target vs Recall")

    axes[2].set_xlabel("Recall Target")
    axes[2].set_ylabel("Mean NProbe")
    axes[2].set_yscale("log")
    axes[2].set_title("APS‑Experiment: Recall Target vs NProbe")

    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plot_path = out / "aps_recall_vs_time.png"
    plt.savefig(plot_path)
    logger.info(f"Plot → {plot_path}")