#!/usr/bin/env python3
"""
NUMA Batch-Query Latency and Recall Benchmark (detailed timers)

Benchmarks a suite of ANN indexes (Quake, Faiss-IVF, SCANN, HNSW, DiskANN, SVS, etc.)
for per-query latency breakdown and recall@k under configurable build and search parameters.
Results are cached and only recomputed if the 'overwrite' flag is set in the configuration.
Outputs per-index CSVs with detailed timing info and unified CSV + plots.

Usage:
    python numa_latency_batch_experiment.py numa_latency_experiment.yaml output_dir
"""
import sys
from pathlib import Path
ing = yaml = None
import yaml
import numpy as np
torch = None
import torch
import pandas as pd
import matplotlib.pyplot as plt

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


def build_and_save(index_cls, params, vecs, path):
    """Build an index and save it to disk."""
    idx = index_cls()
    idx.build(vecs, **params)
    idx.save(str(path))


def extract_ids(res):
    """Normalize returned IDs array."""
    if hasattr(res, "ids"):
        arr = res.ids
    elif hasattr(res, "I"):
        arr = res.I
    elif hasattr(res, "indices"):
        arr = res.indices
    else:
        raise RuntimeError("Search result has no id field.")
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    return np.asarray(arr, dtype=np.int64)


def run_experiment(cfg_path: str, output_dir: str) -> None:
    # Load config
    cfg       = yaml.safe_load(Path(cfg_path).read_text())
    ds_cfg    = cfg["dataset"]
    k         = ds_cfg["k"]
    q_n       = ds_cfg["num_queries"]
    trials    = cfg.get("trials", 5)
    warmup_n  = cfg.get("warmup", 10)
    idx_cfgs  = cfg["indexes"]
    csv_name  = cfg.get("output", {}).get("results_csv", "numa_results.csv")
    overwrite = cfg.get("overwrite", False)

    out_dir = Path(output_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"[INFO] Loading dataset {ds_cfg['name']}...")
    base_vecs, queries, gt = load_dataset(ds_cfg["name"])
    queries = queries[:q_n]
    gt      = gt[:q_n] if gt is not None else None

    # Build shared Quake base index once
    quake_base_params = None
    for c in idx_cfgs:
        if c["index"] == "Quake":
            p = dict(c.get("build_params", {}))
            p.pop("num_workers", None)
            p.pop("use_numa", None)
            quake_base_params = p
            break

    quake_base_path = out_dir / "Quake_base_index.bin"
    if quake_base_params and not quake_base_path.exists():
        print("[BUILD] Quake (shared) base index...")
        build_and_save(QuakeWrapper, quake_base_params, base_vecs, quake_base_path)
        print(f"[SAVE] Quake base index → {quake_base_path}")

    # Collect results per-index
    all_rows = []
    for cfg_i in idx_cfgs:
        name     = cfg_i["name"]
        typ      = cfg_i["index"]
        b_params = dict(cfg_i.get("build_params", {}))
        s_params = dict(cfg_i.get("search_params", {}))
        cls      = INDEX_CLASSES.get(typ)
        if cls is None:
            print(f"[WARN] Unknown index type '{typ}', skipping.")
            continue

        idx_csv  = out_dir / f"{name}_results.csv"
        idx_path = quake_base_path if typ == "Quake" else out_dir / f"{name}_index.bin"

        # Skip if exists
        if idx_csv.exists() and not overwrite:
            print(f"[SKIP] {name} – results exist.")
            all_rows.append(pd.read_csv(idx_csv).iloc[0].to_dict())
            continue

        # Build if needed
        if typ != "Quake" and not idx_path.exists():
            print(f"[BUILD] {name} index...")
            build_and_save(cls, b_params, base_vecs, idx_path)
            print(f"[SAVE] Index → {idx_path}")

        # Load index with params
        if typ == "Quake":
            load_kw = {k: b_params[k] for k in ("use_numa", "num_workers") if k in b_params}
            idx = QuakeWrapper(); idx.load(str(idx_path), **load_kw)
        elif typ == "DiskANN":
            idx = cls(); b_params.pop("metric", None); idx.load(str(idx_path), **b_params)
        else:
            idx = cls(); idx.load(str(idx_path))

        # Warm-up runs
        for _ in range(min(warmup_n, len(queries))):
            for q in queries:
                idx.search(q.unsqueeze(0).float(), k, **s_params)

        # Prepare accumulators for detailed timers
        child_buf_init_trials = []
        child_enq_trials      = []
        child_wait_trials     = []
        child_agg_trials      = []
        child_total_trials    = []
        parent_buf_init_trials = []
        parent_enq_trials      = []
        parent_wait_trials     = []
        parent_agg_trials      = []
        parent_total_trials    = []
        recall_trials          = []

        # Benchmark trials
        for t in range(trials):
            # per-trial lists
            buf_init = []
            enq      = []
            wait     = []
            agg      = []
            tot      = []
            p_buf    = []
            p_enq    = []
            p_wait   = []
            p_agg    = []
            p_tot    = []
            recs     = []

            for qi, q in enumerate(queries):
                res = idx.search(q.unsqueeze(0).float(), k, **s_params)
                ti = res.timing_info

                # child (worker_scan) timers
                buf_init.append(ti.buffer_init_time_ns)
                enq.append(ti.job_enqueue_time_ns)
                wait.append(ti.job_wait_time_ns)
                agg.append(ti.result_aggregate_time_ns)
                tot.append(ti.total_time_ns)

                # parent timers if present
                pi = getattr(ti, "parent_info", None)
                if pi:
                    p_buf.append(pi.buffer_init_time_ns)
                    p_enq.append(pi.job_enqueue_time_ns)
                    p_wait.append(pi.job_wait_time_ns)
                    p_agg.append(pi.result_aggregate_time_ns)
                    p_tot.append(pi.total_time_ns)
                else:
                    p_buf.append(0)
                    p_enq.append(0)
                    p_wait.append(0)
                    p_agg.append(0)
                    p_tot.append(0)

                # recall on first trial
                if gt is not None and t == 0:
                    preds = extract_ids(res)[0]
                    recs.append(float(compute_recall(torch.tensor([preds]), gt[qi:qi+1], k).item()))

            # compute per-trial means (convert ns to ms where appropriate)
            child_buf_init_trials.append(np.mean(buf_init)/1e6)
            child_enq_trials     .append(np.mean(enq)     /1e6)
            child_wait_trials    .append(np.mean(wait)    /1e6)
            child_agg_trials     .append(np.mean(agg)     /1e6)
            child_total_trials   .append(np.mean(tot)     /1e6)
            parent_buf_init_trials.append(np.mean(p_buf)/1e6)
            parent_enq_trials     .append(np.mean(p_enq)/1e6)
            parent_wait_trials    .append(np.mean(p_wait)/1e6)
            parent_agg_trials     .append(np.mean(p_agg)/1e6)
            parent_total_trials   .append(np.mean(p_tot)/1e6)
            recall_trials.append(np.mean(recs) if recs else np.nan)

            print(f" [{name} trial {t+1}/{trials}] "
                  f"child_tot={child_total_trials[-1]:.2f}ms, "
                  f"parent_tot={parent_total_trials[-1]:.2f}ms, "
                  f"R@{k}={recall_trials[-1]:.4f}")

        # assemble result row
        row = {
            "index":               name,
            "n_workers":           b_params.get("num_workers", np.nan),
            "recall_at_%d"%k:      float(np.mean(recall_trials)),
            "child_buffer_init_ms": float(np.mean(child_buf_init_trials)),
            "child_enqueue_ms":     float(np.mean(child_enq_trials)),
            "child_wait_ms":        float(np.mean(child_wait_trials)),
            "child_aggregate_ms":   float(np.mean(child_agg_trials)),
            "child_total_ms":       float(np.mean(child_total_trials)),
            "parent_buffer_init_ms":float(np.mean(parent_buf_init_trials)),
            "parent_enqueue_ms":    float(np.mean(parent_enq_trials)),
            "parent_wait_ms":       float(np.mean(parent_wait_trials)),
            "parent_aggregate_ms":  float(np.mean(parent_agg_trials)),
            "parent_total_ms":      float(np.mean(parent_total_trials)),
        }
        all_rows.append(row)
        pd.DataFrame([row]).to_csv(idx_csv, index=False)

    # write unified CSV
    df = pd.DataFrame(all_rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\n[RESULT] Unified results → {out_csv}")

    # Plot child total latency
    plt.figure()
    for nm in df["index"].unique():
        sub = df[df["index"]==nm]
        lbl = (nm if sub["n_workers"].isnull().all()
               else f"{nm} (nw={int(sub['n_workers'].iloc[0])})")
        plt.errorbar(sub["n_workers"], sub["child_total_ms"],
                     yerr=df[df['index']==nm]["child_total_ms"].std(),
                     marker="o", capsize=5, label=lbl)
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel("Mean Child Total Latency (ms)")
    plt.title("Child Latency Breakdown (per-index)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/ f"{out_csv.stem}_child_breakdown.png")
    print(f"[PLOT] Child breakdown → {out_dir/ f'{out_csv.stem}_child_breakdown.png'}")

    # Plot recall
    plt.figure()
    for nm in df["index"].unique():
        sub = df[df["index"]==nm]
        lbl = (nm if sub["n_workers"].isnull().all()
               else f"{nm} (nw={int(sub['n_workers'].iloc[0])})")
        plt.plot(sub["n_workers"], sub[f"recall_at_{k}"], marker="o", label=lbl)
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel(f"Recall@{k}")
    plt.title(f"Recall@{k} (per-index)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/ f"{out_csv.stem}_recall.png")
    print(f"[PLOT] Recall plot → {out_dir/ f'{out_csv.stem}_recall.png'}")