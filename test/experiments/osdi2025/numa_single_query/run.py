"""
NUMA Single-Query Latency and Recall Benchmark (detailed timers + parent)

Tracks these phases, for both the worker_scan and its parent search:
  - buffer_init_time_ns
  - job_enqueue_time_ns
  - job_wait_time_ns
  - result_aggregate_time_ns
  - total_time_ns

Averages per‐query timers within each trial, then across trials,
and writes them (in ms) to the per‐index CSV.
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from quake.utils import compute_recall
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
from quake.index_wrappers.quake import QuakeWrapper
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
    """Build and persist an index."""
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
    print(f"[INFO] loading dataset {ds_cfg['name']} …")
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
        print("[BUILD] Quake (shared) base index …")
        build_and_save(QuakeWrapper, quake_base_params, base_vecs, quake_base_path)
        print(f"[SAVE] Quake base index → {quake_base_path}")

    results = []
    for cfg_i in idx_cfgs:
        name     = cfg_i["name"]
        typ      = cfg_i["index"]
        b_params = dict(cfg_i.get("build_params", {}))
        s_params = dict(cfg_i.get("search_params", {}))
        cls      = INDEX_CLASSES.get(typ)
        if cls is None:
            print(f"[WARN] unknown index type '{typ}' – skipping.")
            continue

        idx_csv  = out_dir / f"{name}_results.csv"
        idx_path = quake_base_path if typ == "Quake" else out_dir / f"{name}_index.bin"

        if idx_csv.exists() and not overwrite:
            print(f"[SKIP] {name} – results exist.")
            results.append(pd.read_csv(idx_csv).iloc[0].to_dict())
            continue

        # Build non-Quake if needed
        if typ != "Quake" and not idx_path.exists():
            print(f"[BUILD] {name} index …")
            build_and_save(cls, b_params, base_vecs, idx_path)
            print(f"[SAVE] index → {idx_path}")

        # Load index
        if typ == "Quake":
            load_kw = {k: b_params[k] for k in ("use_numa", "num_workers") if k in b_params}
            idx = QuakeWrapper(); idx.load(str(idx_path), **load_kw)
        elif typ == "DiskANN":
            idx = cls(); b_params.pop("metric", None); idx.load(str(idx_path), **b_params)
        else:
            idx = cls(); idx.load(str(idx_path))

        # Dispatch search parameters
        if typ == "SCANN":
            search_kw = {"leaves_to_search": s_params.get("leaves_to_search", 100)}
        elif typ == "HNSW":
            search_kw = {"ef_search": s_params.get("ef_search", 32)}
        elif typ == "DiskANN":
            search_kw = {"complexity": s_params.get("complexity", 32)}
        elif typ == "SVS":
            search_kw = {"search_window_size": s_params.get("search_window_size", 32)}
        else:
            search_kw = {"nprobe": s_params.get("nprobe", 100)}

        # Warm-up
        for q in queries[:min(warmup_n, len(queries))]:
            idx.search(q.unsqueeze(0).float(), k, **search_kw)

        # Prepare accumulators
        child_lat_trials = []
        parent_lat_trials = []
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
        first_trial_recalls    = []

        for t in range(trials):
            # per-query lists
            child_lats = []
            parent_lats = []
            child_buf_init = []
            child_enq      = []
            child_wait     = []
            child_agg      = []
            child_total    = []
            parent_buf_init = []
            parent_enq      = []
            parent_wait     = []
            parent_agg      = []
            parent_total    = []
            recalls = []

            for qi, q in enumerate(queries):
                res = idx.search(q.unsqueeze(0).float(), k, **search_kw)
                ti  = res.timing_info

                # child latency
                child_lats.append(ti.total_time_ns / 1e6)
                child_buf_init.append(ti.buffer_init_time_ns)
                child_enq.append(   ti.job_enqueue_time_ns)
                child_wait.append(  ti.job_wait_time_ns)
                child_agg.append(   ti.result_aggregate_time_ns)
                child_total.append( ti.total_time_ns)

                # parent latency (if exists)
                pi = getattr(ti, "parent_info", None)
                if pi and hasattr(pi, "total_time_ns"):
                    parent_lats.append(pi.total_time_ns / 1e6)
                    parent_buf_init.append(pi.buffer_init_time_ns)
                    parent_enq.append(   pi.job_enqueue_time_ns)
                    parent_wait.append(  pi.job_wait_time_ns)
                    parent_agg.append(   pi.result_aggregate_time_ns)
                    parent_total.append( pi.total_time_ns)
                else:
                    parent_lats.append(0.0)
                    parent_buf_init.append(0)
                    parent_enq.append(0)
                    parent_wait.append(0)
                    parent_agg.append(0)
                    parent_total.append(0)

                # recall on first trial only
                if t == 0 and gt is not None:
                    preds = extract_ids(res)[0]
                    recalls.append(float(compute_recall(torch.tensor([preds]), gt[qi:qi+1], k).item()))

            # compute trial means
            child_lat_trials.append(np.mean(child_lats))
            parent_lat_trials.append(np.mean(parent_lats))
            child_buf_init_trials.append(np.mean(child_buf_init)/1e6)
            child_enq_trials     .append(np.mean(child_enq)     /1e6)
            child_wait_trials    .append(np.mean(child_wait)    /1e6)
            child_agg_trials     .append(np.mean(child_agg)     /1e6)
            child_total_trials   .append(np.mean(child_total)   /1e6)
            parent_buf_init_trials.append(np.mean(parent_buf_init)/1e6)
            parent_enq_trials     .append(np.mean(parent_enq)     /1e6)
            parent_wait_trials    .append(np.mean(parent_wait)    /1e6)
            parent_agg_trials     .append(np.mean(parent_agg)     /1e6)
            parent_total_trials   .append(np.mean(parent_total)   /1e6)

            if t == 0 and recalls:
                first_trial_recalls.append(np.mean(recalls))

            print(f" [{name} trial {t+1}/{trials}] "
                  f"child {child_lat_trials[-1]:.2f} ms, "
                  f"parent {parent_lat_trials[-1]:.2f} ms"
                  + (f" | R@{k} {first_trial_recalls[-1]:.4f}" if recalls else ""))

        # assemble final row
        row = {
            "index":             name,
            "n_workers":         b_params.get("num_workers", np.nan),
            "child_mean_latency_ms": float(np.mean(child_lat_trials)),
            "child_std_latency_ms":  float(np.std(child_lat_trials)),
            f"recall_at_{k}":        (float(first_trial_recalls[0]) if first_trial_recalls else np.nan),
            "child_buffer_init_ms":  float(np.mean(child_buf_init_trials)),
            "child_enqueue_ms":      float(np.mean(child_enq_trials)),
            "child_wait_ms":         float(np.mean(child_wait_trials)),
            "child_aggregate_ms":    float(np.mean(child_agg_trials)),
            "child_total_time_ms":   float(np.mean(child_total_trials)),
            "parent_mean_latency_ms":float(np.mean(parent_lat_trials)),
            "parent_buffer_init_ms": float(np.mean(parent_buf_init_trials)),
            "parent_enqueue_ms":     float(np.mean(parent_enq_trials)),
            "parent_wait_ms":        float(np.mean(parent_wait_trials)),
            "parent_aggregate_ms":   float(np.mean(parent_agg_trials)),
            "parent_total_time_ms":  float(np.mean(parent_total_trials)),
        }
        results.append(row)
        pd.DataFrame([row]).to_csv(idx_csv, index=False)

    # write aggregate CSV + plots
    df = pd.DataFrame(results)
    out_csv = Path(output_dir) / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\n[RESULT] aggregated CSV → {out_csv}")

    # latency plot (child only for clarity)
    plt.figure()
    for nm in df["index"].unique():
        sub = df[df["index"] == nm]
        lbl = (nm if sub["n_workers"].isnull().all()
               else f"{nm} (nw={int(sub['n_workers'].iloc[0])})")
        plt.errorbar(
            sub["n_workers"] if "n_workers" in sub else sub.index,
            sub["child_mean_latency_ms"],
            yerr=sub["child_std_latency_ms"],
            marker="o", capsize=5, label=lbl
        )
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers"); plt.ylabel("Mean Child Latency (ms)")
    plt.title("Single-Query Child Latency"); plt.legend(); plt.tight_layout()
    lat_path = Path(output_dir) / f"{out_csv.stem}_child_latency.png"
    plt.savefig(lat_path)
    print(f"[PLOT] child latency plot → {lat_path}")

    # recall plot
    if f"recall_at_{k}" in df.columns:
        plt.figure()
        for nm in df["index"].unique():
            sub = df[df["index"] == nm]
            lbl = (nm if sub["n_workers"].isnull().all()
                   else f"{nm} (nw={int(sub['n_workers'].iloc[0])})")
            plt.plot(
                sub["n_workers"] if "n_workers" in sub else sub.index,
                sub[f"recall_at_{k}"], marker="o", label=lbl
            )
        plt.xscale("symlog", base=2)
        plt.xlabel("Num Workers"); plt.ylabel(f"Recall@{k}")
        plt.title(f"Recall@{k}"); plt.legend(); plt.tight_layout()
        rec_path = Path(output_dir) / f"{out_csv.stem}_recall.png"
        plt.savefig(rec_path)
        print(f"[PLOT] recall plot → {rec_path}")