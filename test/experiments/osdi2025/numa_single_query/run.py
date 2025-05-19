#!/usr/bin/env python3
"""
NUMA Single-Query Latency and Recall Benchmark (v2)

Builds each non-Quake index once.
Builds **one** shared Quake base-index and re-uses it for every Quake
configuration, so we avoid the O(dataset) rebuild for each worker-count.

Usage
-----
    python numa_latency_experiment.py <config.yaml> <output_dir>
"""

from pathlib import Path
import os
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

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def build_and_save(index_cls, params, vecs, path):
    """Build <index_cls> with <params> on <vecs> and persist to <path>."""
    idx = index_cls()
    idx.build(vecs, **params)
    idx.save(str(path))

def extract_ids(res):
    """Return search‐result ids as np.ndarray."""
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

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run_experiment(cfg_path: str, output_dir: str) -> None:
    cfg          = yaml.safe_load(Path(cfg_path).read_text())
    ds_cfg       = cfg["dataset"]
    k            = ds_cfg["k"]
    q_n          = ds_cfg["num_queries"]
    trials       = cfg.get("trials", 5)
    warmup_n     = cfg.get("warmup", 10)
    idx_cfgs     = cfg["indexes"]
    csv_name     = cfg.get("output", {}).get("results_csv", "numa_results.csv")
    overwrite    = cfg.get("overwrite", False)

    out_dir = Path(output_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Dataset                                                               #
    # --------------------------------------------------------------------- #
    print(f"[INFO] loading dataset {ds_cfg['name']} …")
    base_vecs, queries, gt = load_dataset(ds_cfg["name"])
    queries = queries[:q_n]
    gt      = gt[:q_n] if gt is not None else None

    # --------------------------------------------------------------------- #
    # Build a single Quake base index (if any Quake config exists)          #
    # --------------------------------------------------------------------- #
    quake_base_params = None
    for cfg_i in idx_cfgs:
        if cfg_i["index"] == "Quake":
            quake_base_params = dict(cfg_i.get("build_params", {}))
            # Remove loading-stage-only keys; not needed for the build
            quake_base_params.pop("num_workers", None)
            quake_base_params.pop("use_numa", None)
            break

    quake_base_path = out_dir / "Quake_base_index.bin"
    if quake_base_params:
        if overwrite and quake_base_path.exists():
            os.remove(quake_base_path)
        if not quake_base_path.exists():
            print("[BUILD] Quake (shared) base index …")
            build_and_save(QuakeWrapper, quake_base_params, base_vecs, quake_base_path)
            print(f"[SAVE] Quake base index → {quake_base_path}")

    # --------------------------------------------------------------------- #
    # Per-configuration benchmarking                                        #
    # --------------------------------------------------------------------- #
    rows = []
    for cfg_i in idx_cfgs:
        name        = cfg_i["name"]
        typ         = cfg_i["index"]
        b_params    = dict(cfg_i.get("build_params", {}))       # safe copy
        s_params    = dict(cfg_i.get("search_params", {}))
        cls         = INDEX_CLASSES.get(typ)
        if cls is None:
            print(f"[WARN] unknown index type '{typ}' – skipping.")
            continue

        idx_csv  = out_dir / f"{name}_results.csv"
        idx_path = quake_base_path if typ == "Quake" else out_dir / f"{name}_index.bin"

        # Skip if we already have results and not overwriting
        if idx_csv.exists() and not overwrite:
            print(f"[SKIP] {name} – results exist.")
            rows.append(pd.read_csv(idx_csv).iloc[0].to_dict())
            continue

        # Build non-Quake index (one-off)
        if typ != "Quake" and not idx_path.exists():
            print(f"[BUILD] {name} index …")
            build_and_save(cls, b_params, base_vecs, idx_path)
            print(f"[SAVE] index → {idx_path}")

        # ----------------------------------------------------------------- #
        # Load                                                               #
        # ----------------------------------------------------------------- #
        if typ == "Quake":
            load_kw = {k: b_params[k] for k in ("use_numa", "num_workers") if k in b_params}
            idx = QuakeWrapper()
            idx.load(str(idx_path), **load_kw)
        elif typ == "DiskANN":
            idx = cls()
            b_params.pop("metric", None)      # DiskANN load signature
            idx.load(str(idx_path), **b_params)
        else:
            idx = cls()
            idx.load(str(idx_path))

        # ----------------------------------------------------------------- #
        # Search parameter dispatch                                          #
        # ----------------------------------------------------------------- #
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

        # ----------------------------------------------------------------- #
        # Warm-up                                                           #
        # ----------------------------------------------------------------- #
        for q in queries[:min(warmup_n, len(queries))]:
            idx.search(q.unsqueeze(0).float(), k, **search_kw)

        # ----------------------------------------------------------------- #
        # Trials                                                            #
        # ----------------------------------------------------------------- #
        trial_lat  = []
        trial_rec  = []
        for t in range(trials):
            lats_ms, preds = [], []
            for qi, q in enumerate(queries):
                res = idx.search(q.unsqueeze(0).float(), k, **search_kw)
                # latency collection
                ti = getattr(res, "timing_info", None)
                if ti and hasattr(ti, "total_time_ns"):
                    lats_ms.append(ti.total_time_ns / 1e6)
                elif hasattr(res, "latency_ms"):
                    lats_ms.append(res.latency_ms)
                else:
                    raise RuntimeError("missing timing info.")
                if t == 0 and gt is not None:
                    preds.append(extract_ids(res)[0])  # [k]
            trial_lat.append(float(np.mean(lats_ms)))
            if t == 0 and gt is not None:
                recall = float(compute_recall(torch.tensor(preds), gt, k).mean())
                trial_rec.append(recall)
                print(f" [{name} trial {t+1}/{trials}] {trial_lat[-1]:.2f} ms | R@{k} {recall:.4f}")
            else:
                print(f" [{name} trial {t+1}/{trials}] {trial_lat[-1]:.2f} ms")

        # ----------------------------------------------------------------- #
        # Persist per-config results                                         #
        # ----------------------------------------------------------------- #
        row = {
            "index":            name,
            "n_workers":        b_params.get("num_workers", np.nan),
            "mean_latency_ms":  float(np.mean(trial_lat)),
            "std_latency_ms":   float(np.std(trial_lat)),
            f"recall_at_{k}":   (float(trial_rec[0]) if trial_rec else np.nan),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(idx_csv, index=False)

    # --------------------------------------------------------------------- #
    # Aggregate CSV + plots                                                 #
    # --------------------------------------------------------------------- #
    df = pd.DataFrame(rows)
    out_csv = out_dir / csv_name
    df.to_csv(out_csv, index=False)
    print(f"\n[RESULT] aggregated CSV → {out_csv}")

    # Latency plot
    plt.figure()
    for nm in df["index"].unique():
        sub = df[df["index"] == nm]
        lbl = f"{nm}" if sub["n_workers"].isnull().all() else f"{nm} (nw={sub['n_workers'].iloc[0]})"
        plt.errorbar(sub["n_workers"] if "n_workers" in sub else sub.index,
                     sub["mean_latency_ms"], yerr=sub["std_latency_ms"],
                     marker="o", capsize=5, label=lbl)
    plt.xscale("symlog", base=2)
    plt.xlabel("Num Workers")
    plt.ylabel("Mean Latency (ms)")
    plt.title("Single-Query Latency")
    plt.legend()
    plt.tight_layout()
    lat_path = out_dir / f"{out_csv.stem}_latency.png"
    plt.savefig(lat_path)
    print(f"[PLOT] latency plot → {lat_path}")

    # Recall plot
    if f"recall_at_{k}" in df.columns:
        plt.figure()
        for nm in df["index"].unique():
            sub = df[df["index"] == nm]
            lbl = f"{nm}" if sub["n_workers"].isnull().all() else f"{nm} (nw={sub['n_workers'].iloc[0]})"
            plt.plot(sub["n_workers"] if "n_workers" in sub else sub.index,
                     sub[f"recall_at_{k}"], marker="o", label=lbl)
        plt.xscale("symlog", base=2)
        plt.xlabel("Num Workers")
        plt.ylabel(f"Recall@{k}")
        plt.title(f"Recall@{k}")
        plt.legend()
        plt.tight_layout()
        rec_path = out_dir / f"{out_csv.stem}_recall.png"
        plt.savefig(rec_path)
        print(f"[PLOT] recall plot → {rec_path}")