#!/usr/bin/env python3
"""
APS vs. early-termination baselines
Generates CSV + Markdown table with:
    RecallAchieved | nprobe | QueryLatencyMs | TuningTimeMs
for recall targets 0.80, 0.90, 0.99
"""

from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import yaml

# ── optional plotting ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

# ── Quake bindings ────────────────────────────────────────────────────────────
from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall

# ── LAET (mandatory) ──────────────────────────────────────────────────────────
try:
    from .laet import LAETPipeline
except ImportError as e:
    raise ImportError(
        "LAET is required for this experiment but could not be imported. "
        "Install the laet package or fix PYTHONPATH."
    ) from e

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)7s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("experiment")


# ----------------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------------- #
def make_sp(**attrs) -> SearchParams:
    """Create SearchParams then set attributes (bindings have no kwargs ctor)."""
    sp = SearchParams()
    for k, v in attrs.items():
        setattr(sp, k, v)
    return sp


def tune_auncel_a(
        idx: QuakeIndex,
        queries: torch.Tensor,
        gt: torch.Tensor,
        k: int,
        target_recall: float,
        recompute_thr: float,
        init_frac: float,
        max_iters: int = 15,
        a_min: float = 1e-5,
        a_max: float = .5,
) -> tuple[float, list[tuple[float, float, float]]]:
    """
    Binary-search the smallest 'a' that still meets the recall target.
    Returns (best_a, per_query_stats) where the stats are
    (nprobe, recall, latency_ms) for the final run.
    """
    best_a = a_max
    lo, hi = a_min, a_max
    best_stats: list[tuple[float, float, float]] = []

    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        per_query: list[tuple[float, float, float]] = []

        for q, g in zip(queries, gt):
            sp = make_sp(
                nprobe=-1,
                k=k,
                recall_target=target_recall,
                use_auncel=True,
                auncel_a=mid,
                auncel_b=.5,
                recompute_threshold=recompute_thr,
                initial_search_fraction=init_frac,
            )
            t0 = time.perf_counter_ns()
            res = idx.search(q.unsqueeze(0), sp)
            per_query.append(
                (
                    getattr(res.timing_info, "partitions_scanned", np.nan),
                    compute_recall(res.ids, g.unsqueeze(0), k).item(),
                    (time.perf_counter_ns() - t0) / 1e6,
                )
            )

        mean_recall = np.mean([r for _, r, _ in per_query])
        if mean_recall >= target_recall:
            # feasible → try smaller a
            best_a, best_stats = mid, per_query
            hi = mid * 0.5
        else:
            # not enough recall → increase a
            lo = mid * 1.5
        if hi - lo < 1e-3:
            break

    return best_a, best_stats


def binary_search_best_nprobe(
        idx: QuakeIndex,
        queries: torch.Tensor,
        gt: torch.Tensor,
        k: int,
        target_recall: float,
        max_nprobe: int,
        use_spann: bool = False,
        spann_eps: float = 1.5,
) -> int:
    """Return smallest nprobe reaching `target_recall` on the supplied queries."""
    lo, hi, best = 1, max_nprobe, max_nprobe
    while lo <= hi:
        mid = (lo + hi) // 2
        sp = make_sp(k=k, nprobe=mid, use_spann=use_spann, spann_eps=spann_eps if use_spann else 0.0)
        ids_all = [
            idx.search(q.unsqueeze(0), sp).ids
            for q in queries
            if idx.search(q.unsqueeze(0), sp).ids.numel()
        ]
        mean_rec = (
            compute_recall(torch.cat(ids_all, 0), gt, k).mean().item() if ids_all else 0.0
        )
        if mean_rec >= target_recall:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return best


def print_markdown(df: pd.DataFrame) -> None:
    print("\n" + "-" * 80)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("-" * 80 + "\n")


# ----------------------------------------------------------------------------- #
#  Core experiment
# ----------------------------------------------------------------------------- #
def run_experiment(cfg_path: str, output_dir: str) -> None:
    # 1) ---------------------------------------------------------------- config
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    k = cfg["experiment"]["k"]
    recall_targets: List[float] = cfg["experiment"]["recall_targets"]
    table_targets = {0.80, 0.90, 0.99}

    # 2) ---------------------------------------------------------------- data
    vecs, queries, gt = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("path", ""))
    nq = min(cfg["experiment"].get("nq", queries.shape[0]), queries.shape[0])
    queries, gt = queries[:nq], gt[:nq]
    logger.info(f"Dataset loaded – {len(vecs):,} base vecs, {nq} queries.")

    # 3) ---------------------------------------------------------------- index
    idx = QuakeIndex()
    bp = IndexBuildParams()
    bp.nlist = cfg["index"]["nlist"]
    bp.metric = cfg["index"]["metric"]
    bp.num_workers = cfg["experiment"].get("n_workers", 0)

    idx_file = Path(cfg["paths"]["index_dir"]) / f"{cfg['dataset']['name']}_quake_ivf{bp.nlist}.index"
    idx_file.parent.mkdir(parents=True, exist_ok=True)
    if idx_file.exists() and not cfg["overwrite"].get("index", False):
        idx.load(str(idx_file), 0)
        logger.info("Index loaded from disk.")
    else:
        logger.info("Building index …")
        idx.build(vecs, torch.arange(len(vecs)), bp)
        idx.save(str(idx_file))
        logger.info(f"Index saved to {idx_file}")

    max_nprobe = idx.nlist() if callable(getattr(idx, "nlist", None)) else bp.nlist

    # 4) ---------------------------------------------------------------- LAET (train / load)
    laet_cfg = cfg["laet_config"]

    # required dicts for get_or_train_model
    data_paths_cfg = {
        "base_vectors": laet_cfg["base_vectors_path_for_laet_train"],
        "train_query_vectors": laet_cfg["train_query_vectors_path_for_laet_train"],
        "gt_train": laet_cfg["gt_train_path_for_laet_train"],
        "gt_format": laet_cfg.get("gt_format_for_laet_train", "ivecs"),
        "training_vectors_for_faiss_index": laet_cfg["faiss_training_vectors_path_for_laet_train"],
    }
    generation_params_cfg = {
        "num_train_queries_to_process": laet_cfg.get("num_laet_training_samples", 10000),
        "k_gt_load_for_target": laet_cfg.get("laet_train_k_gt_load", 1),
        "k_search_for_target": laet_cfg.get("laet_train_k_search", 10),
        "max_nprobe_scan": laet_cfg.get("laet_train_max_nprobe", 128),
        "max_efsearch_scan": laet_cfg.get("laet_train_max_efsearch", 256),
        "param_scan_step": laet_cfg.get("laet_train_param_scan_step", 1),
    }
    faiss_params_cfg = {
        "faiss_index_key_for_gen": laet_cfg.get("faiss_index_key_for_laet_train_gen", f"IVF{max_nprobe},Flat")
    }
    training_params_cfg = {
        "log_target_training": laet_cfg.get("train_log_target", False),
        "test_size": laet_cfg.get("train_gbdt_test_size", 0.1),
        "num_leaves": laet_cfg.get("train_gbdt_num_leaves", 31),
        "learning_rate": laet_cfg.get("train_gbdt_lr", 0.05),
        "n_estimators": laet_cfg.get("train_gbdt_n_estimators", 100),
        "feature_fraction": laet_cfg.get("train_gbdt_feature_fraction", 0.9),
        "bagging_fraction": laet_cfg.get("train_gbdt_bagging_fraction", 0.8),
        "bagging_freq": laet_cfg.get("train_gbdt_bagging_freq", 5),
    }

    laet = LAETPipeline(
        vector_dim=queries.shape[1],
        laet_artifacts_config={
            "base_dir": cfg["paths"]["laet_artifacts_base_dir"],
            "dataset_name_for_files": laet_cfg.get(
                "dataset_name_for_files", cfg["dataset"]["name"].upper()
            ),
            "index_key_for_files": laet_cfg.get("index_key_for_files", f"IVF{max_nprobe}_Flat"),
            "model_subdir": laet_cfg.get("model_subdir", "laet_gbdt_models"),
        },
    )

    t_train0 = time.perf_counter_ns()
    laet.get_or_train_model(
        overwrite_model=False,
        data_paths_config=data_paths_cfg,
        generation_params_config=generation_params_cfg,
        faiss_params_config=faiss_params_cfg,
        training_params_config=training_params_cfg,
    )
    laet_training_ns = time.perf_counter_ns() - t_train0
    logger.info(f"LAET model ready (train/load time: {laet_training_ns/1e6:.1f} ms).")

    # 5) --------------------------------------------------------------- run methods
    records: List[Dict] = []

    for method in cfg["methods"]:
        for rt in recall_targets:
            logger.info(f"{method:>10s} | target recall {rt:.2f}")
            per_query = []  # tuples of (nprobe, recall, q_ms)
            tuning_ns = 0

            # ---------------------------------------------------------------- #
            if method == "Oracle":
                for q, g in zip(queries, gt):
                    t0 = time.perf_counter_ns()
                    best_np = binary_search_best_nprobe(
                        idx, q.unsqueeze(0), g.unsqueeze(0), k, rt, max_nprobe
                    )
                    tuning_ns += time.perf_counter_ns() - t0
                    sp = make_sp(k=k, nprobe=best_np)
                    t1 = time.perf_counter_ns()
                    res = idx.search(q.unsqueeze(0), sp)
                    per_query.append(
                        (
                            best_np,
                            compute_recall(res.ids, g.unsqueeze(0), k).item(),
                            (time.perf_counter_ns() - t1) / 1e6,
                        )
                    )

            elif method in ("FixedNProbe", "SPANN"):
                t0 = time.perf_counter_ns()
                best_np = binary_search_best_nprobe(
                    idx,
                    queries,
                    gt,
                    k,
                    rt,
                    max_nprobe,
                    use_spann=(method == "SPANN"),
                    spann_eps=1.5,
                )
                tuning_ns = time.perf_counter_ns() - t0
                for q, g in zip(queries, gt):
                    sp = make_sp(
                        k=k,
                        nprobe=best_np,
                        use_spann=(method == "SPANN"),
                        spann_eps=1.5 if method == "SPANN" else 0.0,
                    )
                    t1 = time.perf_counter_ns()
                    res = idx.search(q.unsqueeze(0), sp)
                    per_query.append(
                        (
                            best_np,
                            compute_recall(res.ids, g.unsqueeze(0), k).item(),
                            (time.perf_counter_ns() - t1) / 1e6,
                        )
                    )

            elif method == "APS":
                for q, g in zip(queries, gt):
                    sp = make_sp(
                        nprobe=-1,
                        k=k,
                        recall_target=rt,
                        recompute_threshold=cfg["experiment"]["recompute_ratio"],
                        use_precomputed=cfg["experiment"]["use_precompute"],
                        initial_search_fraction=cfg["experiment"]["initial_search_fraction"],
                    )
                    t1 = time.perf_counter_ns()
                    res = idx.search(q.unsqueeze(0), sp)
                    per_query.append(
                        (
                            getattr(res.timing_info, "partitions_scanned", np.nan),
                            compute_recall(res.ids, g.unsqueeze(0), k).item(),
                            (time.perf_counter_ns() - t1) / 1e6,
                        )
                    )

            elif method == "Auncel":
                t0 = time.perf_counter_ns()
                best_a, per_query = tune_auncel_a(
                    idx,
                    queries,
                    gt,
                    k,
                    rt,
                    recompute_thr=cfg["experiment"]["recompute_ratio"],
                    init_frac=cfg["experiment"]["initial_search_fraction"],
                )
                tuning_ns = time.perf_counter_ns() - t0
                logger.info(f"Auncel @ RT={rt:.2f} → selected a = {best_a:.4f}")

            elif method == "LAET":
                t0 = time.perf_counter_ns()
                mean_np, mean_rec, mean_ms = laet.run_inference_for_quake_experiment(
                    quake_idx=idx,
                    queries_torch=queries,
                    gt_torch=gt,
                    target_recall=rt,
                    k_search=k,
                    base_quake_sp=make_sp(k=k),
                )
                tuning_ns = laet_training_ns + (time.perf_counter_ns() - t0)
                per_query.append((mean_np, mean_rec, mean_ms))

            else:
                logger.warning(f"Unknown method '{method}' – skipping.")
                continue
            # ---------------------------------------------------------------- #

            if not per_query:
                continue
            arr = np.asarray(per_query, dtype=float)
            mean_np, mean_rec, mean_q_ms = np.nanmean(arr, axis=0)

            records.append(
                dict(
                    Method=method,
                    RecallTarget=rt,
                    RecallAchieved=mean_rec,
                    nprobe=mean_np,
                    QueryLatencyMs=mean_q_ms,
                    TuningTimeMs=tuning_ns / 1e6,
                )
            )

    # 6) ------------------------------------------------------------- results
    df = pd.DataFrame(records)
    df_out = df[df["RecallTarget"].isin(table_targets)].copy().sort_values(
        ["Method", "RecallTarget"]
    )
    csv_path = out_dir / f"aps_et_results_{cfg['dataset']['name']}.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"Results written to {csv_path}")
    print_markdown(df_out)

    # 7) ------------------------------------------------------------- plot
    if cfg.get("plot", {}).get("enable", False) and _HAVE_MPL:
        plt.figure(figsize=(6, 4))
        for m in df["Method"].unique():
            d = df[df["Method"] == m]
            plt.plot(d["RecallTarget"], d["QueryLatencyMs"], marker="o", label=m, alpha=0.7)
        plt.yscale("log")
        plt.xlabel("Recall target")
        plt.ylabel("Mean query latency (ms)")
        plt.title(cfg["dataset"]["name"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"latency_plot_{cfg['dataset']['name']}.png", dpi=200)