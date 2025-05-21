# File: test/experiments/osdi2025/common/experiment_utils.py
import time
import yaml
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.datasets.ann_datasets import load_dataset as quake_load_dataset
from quake.utils import compute_recall as quake_compute_recall

logger = logging.getLogger(__name__)

def load_config(cfg_path: str) -> dict:
    return yaml.safe_load(Path(cfg_path).read_text())

def load_data(dataset_name: str, dataset_path: str, nq_override: int = None):
    vecs, queries, gt = quake_load_dataset(dataset_name, dataset_path)
    if nq_override is not None and nq_override > 0 and nq_override < len(queries):
        queries = queries[:nq_override]
        if gt is not None:
            gt = gt[:nq_override]
    return vecs, queries, gt

def prepare_quake_index(
        vecs: torch.Tensor,
        build_params_dict: dict,
        index_file_path: Path,
        force_rebuild: bool = False,
        num_workers_load: int = 0,
        num_parent_workers_load: int = 0,
        use_numa: bool = False
) -> QuakeIndex:
    idx = QuakeIndex()
    if index_file_path.exists() and not force_rebuild:
        logger.info(f"Loading index from {index_file_path}")
        idx.load(str(index_file_path), num_workers_load, use_numa, num_parent_workers_load)
    else:
        logger.info(f"Building index -> {index_file_path}")
        bp = IndexBuildParams()
        for key, value in build_params_dict.items():
            if hasattr(bp, key):
                setattr(bp, key, value)
        idx.build(vecs, torch.arange(len(vecs)), bp)
        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        idx.save(str(index_file_path))
        logger.info(f"Index saved to {index_file_path}")
    return idx

def create_search_params(**attrs) -> SearchParams:
    sp = SearchParams()
    for k, v in attrs.items():
        if hasattr(sp, k):
            setattr(sp, k, v)
    return sp

def run_search_trial(
        idx: QuakeIndex,
        query: torch.Tensor,
        gt_vector: torch.Tensor, # Ground truth for this single query
        k_val: int,
        search_params: SearchParams
) -> tuple:
    t0 = time.perf_counter_ns()
    res = idx.search(query.unsqueeze(0), search_params)
    elapsed_ns = time.perf_counter_ns() - t0
    rec = quake_compute_recall(res.ids, gt_vector.unsqueeze(0), k_val).item()
    partitions_scanned = getattr(res.timing_info, "partitions_scanned", np.nan)
    return partitions_scanned, rec, elapsed_ns / 1e6 # nprobe, recall, time_ms

def save_results_csv(records: list, output_path: Path):
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

def plot_recall_performance(
        df: pd.DataFrame,
        x_metric: str, # e.g., "RecallTarget"
        y_metrics: list, # e.g., ["mean_time_ms", "mean_recall", "mean_nprobe"]
        y_labels: list,
        y_scales: list = None, # e.g., ["linear", "linear", "log"]
        plot_title_prefix: str = "",
        output_path: Path = None,
        recall_targets_for_line: list = None
):
    num_plots = len(y_metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 3 * num_plots + 1), sharex=True)
    if num_plots == 1:
        axes = [axes]

    for m_idx, method_name in enumerate(df["Method"].unique()):
        grp = df[df["Method"] == method_name]
        for i, y_col in enumerate(y_metrics):
            y_err_col = df.columns[df.columns.get_loc(y_col) + 1] if f"std_{y_col.split('_')[1]}" in df.columns else None
            yerr_values = grp[y_err_col] if y_err_col and y_err_col in grp.columns else None
            axes[i].errorbar(
                grp[x_metric], grp[y_col],
                yerr=yerr_values,
                marker="o", capsize=4, label=method_name,
                linestyle='-' if m_idx == 0 else '--' # Differentiate methods
            )

    for i, y_col in enumerate(y_metrics):
        axes[i].set_ylabel(y_labels[i])
        if y_scales and y_scales[i]:
            axes[i].set_yscale(y_scales[i])
        if y_col == "mean_recall" and recall_targets_for_line:
            axes[i].plot(recall_targets_for_line, recall_targets_for_line, "k--", label="Target")
        axes[i].legend()
        axes[i].grid(True, which="both", ls=":")
        axes[i].set_title(f"{plot_title_prefix}: {x_metric} vs {y_labels[i]}")

    axes[-1].set_xlabel(x_metric)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for suptitle
    fig.suptitle(f"{plot_title_prefix} Performance", fontsize=14)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Plot saved to {output_path}")
    plt.close(fig)