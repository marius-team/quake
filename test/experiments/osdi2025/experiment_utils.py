# File: test/experiments/osdi2025/common/experiment_utils.py
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.datasets.ann_datasets import load_dataset as quake_load_dataset
from quake.utils import compute_recall as quake_compute_recall
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

logger = logging.getLogger(__name__)

def load_config(cfg_path: str) -> dict:
    return yaml.safe_load(Path(cfg_path).read_text())

def load_data(dataset_name: str, dataset_path: str = "data/", nq_override: int = None):
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
        use_numa: bool = False,
        load: bool = True
) -> QuakeIndex:
    idx = QuakeIndex()
    if index_file_path.exists() and not force_rebuild and load:
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

def generate_dynamic_workload(
        dataset_main_cfg: Dict[str, Any],
        workload_generator_cfg: Dict[str, Any],
        global_output_dir: Path,
        overwrite_workload: bool
) -> Path:
    """
    Loads dataset and generates the dynamic workload.
    Returns the path to the workload directory (which is the global_output_dir).
    """
    log = logger # Use the logger defined in common_utils

    log.info(f"Preparing dataset '{dataset_main_cfg['name']}' for dynamic workload.")
    # Assuming common_utils.load_data is defined elsewhere in experiment_utils.py
    base_vectors, query_vectors, _ = load_data(
        dataset_main_cfg["name"], dataset_main_cfg.get("path", "")
    )

    log.info("Setting up DynamicWorkloadGenerator.")
    generator = DynamicWorkloadGenerator(
        workload_dir=global_output_dir,
        base_vectors=base_vectors,
        metric=dataset_main_cfg["metric"],
        insert_ratio=workload_generator_cfg["insert_ratio"],
        delete_ratio=workload_generator_cfg["delete_ratio"],
        query_ratio=workload_generator_cfg["query_ratio"],
        update_batch_size=workload_generator_cfg["update_batch_size"],
        query_batch_size=workload_generator_cfg["query_batch_size"],
        number_of_operations=workload_generator_cfg["number_of_operations"],
        initial_size=workload_generator_cfg["initial_size"],
        cluster_size=workload_generator_cfg["cluster_size"],
        cluster_sample_distribution=workload_generator_cfg["cluster_sample_distribution"],
        queries=query_vectors,
        query_cluster_sample_distribution=workload_generator_cfg["query_cluster_sample_distribution"],
        seed=workload_generator_cfg["seed"]
    )

    if not generator.workload_exists() or overwrite_workload:
        log.info(f"Generating workload in {global_output_dir}...")
        generator.generate_workload()
    else:
        log.info(f"Existing workload found in {global_output_dir} – generation skipped.")
    return global_output_dir


def evaluate_index_on_dynamic_workload(
        index_config: Dict[str, Any],       # Single entry from the 'indexes' list in YAML
        index_class_mapping: Dict[str, type], # Maps index type string to class (e.g., {"Quake": QuakeWrapper})
        workload_data_dir: Path,            # Path where DynamicWorkloadGenerator output (operations, groundtruth) is stored
        experiment_main_output_dir: Path, # Base output dir for the whole experiment (e.g., .../maintenance_ablation/results/sift1m/)
        overwrite_idx_results: bool,
        do_maintenance_flag: bool = False   # Specific to experiments that test maintenance
):
    """
    Evaluates a single index configuration against a pre-generated dynamic workload.
    Results are saved by WorkloadEvaluator in a subdirectory named after the index.
    """
    log = logger # Use the logger defined in common_utils
    idx_name = index_config["name"]
    index_type_str = index_config["index"]

    # Determine the directory for this specific index's results
    idx_specific_output_dir = experiment_main_output_dir / idx_name
    idx_specific_output_dir.mkdir(parents=True, exist_ok=True)

    per_index_results_csv = idx_specific_output_dir / "results.csv"

    if per_index_results_csv.exists() and not overwrite_idx_results:
        log.info(f"Results for {idx_name} exist at {per_index_results_csv} – skipping evaluation.")
        return

    log.info(f"Evaluating index configuration: {idx_name} (Type: {index_type_str})")

    index_class_constructor = index_class_mapping.get(index_type_str)
    if not index_class_constructor:
        log.error(f"Unknown index type '{index_type_str}' in mapping for index '{idx_name}'. Skipping.")
        return

    index_instance = index_class_constructor()

    evaluator = WorkloadEvaluator(
        workload_dir=workload_data_dir,       # Where operation files are located
        output_dir=idx_specific_output_dir  # Where this index's results.csv will be saved
    )

    evaluator.evaluate_workload(
        name=idx_name,
        index=index_instance,
        build_params=index_config.get("build_params", {}),
        search_params=index_config.get("search_params", {}),
        m_params=index_config.get("maintenance_params", {}), # Pass maintenance_params
        do_maintenance=do_maintenance_flag,
    )
    log.info(f"Finished evaluating {idx_name}. Results should be in {idx_specific_output_dir}")


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

def expand_search_sweep(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expands nprobes or recall_targets from sweep_config into individual search dicts."""
    base_params = {k: v for k, v in sweep_config.items() if k not in ["nprobes", "recall_targets"]}
    expanded_list = []
    if "nprobes" in sweep_config:
        for nprobe_val in sweep_config["nprobes"]:
            expanded_list.append({**base_params, "nprobe": nprobe_val})
    elif "recall_targets" in sweep_config:
        for rt_val in sweep_config["recall_targets"]:
            expanded_list.append({**base_params, "recall_target": rt_val})
    else: # No sweep, just use base_params
        expanded_list.append(base_params)
    return expanded_list