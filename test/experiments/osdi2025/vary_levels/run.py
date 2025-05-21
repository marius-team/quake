#!/usr/bin/env python3
"""
vary_levels.py - Benchmarks Quake with varying hierarchical structures.

This script evaluates Quake's performance with an arbitrary number of index levels.
It builds a base child index (L0) once. Then, for each defined 'experiment_setup',
it loads the base index and adds any specified additional parent levels (L1, L2, etc.)
on top.

Each complete multi-level index is then benchmarked against a sweep of search
parameters applied to the innermost (L0) level. Search parameters for higher
levels (L1, L2, ...) are fixed for each experiment_setup.

Key metrics, including overall QPS, latency, recall, and per-level scan times
and statistics (e.g., partitions scanned), are recorded.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from quake import QuakeIndex, IndexBuildParams, SearchParams, SearchTimingInfo
from test.experiments.osdi2025 import experiment_utils as common_utils

logger = logging.getLogger("vary_levels_exp")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

def build_hierarchical_index(
        base_vectors: torch.Tensor,
        child_index_config: Dict[str, Any],
        additional_levels_config: List[Dict[str, Any]],
        base_child_index_path: Path,
        force_rebuild_base: bool,
        exp_setup_name: str
) -> Tuple[QuakeIndex | None, float]:
    """
    Builds or loads the base child index (L0), then adds additional levels on top.
    Returns the fully constructed QuakeIndex instance and total build time for L0.
    Build time for additional levels is not explicitly tracked here but happens during add_level.
    """
    # Prepare L0 (base child index)
    # Note: build_params_dict for prepare_quake_index should be flat, matching IndexBuildParams attributes
    l0_build_params_dict = child_index_config.get("build_params", {})

    # Ensure metric is in the build_params_dict for prepare_quake_index if it expects it
    if 'metric' not in l0_build_params_dict and 'metric' in child_index_config:
        l0_build_params_dict['metric'] = child_index_config['metric']
    if 'metric' not in l0_build_params_dict and 'metric' in dataset_cfg: # Fallback to dataset metric
        l0_build_params_dict['metric'] = dataset_cfg['metric']


    # Use common_utils to prepare the base L0 index
    # This function handles building if not exists or force_rebuild is true, and saving.
    # It returns the loaded/built index instance.
    idx = common_utils.prepare_quake_index(
        vecs=base_vectors,
        build_params_dict=l0_build_params_dict,
        index_file_path=base_child_index_path,
        force_rebuild=force_rebuild_base,
        # Pass load-time worker/NUMA settings for L0 if they were part of its build_params
        # For Quake, these are typically search-time. If prepare_quake_index needs them for load:
        num_workers_load=l0_build_params_dict.get("num_workers", 0), # Assuming build num_workers is also load num_workers
        use_numa=l0_build_params_dict.get("use_numa", False) # Assuming build use_numa is also load use_numa
    )
    # Build time for L0 is implicitly handled by prepare_quake_index if it were to return it.
    # For now, we'll assume build_time_l0 is primarily for the initial creation.
    # If the file existed, build_time_l0 would be 0.
    build_time_l0 = 0.0 # Placeholder, as prepare_quake_index doesn't return build time directly.

    if idx is None:
        logger.error(f"Failed to prepare base child index for {exp_setup_name}.")
        return None, build_time_l0

    # Add additional levels (L1, L2, ...)
    for i, level_cfg in enumerate(additional_levels_config):
        level_num = i + 1 # L1, L2, ...
        logger.info(f"For '{exp_setup_name}', adding L{level_num}...")

        add_level_build_params_dict = level_cfg.get("build_params", {})
        if not add_level_build_params_dict.get("nc"):
            logger.warning(f"Skipping L{level_num} for '{exp_setup_name}' due to missing 'nc' in build_params.")
            continue

        # Create IndexBuildParams for adding the level
        # The metric for parent levels is typically inherited or fixed by Quake internally.
        # We primarily need to set 'nc' and 'num_workers' for building this parent level.
        parent_bp = IndexBuildParams()
        parent_bp.nlist = add_level_build_params_dict["nc"]
        parent_bp.num_workers = add_level_build_params_dict.get("num_workers", 0)
        parent_bp.metric = l0_build_params_dict.get("metric", "l2") # Parent inherits metric

        try:
            idx.add_level(parent_bp) # Assumes QuakeIndex has .index attribute pointing to C++ object
            logger.info(f"L{level_num} (NC={parent_bp.nlist}) added successfully for '{exp_setup_name}'.")
        except Exception as e:
            logger.error(f"Failed to add L{level_num} for '{exp_setup_name}': {e}")
            return None, build_time_l0

    return idx, build_time_l0


def construct_hierarchical_search_params(
        innermost_level_sp_item: Dict[str, Any],
        additional_levels_config: List[Dict[str, Any]]
) -> SearchParams:
    """
    Constructs a nested SearchParams object for the entire hierarchy.
    """
    # Innermost level (L0) search parameters
    l0_sp = common_utils.create_search_params(**innermost_level_sp_item)

    current_parent_sp = l0_sp
    for level_cfg in additional_levels_config: # L1, L2, ...
        level_search_params_dict = level_cfg.get("search_params", {})
        if not level_search_params_dict: # If no specific search params, create default
            level_search_params_dict = {"nprobe": 1} # Minimal valid SearchParams

        # These are search-time worker/NUMA settings for this specific parent level's scan
        # common_utils.create_search_params will set these if present in level_search_params_dict
        higher_level_sp = common_utils.create_search_params(**level_search_params_dict)

        current_parent_sp.parent_params = higher_level_sp
        current_parent_sp = higher_level_sp

    return l0_sp


def _extract_recursive_timings(innermost_timing_info: SearchTimingInfo | None, num_levels: int) -> Dict[str, Any]:
    """
    Extracts scan time and other stats for each level from the TimingInfo hierarchy.
    The innermost_timing_info is for L0. Its .parent_info is for L1, and so on.
    """
    timings = {}
    current_ti = innermost_timing_info

    for i in range(num_levels):
        level_prefix = f"level_{i}"
        if current_ti:
            timings[f"{level_prefix}_scan_time_ms"] = current_ti.total_time_ns / 1e6
            timings[f"{level_prefix}_buffer_init_ms"] = current_ti.buffer_init_time_ns / 1e6
            timings[f"{level_prefix}_enqueue_ms"] = current_ti.job_enqueue_time_ns / 1e6
            timings[f"{level_prefix}_wait_ms"] = current_ti.job_wait_time_ns / 1e6
            timings[f"{level_prefix}_aggregate_ms"] = current_ti.result_aggregate_time_ns / 1e6
            timings[f"{level_prefix}_partitions_scanned"] = getattr(current_ti, "partitions_scanned", np.nan)
            current_ti = current_ti.parent_info
        else:
            timings[f"{level_prefix}_scan_time_ms"] = np.nan
            timings[f"{level_prefix}_buffer_init_ms"] = np.nan
            timings[f"{level_prefix}_enqueue_ms"] = np.nan
            timings[f"{level_prefix}_wait_ms"] = np.nan
            timings[f"{level_prefix}_aggregate_ms"] = np.nan
            timings[f"{level_prefix}_partitions_scanned"] = np.nan
    return timings


def benchmark_search_config(
        idx: QuakeIndex,
        queries: torch.Tensor,
        ground_truth: np.ndarray | None,
        hierarchical_sp: SearchParams,
        num_levels_in_hierarchy: int,
        num_trials: int,
        warmup_trials: int
) -> Dict[str, Any]:
    """
    Benchmarks a specific hierarchical search configuration.
    Returns aggregated metrics including per-level timings.
    """
    k_val = hierarchical_sp.k # k from the L0 SearchParams

    # Warm-up
    for _ in range(warmup_trials):
        idx.search(queries, hierarchical_sp)

    trial_metrics_list = []
    for _ in range(num_trials):
        t_start_ns = time.perf_counter_ns()
        search_result = idx.search(queries, hierarchical_sp)
        t_elapsed_ns = time.perf_counter_ns() - t_start_ns

        current_recall = np.nan
        if ground_truth is not None and search_result.ids is not None:
            # Ensure k_val does not exceed columns in ground_truth or search_result.ids
            max_k_recall = min(k_val, ground_truth.shape[1], search_result.ids.shape[1])
            if max_k_recall > 0 :
                current_recall = common_utils.quake_compute_recall(
                    search_result.ids[:, :max_k_recall],
                    ground_truth[:, :max_k_recall],
                    max_k_recall
                ).mean().item()


        per_level_timings = _extract_recursive_timings(search_result.timing_info, num_levels_in_hierarchy)

        trial_metrics_list.append({
            "overall_latency_ns": t_elapsed_ns,
            "recall": current_recall,
            **per_level_timings
        })

    # Aggregate results from trials
    num_queries_total = len(queries)
    aggregated_results = {
        "qps_mean": np.mean([num_queries_total / (m["overall_latency_ns"] / 1e9) for m in trial_metrics_list]),
        "overall_latency_ms_mean": np.mean([m["overall_latency_ns"] / 1e6 for m in trial_metrics_list]),
        "recall_mean": np.nanmean([m["recall"] for m in trial_metrics_list]),
    }

    # Aggregate per-level timings and stats
    for i in range(num_levels_in_hierarchy):
        level_prefix = f"level_{i}"
        for metric_suffix in ["scan_time_ms", "buffer_init_ms", "enqueue_ms", "wait_ms", "aggregate_ms", "partitions_scanned"]:
            key = f"{level_prefix}_{metric_suffix}"
            valid_values = [m[key] for m in trial_metrics_list if key in m and pd.notna(m[key])]
            if valid_values:
                aggregated_results[f"{key}_mean"] = np.mean(valid_values)
            else:
                aggregated_results[f"{key}_mean"] = np.nan

    return aggregated_results


def run_experiment(config_path_str: str, output_dir_str: str) -> None:
    global dataset_cfg # Make dataset_cfg accessible to build_hierarchical_index if needed for metric fallback
    config = common_utils.load_config(config_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config["dataset"]
    child_index_cfg = config["child_index_config"] # L0 build config
    innermost_search_sweep_cfg = config["global_innermost_search_sweep"] # L0 search sweep
    experiment_setups_cfg = config["experiment_setups"]
    exp_params_cfg = config["experiment_params"]
    overwrite_cfg = config["overwrite"]

    logger.info(f"Loading dataset: {dataset_cfg['name']}")
    base_vecs, queries, gt_numpy = common_utils.load_data(
        dataset_cfg["name"], dataset_cfg.get("path", ""), nq_override=dataset_cfg.get("nq")
    )

    # Path for the L0 index binary
    base_child_index_path = output_dir / "base_index.bin"

    l0_build_params_dict = child_index_cfg.get("build_params", {})
    if 'metric' not in l0_build_params_dict: # Ensure metric is set for L0 build
        l0_build_params_dict['metric'] = dataset_cfg.get('metric', 'l2')

    _ = common_utils.prepare_quake_index( # Build L0 and save it
        vecs=base_vecs,
        build_params_dict=l0_build_params_dict,
        index_file_path=base_child_index_path,
        force_rebuild=overwrite_cfg.get("index", False)
    )
    logger.info(f"Base L0 index prepared at {base_child_index_path}")

    all_experiment_run_records = []
    innermost_sweep_list = common_utils.expand_search_sweep(innermost_search_sweep_cfg) # Adapted from common_utils or defined here

    for exp_setup_item in experiment_setups_cfg:
        exp_setup_name = exp_setup_item["name"]
        additional_levels_config = exp_setup_item.get("additional_levels", [])
        num_actual_levels = 1 + len(additional_levels_config) # L0 + additional levels

        exp_setup_results_path = output_dir / f"{exp_setup_name}_results.csv"
        if exp_setup_results_path.exists() and not overwrite_cfg.get("results", True):
            logger.info(f"Results for setup '{exp_setup_name}' exist, loading cached.")
            try:
                cached_df = pd.read_csv(exp_setup_results_path)
                all_experiment_run_records.extend(cached_df.to_dict('records'))
            except Exception as e:
                logger.error(f"Could not load cached results for {exp_setup_name}: {e}")
            continue

        logger.info(f"--- Processing Experiment Setup: {exp_setup_name} ({num_actual_levels} levels) ---")

        # Assemble the full hierarchical index for this setup
        # This loads L0 and adds L1, L2... on top.
        current_hier_idx, _ = build_hierarchical_index(
            base_vecs, child_index_cfg, additional_levels_config,
            base_child_index_path, False, # False: don't force rebuild L0 if already done
            exp_setup_name
        )
        if not current_hier_idx:
            logger.error(f"Failed to build index for setup '{exp_setup_name}'. Skipping.")
            continue

        setup_specific_records = []
        for l0_sp_item in innermost_sweep_list: # Iterate L0 search params (nprobe/recall_target)
            param_key = "nprobe" if "nprobe" in l0_sp_item else "recall_target"
            param_val = l0_sp_item.get(param_key, "N/A")
            logger.info(f"Benchmarking '{exp_setup_name}' with L0 {param_key}={param_val}...")

            # Construct the full stack of SearchParams for this run
            hierarchical_sp = construct_hierarchical_search_params(l0_sp_item, additional_levels_config)

            benchmark_stats = benchmark_search_config(
                current_hier_idx, queries, gt_numpy, hierarchical_sp,
                num_levels_in_hierarchy=num_actual_levels,
                num_trials=exp_params_cfg.get("trials", 3),
                warmup_trials=exp_params_cfg.get("warmup", 1)
            )

            # Record configuration and results
            record = {"exp_setup_name": exp_setup_name}
            record[f"L0_{param_key}"] = param_val # L0 sweep parameter
            record["L0_k"] = l0_sp_item.get("k")
            record["L0_batch_size"] = l0_sp_item.get("batch_size")
            record["L0_nc"] = child_index_cfg.get("build_params", {}).get("nc")


            for i, add_level_cfg in enumerate(additional_levels_config):
                level_num = i + 1 # L1, L2...
                record[f"L{level_num}_nc"] = add_level_cfg.get("build_params",{}).get("nc")
                record[f"L{level_num}_build_workers"] = add_level_cfg.get("build_params",{}).get("num_workers")
                record[f"L{level_num}_search_nprobe"] = add_level_cfg.get("search_params",{}).get("nprobe")
                record[f"L{level_num}_search_recall_target"] = add_level_cfg.get("search_params",{}).get("recall_target")
                record[f"L{level_num}_search_workers"] = add_level_cfg.get("search_params",{}).get("num_workers")
                record[f"L{level_num}_search_use_numa"] = add_level_cfg.get("search_params",{}).get("use_numa")

            record.update(benchmark_stats)
            setup_specific_records.append(record)
            all_experiment_run_records.append(record)

        if setup_specific_records:
            setup_df = pd.DataFrame(setup_specific_records)
            setup_df.to_csv(exp_setup_results_path, index=False)
            logger.info(f"Results for setup '{exp_setup_name}' saved to {exp_setup_results_path}")

    if all_experiment_run_records:
        final_df = pd.DataFrame(all_experiment_run_records)
        unified_csv_path = output_dir / f"{dataset_cfg['name']}_vary_levels_all_setups_summary.csv"
        final_df.to_csv(unified_csv_path, index=False)
        logger.info(f"Unified results for all setups saved to {unified_csv_path}")


        if not final_df.empty:
            plot_metric_vs_l0_param(final_df, config, output_dir, metric_to_plot="recall_mean", ylabel="Mean Recall")
            plot_metric_vs_l0_param(final_df, config, output_dir, metric_to_plot="overall_latency_ms_mean", ylabel="Mean Overall Latency (ms)")
            plot_level_scan_times(final_df, config, output_dir)

    else:
        logger.info("No results generated across all setups.")

    logger.info("Vary Levels (Generalized) experiment finished.")

def plot_metric_vs_l0_param(results_df: pd.DataFrame, config: Dict[str, Any], output_dir: Path, metric_to_plot: str, ylabel: str):
    """Plots a given metric against the L0 sweep parameter for different experiment setups."""
    if results_df.empty or metric_to_plot not in results_df.columns:
        logger.warning(f"Cannot plot '{metric_to_plot}': Data missing or empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_styles = config.get("plot", {}).get("styles", {})

    # Determine L0 sweep key (nprobe or recall_target)
    l0_sweep_key = None
    if "L0_nprobe" in results_df.columns and results_df["L0_nprobe"].notna().any():
        l0_sweep_key = "L0_nprobe"
    elif "L0_recall_target" in results_df.columns and results_df["L0_recall_target"].notna().any():
        l0_sweep_key = "L0_recall_target"

    if not l0_sweep_key:
        logger.warning("L0 sweep parameter (nprobe/recall_target) not found in results. Skipping plot.")
        plt.close(fig)
        return

    for exp_name, group in results_df.groupby("exp_setup_name"):
        style = plot_styles.get(exp_name, {})
        sorted_group = group.sort_values(by=l0_sweep_key)
        ax.plot(
            sorted_group[l0_sweep_key],
            sorted_group[metric_to_plot],
            label=exp_name,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-")
        )

    ax.set_xlabel(f"Innermost Level (L0) Search: {l0_sweep_key.replace('_', ' ').title()}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{config['dataset']['name']}: {ylabel} vs. L0 Search Parameter")
    ax.legend(title="Experiment Setup", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls=":", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.78, 1]) # Adjust for legend

    plot_filename = f"{config['dataset']['name']}_vary_levels_{metric_to_plot}_vs_L0param.png"
    plt.savefig(output_dir / plot_filename, dpi=150)
    logger.info(f"Plot saved: {output_dir / plot_filename}")
    plt.close(fig)


def plot_level_scan_times(results_df: pd.DataFrame, config: Dict[str, Any], output_dir: Path):
    """Plots the breakdown of scan times per level for each experiment setup, at a chosen L0 parameter value."""
    if results_df.empty:
        logger.warning("Cannot plot level scan times: Data missing or empty.")
        return

    l0_sweep_key = None
    if "L0_nprobe" in results_df.columns and results_df["L0_nprobe"].notna().any():
        l0_sweep_key = "L0_nprobe"
    elif "L0_recall_target" in results_df.columns and results_df["L0_recall_target"].notna().any():
        l0_sweep_key = "L0_recall_target"
    if not l0_sweep_key: return


    # Select a representative L0 parameter value for the bar chart (e.g., median or a common high-recall point)
    if results_df[l0_sweep_key].nunique() > 1:
        target_l0_val = np.percentile(results_df[l0_sweep_key].dropna(), 75) # Example: 75th percentile
        # Find closest available value in the df
        available_l0_vals = results_df[l0_sweep_key].dropna().unique()
        closest_l0_val = available_l0_vals[np.abs(available_l0_vals - target_l0_val).argmin()]

        df_subset = results_df[results_df[l0_sweep_key] == closest_l0_val].copy()
        plot_title_suffix = f"(at L0 {l0_sweep_key.split('_')[-1]} ≈ {closest_l0_val:.2f})"
    else:
        df_subset = results_df.copy()
        plot_title_suffix = f"(at L0 {l0_sweep_key.split('_')[-1]} = {df_subset[l0_sweep_key].iloc[0]:.2f})"


    if df_subset.empty:
        logger.warning("No data for selected L0 parameter for level scan time plot.")
        return

    level_time_cols = sorted([col for col in df_subset.columns if col.startswith("level_") and col.endswith("_scan_time_ms_mean")])
    if not level_time_cols:
        logger.warning("No per-level scan time columns found for plotting.")
        return

    max_level_idx = max([int(c.split('_')[1]) for c in level_time_cols])
    num_levels_to_plot = max_level_idx + 1

    plot_data = df_subset.set_index("exp_setup_name")[level_time_cols]

    # Ensure all level columns up to max_level_idx are present, fill with 0 if not (for consistent plotting)
    for i in range(num_levels_to_plot):
        col_name = f"level_{i}_scan_time_ms_mean"
        if col_name not in plot_data.columns:
            plot_data[col_name] = 0
    # Reorder columns for stacked bar plot
    ordered_level_time_cols = [f"level_{i}_scan_time_ms_mean" for i in range(num_levels_to_plot)]
    plot_data = plot_data[ordered_level_time_cols]


    plot_data.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')

    plt.xlabel("Experiment Setup")
    plt.ylabel("Mean Scan Time per Level (ms)")
    plt.title(f"{config['dataset']['name']}: Scan Time Breakdown by Level {plot_title_suffix}")
    plt.xticks(rotation=30, ha='right')
    plt.legend(title="Index Level (L0=innermost)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust for legend

    plot_filename = f"{config['dataset']['name']}_vary_levels_scan_time_breakdown.png"
    plt.savefig(output_dir / plot_filename, dpi=150)
    logger.info(f"Plot saved: {output_dir / plot_filename}")
    plt.close()