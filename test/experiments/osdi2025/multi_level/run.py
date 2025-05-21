#!/usr/bin/env python3
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import sys
import yaml
from quake import QuakeIndex, IndexBuildParams, SearchParams, SearchTimingInfo
from test.experiments.osdi2025 import experiment_utils as common_utils
from quake.utils import compute_recall

logger = logging.getLogger("multi_level_debug_exp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def build_or_verify_l0(
        l0_key_name: str,
        l0_build_config: Dict[str, Any],
        base_vectors: torch.Tensor,
        dataset_metric: str,
        output_dir: Path,
        force_rebuild: bool,
        quake_verbose_log_path: Path # Added for redirection
) -> Path:
    index_file_path = output_dir / f"l0_index_{l0_key_name}.bin"
    l0_type = l0_build_config.get("type", "Quake")
    if l0_type != "Quake":
        raise NotImplementedError(f"L0 index type '{l0_type}' not yet supported.")

    build_params_dict = copy.deepcopy(l0_build_config.get("build_params", {}))
    if 'metric' not in build_params_dict:
        build_params_dict['metric'] = dataset_metric

    if not index_file_path.exists() or force_rebuild:
        logger.info(f"Building L0 '{l0_key_name}' (nlist={build_params_dict.get('nlist')}) -> {index_file_path}")
        with common_utils.redirect_all_stdout_to_file(quake_verbose_log_path, mode='a'):
            _ = common_utils.prepare_quake_index(
                vecs=base_vectors,
                build_params_dict=build_params_dict,
                index_file_path=index_file_path,
                force_rebuild=True,
                load=False
            )
    else:
        logger.info(f"Using existing L0 '{l0_key_name}' from {index_file_path}")
    return index_file_path

def construct_full_search_params(
        l0_variant_params: Dict[str, Any],
        parent_levels_config: List[Dict[str, Any]]
) -> SearchParams:
    current_sp = common_utils.create_search_params(**l0_variant_params)
    previous_level_sp = current_sp
    for parent_cfg in parent_levels_config:
        parent_search_dict = parent_cfg.get("search_params", {"nprobe": 1})
        parent_level_sp = common_utils.create_search_params(**parent_search_dict)
        previous_level_sp.parent_params = parent_level_sp
        previous_level_sp = parent_level_sp
    return current_sp

def extract_recursive_timing_stats(
        timing_info: SearchTimingInfo | None, max_levels: int
) -> Dict[str, Any]:
    stats = {}
    current_ti = timing_info
    for i in range(max_levels):
        level_prefix = f"L{i}"
        if current_ti:
            stats[f"{level_prefix}_total_scan_ms"] = current_ti.total_time_ns / 1e6
            stats[f"{level_prefix}_partitions_scanned"] = getattr(current_ti, "partitions_scanned", np.nan)
            current_ti = current_ti.parent_info
        else:
            stats[f"{level_prefix}_total_scan_ms"] = np.nan
            stats[f"{level_prefix}_partitions_scanned"] = np.nan
    return stats

def benchmark_hierarchy_search_point(
        index_instance: QuakeIndex,
        queries: torch.Tensor,
        ground_truth: np.ndarray | None,
        full_hierarchical_sp: SearchParams,
        num_actual_levels: int,
        trials: int,
        warmup: int,
        quake_verbose_log_path: Path # Added for redirection
) -> Dict[str, Any]:
    trial_latencies_ns = []
    trial_recalls = []
    trial_per_level_stats_list = []
    k_val_for_recall = full_hierarchical_sp.k

    with common_utils.redirect_all_stdout_to_file(quake_verbose_log_path, mode='a'):
        for _ in range(warmup):
            index_instance.search(queries, full_hierarchical_sp)
        for _ in range(trials):
            t_start_ns = time.perf_counter_ns()
            search_result = index_instance.search(queries, full_hierarchical_sp)
            t_elapsed_ns = time.perf_counter_ns() - t_start_ns
            trial_latencies_ns.append(t_elapsed_ns)

            if ground_truth is not None and search_result.ids is not None:
                max_k_recall = min(k_val_for_recall, ground_truth.shape[1], search_result.ids.shape[1])
                recall_val = compute_recall(
                    search_result.ids[:, :max_k_recall],
                    ground_truth[:, :max_k_recall],max_k_recall
                ).mean().item() if max_k_recall > 0 else np.nan
                trial_recalls.append(recall_val)
            else:
                trial_recalls.append(np.nan)
            trial_per_level_stats_list.append(
                extract_recursive_timing_stats(search_result.timing_info, num_actual_levels)
            )

    mean_latency_ms = np.mean(trial_latencies_ns) / 1e6 if trial_latencies_ns else np.nan
    qps_mean = len(queries) / (mean_latency_ms / 1000) if mean_latency_ms > 0 else np.inf
    results = {
        "qps_mean": qps_mean,
        "recall_mean": np.nanmean(trial_recalls) if trial_recalls else np.nan,
        "overall_latency_ms_mean": mean_latency_ms,
    }
    if trial_per_level_stats_list:
        df_level_stats = pd.DataFrame(trial_per_level_stats_list)
        for col in df_level_stats.columns:
            results[f"{col}_mean"] = df_level_stats[col].mean()
    return results

def run_experiment(config_path_str: str, output_dir_str: str):
    config = common_utils.load_config(config_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    quake_verbose_log = output_dir / "quake_internal_details.log"
    if quake_verbose_log.exists(): quake_verbose_log.unlink() # Clear log at start of run

    logger.info(f"Loaded configuration from: {config_path_str}")
    dataset_cfg = config["dataset"]
    overwrite_cfg = config["overwrite"]
    l0_definitions = config.get("l0_definitions", {})
    global_l0_sweep_cfg = config["global_l0_search_sweep"]
    index_hierarchies_cfg = config["index_hierarchies"]
    run_settings = config["run_settings"]
    original_omp_threads = os.environ.get("OMP_NUM_THREADS")
    plot_styles = config.get("plot_styles", {})

    logger.info(f"Dataset: {dataset_cfg['name']}, Metric: {dataset_cfg['metric']}, NQ: {dataset_cfg['nq']}")
    logger.info(f"Overwrite index_files: {overwrite_cfg['index_files']}, Overwrite results_csv: {overwrite_cfg['results_csv']}")
    logger.info(f"Verbose Quake output will be logged to: {quake_verbose_log}")


    base_vectors, queries, gt_numpy = common_utils.load_data(
        dataset_cfg["name"], dataset_cfg.get("path", "data/"), nq_override=dataset_cfg.get("nq")
    )
    logger.info(f"Loaded data: Base vectors {base_vectors.shape}, Queries {queries.shape}, GT {gt_numpy.shape if gt_numpy is not None else 'None'}")

    built_l0_files = {}
    all_experiment_records = []

    omp_build_threads = str(run_settings.get("omp_threads_build", os.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = omp_build_threads
    logger.info(f"BUILD PHASE: Set OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    unique_l0s_to_process = {}
    for l0_ref_name, l0_conf in l0_definitions.items():
        unique_l0s_to_process[l0_ref_name] = l0_conf
    for hierarchy_cfg in index_hierarchies_cfg:
        if "custom_l0" in hierarchy_cfg:
            unique_l0s_to_process[f"custom_for_{hierarchy_cfg['name']}"] = hierarchy_cfg["custom_l0"]
    logger.info(f"Found {len(unique_l0s_to_process)} unique L0 configurations.")
    for l0_key, l0_build_conf in unique_l0s_to_process.items():
        built_l0_files[l0_key] = build_or_verify_l0(
            l0_key, l0_build_conf, base_vectors, dataset_cfg["metric"],
            output_dir, overwrite_cfg["index_files"], quake_verbose_log # Pass log path
        )

    l0_search_variants = common_utils.expand_search_sweep(global_l0_sweep_cfg)
    global_k_val = global_l0_sweep_cfg["k"]
    global_batch_size = global_l0_sweep_cfg.get("batch_size", run_settings.get("default_batch_size", 1))
    for variant in l0_search_variants:
        if "k" not in variant: variant["k"] = global_k_val
        if "batch_size" not in variant: variant["batch_size"] = global_batch_size
    logger.info(f"Global L0 search sweep: {len(l0_search_variants)} variants. k={global_k_val}, batch_size={global_batch_size}.")

    omp_search_threads = str(run_settings.get("omp_threads_search", "1"))
    os.environ["OMP_NUM_THREADS"] = omp_search_threads
    logger.info(f"SEARCH PHASE: Set OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    for hierarchy_cfg in index_hierarchies_cfg:
        hierarchy_name = hierarchy_cfg["name"]
        logger.info(f"===== Processing Hierarchy: {hierarchy_name} =====")
        l0_file_to_load, l0_build_details_for_reporting, l0_load_params_dict = None, {}, {}

        if "l0_ref" in hierarchy_cfg:
            ref_name = hierarchy_cfg["l0_ref"]
            l0_file_to_load = built_l0_files.get(ref_name)
            l0_build_details_for_reporting = l0_definitions.get(ref_name, {}).get("build_params", {})
        elif "custom_l0" in hierarchy_cfg:
            custom_l0_key = f"custom_for_{hierarchy_name}"
            l0_file_to_load = built_l0_files.get(custom_l0_key)
            l0_build_details_for_reporting = hierarchy_cfg["custom_l0"].get("build_params", {})
        if not l0_file_to_load:
            logger.error(f"L0 file path not found for {hierarchy_name}. Skipping.")
            continue

        l0_load_params_dict = {
            "num_workers": l0_build_details_for_reporting.get("num_workers",0),
            "use_numa": l0_build_details_for_reporting.get("use_numa",False),
            "parent_num_workers": 0 # For L0 load itself
        }

        num_actual_levels = 1 + len(hierarchy_cfg.get("parent_levels", []))

        current_index = QuakeIndex()
        with common_utils.redirect_all_stdout_to_file(quake_verbose_log, mode='a'):
            current_index.load(str(l0_file_to_load), l0_load_params_dict["num_workers"], l0_load_params_dict["use_numa"], l0_load_params_dict["parent_num_workers"])

        parent_levels_config_for_sp = []
        for i, parent_cfg in enumerate(hierarchy_cfg.get("parent_levels", [])):
            parent_bp = common_utils.create_index_build_params(**parent_cfg["build_params"])
            if not parent_bp.metric: parent_bp.metric = dataset_cfg["metric"]
            with common_utils.redirect_all_stdout_to_file(quake_verbose_log, mode='a'):
                current_index.add_level(parent_bp)
            parent_levels_config_for_sp.append(parent_cfg)

        for l0_variant in l0_search_variants:
            l0_nprobe_val = l0_variant.get("nprobe")
            l0_rt_val = l0_variant.get("recall_target")
            sweep_param_log = f"nprobe={l0_nprobe_val}" if l0_nprobe_val is not None else f"RT={l0_rt_val}"
            logger.info(f"  Testing L0 Sweep: {sweep_param_log} for '{hierarchy_name}'")

            full_sp = construct_full_search_params(l0_variant, parent_levels_config_for_sp)
            metrics = benchmark_hierarchy_search_point(
                current_index, queries, gt_numpy, full_sp,
                num_actual_levels, run_settings["trials"], run_settings["warmup"],
                quake_verbose_log
            )

            record = {
                "hierarchy_name": hierarchy_name,
                "L0_nlist": l0_build_details_for_reporting.get("nlist"),
                "L0_nprobe": l0_nprobe_val, "L0_recall_target": l0_rt_val,
                "L0_k": l0_variant["k"], "L0_batch_size": l0_variant["batch_size"],
            }
            for i, p_cfg in enumerate(hierarchy_cfg.get("parent_levels", [])):
                level_num = i + 1
                record[f"L{level_num}_nlist"] = p_cfg.get("build_params", {}).get("nlist")
                record[f"L{level_num}_search_nprobe"] = p_cfg.get("search_params", {}).get("nprobe")
            record.update(metrics)
            all_experiment_records.append(record)
            logger.info(f"    Done: Recall={metrics.get('recall_mean',0):.4f}, QPS={metrics.get('qps_mean',0):.1f}, Latency={metrics.get('overall_latency_ms_mean',0):.2f}ms")
        del current_index

    if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
    elif "OMP_NUM_THREADS" in os.environ: del os.environ["OMP_NUM_THREADS"]
    logger.info(f"OMP_NUM_THREADS settings reverted.")

    if not all_experiment_records:
        logger.warning("No results collected.")
        return

    final_df = pd.DataFrame(all_experiment_records)
    unified_csv_path = output_dir / f"{dataset_cfg['name']}_summary.csv" # Simplified name
    common_utils.save_results_df(final_df, unified_csv_path)

    plot_qps_vs_recall(final_df, plot_styles, dataset_cfg, output_dir)
    plot_level_scan_time_breakdown(final_df, plot_styles, dataset_cfg, global_l0_sweep_cfg, output_dir)

    logger.info(f"===== Experiment Finished: {config_path_str} =====")
    logger.info(f"Output directory: {output_dir.resolve()}")

    logger.info("\n--- Per-Hierarchy Per-L0-Parameter Summary ---")
    for name, group in final_df.groupby('hierarchy_name'):
        logger.info(f"\nResults for Hierarchy: {name}")
        sweep_param = "L0_nprobe" if "L0_nprobe" in group.columns and group["L0_nprobe"].notna().any() else "L0_recall_target"
        summary_cols = [sweep_param, 'recall_mean', 'overall_latency_ms_mean', 'qps_mean']
        # Filter out columns that might be all NaN if a sweep param wasn't used
        summary_cols = [col for col in summary_cols if col in group.columns and group[col].notna().any()]
        if not summary_cols or sweep_param not in summary_cols : # if sweep_param is all NaN
            logger.info(group[['recall_mean', 'overall_latency_ms_mean', 'qps_mean']].to_string(float_format="%.4f"))
        else:
            logger.info(group[summary_cols].sort_values(by=sweep_param).to_string(float_format="%.4f"))
    logger.info("-----------------------------------------------\n")


def plot_qps_vs_recall(results_df: pd.DataFrame, plot_styles: Dict[str, Any], dataset_cfg: dict, output_dir: Path):
    if results_df.empty: return
    plt.figure(figsize=(10, 6))
    for hierarchy_name, group in results_df.groupby("hierarchy_name"):
        style = plot_styles.get(hierarchy_name, {})
        sorted_group = group.sort_values(by="recall_mean").dropna(subset=["qps_mean", "recall_mean"])
        if sorted_group.empty: continue
        plt.plot(
            sorted_group["recall_mean"], sorted_group["qps_mean"], label=hierarchy_name,
            marker=style.get("marker", "o"), color=style.get("color"), linestyle=style.get("linestyle", "-")
        )
    plt.xlabel(f"Mean Recall@{results_df['L0_k'].iloc[0] if 'L0_k' in results_df.columns and results_df['L0_k'].notna().any() else 'k'}")
    plt.ylabel("Mean QPS")
    plt.title(f"{dataset_cfg['name']}: QPS vs. Recall")
    plt.legend(title="Hierarchy Config", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.xlim(left=max(0.0, results_df["recall_mean"].min() - 0.05 if results_df["recall_mean"].notna().any() else 0.0), right=1.05)
    if results_df["qps_mean"].notna().any() and results_df["qps_mean"].max() > 10 * results_df["qps_mean"].min() and results_df["qps_mean"].min() > 0:
        plt.yscale("log")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_dir / f"{dataset_cfg['name']}_qps_vs_recall.png", dpi=150)
    plt.close()


def find_best_op_point_for_recall_plot(
        hierarchy_data: pd.DataFrame,
        target_recall: float,
        l0_sweep_key: str
):
    # Find points meeting or exceeding target recall
    valid_points = hierarchy_data[hierarchy_data['recall_mean'] >= target_recall]

    if valid_points.empty:
        # If none meet, pick the one with highest recall
        if not hierarchy_data.empty:
            best_effort_point = hierarchy_data.loc[hierarchy_data['recall_mean'].idxmax()]
            logger.warning(
                f"For breakdown plot of '{best_effort_point['hierarchy_name']}', no L0 param met target recall {target_recall:.3f}. "
                f"Using best achieved: R={best_effort_point['recall_mean']:.3f} "
                f"(L0 {l0_sweep_key.split('_')[-1]}={best_effort_point[l0_sweep_key]})"
            )
            return best_effort_point
        return None

    # Among valid points, pick one with highest QPS, then lowest latency
    best_point = valid_points.sort_values(by=['qps_mean', 'overall_latency_ms_mean'], ascending=[False, True]).iloc[0]
    return best_point

def plot_level_scan_time_breakdown(
        results_df: pd.DataFrame,
        plot_styles: Dict[str, Any],
        dataset_cfg: dict,
        l0_sweep_cfg: Dict[str, Any],
        output_dir: Path,
        target_recall_for_plot: float = 0.9 # New parameter
):
    if results_df.empty:
        logger.warning("Level scan time breakdown plot: No data.")
        return

    l0_sweep_key = "L0_nprobe" if "nprobes" in l0_sweep_cfg else "L0_recall_target"

    points_for_plot = []
    for hierarchy_name, group in results_df.groupby("hierarchy_name"):
        op_point = find_best_op_point_for_recall_plot(group, target_recall_for_plot, l0_sweep_key)
        if op_point is not None:
            points_for_plot.append(op_point)

    if not points_for_plot:
        logger.warning(f"No suitable data points found for any hierarchy at target recall ~{target_recall_for_plot} for breakdown plot.")
        return

    df_subset = pd.DataFrame(points_for_plot)
    plot_title_suffix = f"(at or above R≈{target_recall_for_plot:.2f})"

    max_level = max([-1] + [int(c.split('_')[0][1:]) for c in df_subset.columns if c.startswith("L") and "_total_scan_ms_mean" in c])
    if max_level == -1:
        logger.warning("No per-level scan time columns found for breakdown plot.")
        return

    level_cols = [f"L{i}_total_scan_ms_mean" for i in range(max_level + 1)]
    for col in level_cols:
        if col not in df_subset.columns: df_subset[col] = 0.0

    plot_data = df_subset.set_index("hierarchy_name")[level_cols].fillna(0)
    plot_data.columns = [f"L{i} Scan (ms)" for i in range(max_level + 1)]

    ax = plot_data.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis_r')

    # Add text for achieved recall on top of each bar
    for i, hierarchy_name in enumerate(df_subset["hierarchy_name"]):
        row_data = df_subset.iloc[i]
        total_height = plot_data.loc[hierarchy_name].sum()
        actual_recall = row_data.get('recall_mean', np.nan)
        if pd.notna(actual_recall):
            ax.text(i, total_height * 1.01, f"R={actual_recall:.3f}", ha='center', va='bottom', fontsize=8, color='dimgrey')

    plt.xlabel("Hierarchy Configuration")
    plt.ylabel("Mean Scan Time per Level (ms)")
    plt.title(f"{dataset_cfg['name']}: Scan Time Breakdown {plot_title_suffix}")
    plt.xticks(rotation=25, ha='right')
    plt.legend(title="Level", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_dir / f"{dataset_cfg['name']}_level_scan_breakdown_at_recall_target.png", dpi=150)
    plt.close()
    logger.info(f"Level scan time breakdown (at R~{target_recall_for_plot:.2f}) plot saved.")