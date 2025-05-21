#!/usr/bin/env python3
"""
NUMA Multi-Query Batch Performance Benchmark
───────────────────────────────────────────
Benchmarks Quake (with varying NUMA/worker settings) and other ANN indexes
for batch-query latency, per-phase breakdown (for Quake), and recall@k.
Results are cached. Outputs per-index CSVs, a unified CSV, and plots.

This script tests how different configurations handle batches of queries,
focusing on Quake's behavior with NUMA-awareness and threading for parallel
processing of queries or internal operations.
"""
import logging # Standard library logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
# time.time() is still used for the benchmark loop due to its specificity.

# Common utilities
import test.experiments.osdi2025.experiment_utils as common_utils

# Quake specific imports
from quake.utils import compute_recall # compute_recall remains useful
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
    "SVS":     Vamana, # Assuming Vamana is the class for "SVS"
}

# Setup logging (can be centralized in common_utils or main runner if preferred)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("numa_multi_query_exp")


def run_experiment(cfg_path_str: str, output_dir_str: str):
    cfg = common_utils.load_config(cfg_path_str)

    dataset_cfg = cfg["dataset"]
    num_queries_to_use = dataset_cfg["num_queries"]
    k_val = dataset_cfg["k"]

    num_trials = cfg.get("trials", 5)
    num_warmup = cfg.get("warmup", 1) # Reduced default warmup, can be configured
    indexes_config_list = cfg["indexes"]
    output_csv_name = cfg.get("output", {}).get("results_csv", "numa_multi_query_results.csv")
    force_overwrite = cfg.get("overwrite", False)
    force_rebuild = cfg.get("force_rebuild", False)

    main_output_dir = Path(output_dir_str)
    main_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset '{dataset_cfg['name']}'...")
    # Use common_utils.load_data, pass num_queries_to_use for slicing
    base_vectors, query_vectors, gt_vectors = common_utils.load_data(
        dataset_cfg["name"],
        dataset_path="", # Assuming load_data handles default path or it's in dataset_cfg
        nq_override=num_queries_to_use
    )

    # --- Build shared Quake base index once (specific to this experiment script) ---
    quake_base_build_params = None
    shared_quake_index_file = main_output_dir / "Quake_shared_base_index.bin" # More descriptive name

    for idx_c in indexes_config_list:
        if idx_c["index"] == "Quake":
            # These are params for the *base* index, stripped of worker/NUMA specifics
            quake_base_build_params = {k: v for k, v in idx_c.get("build_params", {}).items()
                                       if k not in ["num_workers", "use_numa", "parent_num_workers"]}
            break

    if quake_base_build_params:
        if not shared_quake_index_file.exists() or force_rebuild: # Respect overwrite for base index too
            logger.info(f"Building shared Quake base index at {shared_quake_index_file}...")
            # Using common_utils.prepare_index for this specialized build.
            # It requires IndexClass, build_params, base_vectors, path, force_rebuild.
            common_utils.prepare_index(
                IndexClass=QuakeWrapper,
                index_build_params=quake_base_build_params,
                base_vectors=base_vectors,
                index_file_path=shared_quake_index_file,
                force_rebuild=force_rebuild, # Force rebuild if top-level overwrite is true
                save_after_build=True
            )
            logger.info(f"Shared Quake base index saved at {shared_quake_index_file}")
        else:
            logger.info(f"Using existing shared Quake base index from {shared_quake_index_file}")

    all_experiment_rows = []

    for idx_conf in indexes_config_list:
        index_name = idx_conf["name"]
        index_type_str = idx_conf["index"]
        current_build_params = dict(idx_conf.get("build_params", {}))
        current_search_params = dict(idx_conf.get("search_params", {}))

        IndexCls = INDEX_CLASSES.get(index_type_str)
        if IndexCls is None or (IndexCls == Scann and Scann is None): # Handle optional Scann
            logger.warning(f"Index type '{index_type_str}' for '{index_name}' is not available or unknown. Skipping.")
            continue

        per_index_csv_path = main_output_dir / f"{index_name}_results.csv"

        # Determine actual index file path (shared for Quake, specific for others)
        actual_index_file_path = shared_quake_index_file if index_type_str == "Quake" else main_output_dir / f"{index_name}_index.bin"

        if per_index_csv_path.exists() and not force_overwrite:
            logger.info(f"Results for {index_name} exist. Use overwrite:true to rerun. Loading cached.")
            try:
                # Ensure only one row is taken if multiple exist from partial runs.
                cached_df = pd.read_csv(per_index_csv_path)
                if not cached_df.empty:
                    all_experiment_rows.append(cached_df.iloc[0].to_dict())
            except Exception as e:
                logger.error(f"Could not load cached results for {index_name} from {per_index_csv_path}: {e}")
            continue

        logger.info(f"Processing index: {index_name} (Type: {index_type_str})")

        # Load or build index
        idx_instance = None
        if index_type_str == "Quake":
            if not shared_quake_index_file.exists():
                logger.error(f"Shared Quake base index {shared_quake_index_file} not found for {index_name}. Skipping.")
                continue
            idx_instance = QuakeWrapper()
            # These are load-time parameters for Quake specified in each config entry
            quake_load_kwargs = {
                k: current_build_params[k] for k in
                ("use_numa", "num_workers", "parent_num_workers") if k in current_build_params
            }
            logger.info(f"Loading Quake index {index_name} from shared base with params: {quake_load_kwargs}")
            idx_instance.load(str(shared_quake_index_file), **quake_load_kwargs)
        else: # For non-Quake indexes
            # DiskANN needs metric popped from build_params before load, if present during build
            diskann_load_params = current_build_params.copy()
            if index_type_str == "DiskANN": diskann_load_params.pop("metric", None)

            idx_instance = common_utils.prepare_index(
                IndexClass=IndexCls,
                index_build_params=current_build_params,
                base_vectors=base_vectors, # Needed if building
                index_file_path=actual_index_file_path,
                force_rebuild=force_rebuild, # Rebuild this specific non-Quake index if needed
                load_kwargs=diskann_load_params if index_type_str == "DiskANN" else None,
                save_after_build=True
            )

        if idx_instance is None: # Should not happen if logic is correct
            logger.error(f"Failed to load or build index {index_name}. Skipping.")
            continue

        logger.info(f"Warmup for {index_name} ({num_warmup} iterations)...")
        for _ in range(num_warmup): # Batched search for warmup
            _ = idx_instance.search(query_vectors, k_val, **current_search_params)

        # Benchmark Trials (batch search)
        trial_timings = {
            "child_total_ms": [], "parent_total_ms": [], "recall": [],
            "child_buffer_init_ms": [], "child_copy_query_ms": [], "child_enqueue_ms": [],
            "child_wait_ms": [], "child_aggregate_ms": [],
            "parent_buffer_init_ms": [], "parent_copy_query_ms": [], "parent_enqueue_ms": [],
            "parent_wait_ms": [], "parent_aggregate_ms": []
        }

        logger.info(f"Running benchmark for {index_name} ({num_trials} trials)...")
        for t_idx in range(num_trials):
            search_result = idx_instance.search(query_vectors, k_val, **current_search_params)
            timing_info = search_result.timing_info # Assuming this is always present for Quake

            # Collect detailed timings if available (primarily for Quake)
            for key in trial_timings:
                if key.startswith("child_") and hasattr(timing_info, key.replace("_ms", "_time_ns").replace("child_", "")):
                    trial_timings[key].append(getattr(timing_info, key.replace("_ms", "_time_ns").replace("child_", "")) / 1e6)
                elif key.startswith("parent_") and hasattr(timing_info, "parent_info") and timing_info.parent_info:
                    parent_attr_name = key.replace("_ms", "_time_ns").replace("parent_", "")
                    if hasattr(timing_info.parent_info, parent_attr_name):
                        trial_timings[key].append(getattr(timing_info.parent_info, parent_attr_name) / 1e6)
                    else:
                        trial_timings[key].append(0.0) # Parent info might not have all fields
                elif key == "recall" and gt_vectors is not None:
                    pred_ids = search_result.ids
                    current_recall = float(compute_recall(pred_ids, gt_vectors, k_val).mean())
                    trial_timings["recall"].append(current_recall)

            log_msg_trial = f"  [{index_name} trial {t_idx+1}/{num_trials}] "
            if trial_timings["child_total_ms"]: log_msg_trial += f"child_total={trial_timings['child_total_ms'][-1]:.2f}ms "
            if trial_timings["parent_total_ms"] and trial_timings['parent_total_ms'][-1] > 0: log_msg_trial += f"parent_total={trial_timings['parent_total_ms'][-1]:.2f}ms "
            if trial_timings["recall"]: log_msg_trial += f"| recall@{k_val}={trial_timings['recall'][-1]:.4f}"
            logger.info(log_msg_trial)


        # Aggregate metrics from trials
        final_row = {"index": index_name, "n_workers": current_build_params.get("num_workers")}
        for key, values in trial_timings.items():
            if values:
                final_row[f"mean_{key}"] = float(np.mean(values))
                final_row[f"std_{key}"] = float(np.std(values))
            else: # Handle cases where timings might not be available (e.g. non-Quake, or no GT)
                final_row[f"mean_{key}"] = np.nan
                final_row[f"std_{key}"] = np.nan

        # Rename for CSV consistency if needed, e.g., mean_child_total_ms to mean_latency_ms
        if "mean_child_total_ms" in final_row:
            final_row["mean_latency_ms"] = final_row.pop("mean_child_total_ms")
        if "std_child_total_ms" in final_row:
            final_row["std_latency_ms"] = final_row.pop("std_child_total_ms")
        if f"mean_recall" in final_row: # Ensure recall key matches expected format
            final_row[f"recall_at_{k_val}"] = final_row.pop(f"mean_recall")
            if f"std_recall" in final_row: # std for recall might not be needed in final summary
                final_row.pop(f"std_recall")


        all_experiment_rows.append(final_row)
        common_utils.save_results_csv(pd.DataFrame([final_row]), per_index_csv_path) # Save per-index

    # Save unified CSV for all indexes
    if not all_experiment_rows:
        logger.warning("No results collected. Skipping final CSV and plots.")
        return

    final_df = pd.DataFrame(all_experiment_rows)
    unified_csv_path = main_output_dir / output_csv_name
    common_utils.save_results_csv(final_df, unified_csv_path)
    logger.info(f"Unified results written to {unified_csv_path}")

    # Plotting (remains specific to this experiment's needs)
    plot_suffix = Path(output_csv_name).stem

    # Plot 1: Batch total latency (child_total_ms) vs. Num Workers
    if "mean_latency_ms" in final_df.columns and "n_workers" in final_df.columns:
        plt.figure(figsize=(8,6))
        for idx_name_plot in final_df["index"].unique():
            subset = final_df[final_df["index"] == idx_name_plot].sort_values(by="n_workers")
            label = idx_name_plot # Simplified label
            yerr_values = subset["std_latency_ms"] if "std_latency_ms" in subset else None
            plt.errorbar(
                subset["n_workers"].astype(int),
                subset["mean_latency_ms"],
                yerr=yerr_values,
                marker="o", capsize=4, label=label, alpha=0.8, linestyle='-'
            )
        # plt.xscale("symlog", base=2) # May not be ideal if n_workers are not powers of 2 or include 0
        plt.xlabel("Num Workers (Configuration)")
        plt.ylabel(f"Mean Batch Query Latency (ms) - Child Total")
        plt.title(f"NUMA Multi-Query: Latency vs. Num Workers ({dataset_cfg['name']})")
        plt.legend(title="Index Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0,0,0.8,1]) # Adjust for legend
        plt.savefig(main_output_dir / f"{plot_suffix}_batch_latency.png", dpi=150)
        logger.info(f"Latency plot saved to {main_output_dir / f'{plot_suffix}_batch_latency.png'}")
        plt.close()

    logger.info("NUMA Multi-Query experiment finished.")