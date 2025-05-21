import sys
from pathlib import Path
import logging
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from quake import SearchParams, SearchTimingInfo # Quake specific
from test.experiments.osdi2025 import experiment_utils as common_utils

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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("numa_single_query_exp")


def build_and_save_generic_index(index_cls, build_params_dict, base_vectors, index_file_path: Path):
    if index_file_path.exists():
        logger.info(f"Index {index_file_path} already exists. Skipping build.")
        return
    logger.info(f"Building index {index_file_path}...")
    idx_instance = index_cls()
    # FaissIVF and FaissHNSW expect metric at build, not in build_params dict necessarily
    metric = build_params_dict.pop("metric", "l2")
    idx_instance.build(base_vectors, metric=metric, **build_params_dict)
    index_file_path.parent.mkdir(parents=True, exist_ok=True)
    idx_instance.save(str(index_file_path))
    logger.info(f"Index saved to {index_file_path}")

def get_timing_details(timing_info_obj: SearchTimingInfo | None, is_parent: bool = False) -> dict:
    prefix = "parent_" if is_parent else "child_"
    details = {
        f"{prefix}buffer_init_ns": np.nan,
        f"{prefix}enqueue_ns": np.nan,
        f"{prefix}wait_ns": np.nan,
        f"{prefix}aggregate_ns": np.nan,
        f"{prefix}total_time_ns": np.nan,
    }
    if timing_info_obj:
        details[f"{prefix}buffer_init_ns"] = getattr(timing_info_obj, "buffer_init_time_ns", 0)
        details[f"{prefix}enqueue_ns"] = getattr(timing_info_obj, "job_enqueue_time_ns", 0)
        details[f"{prefix}wait_ns"] = getattr(timing_info_obj, "job_wait_time_ns", 0)
        details[f"{prefix}aggregate_ns"] = getattr(timing_info_obj, "result_aggregate_time_ns", 0)
        details[f"{prefix}total_time_ns"] = getattr(timing_info_obj, "total_time_ns", 0)
    return details


def benchmark_single_query_detailed(
        idx,
        queries_tensor: torch.Tensor,
        gt_tensor: torch.Tensor | None,
        k_val: int,
        search_params_dict: dict,
        index_type: str,
        num_trials: int,
        warmup_runs: int
) -> dict:

    search_kwargs_final = {}
    if index_type == "Quake":
        # For Quake, convert dict to SearchParams object
        # And pop non-SearchParams like 'nprobe' if they are handled differently
        # For single query, SearchParams is directly constructed.
        pass # SearchParams will be constructed per query for Quake
    elif index_type == "SCANN":
        search_kwargs_final = {"leaves_to_search": search_params_dict.get("leaves_to_search", 100)}
    elif index_type == "HNSW":
        search_kwargs_final = {"ef_search": search_params_dict.get("ef_search", 32)}
    elif index_type == "DiskANN":
        search_kwargs_final = {"complexity": search_params_dict.get("complexity", 32)}
    elif index_type == "SVS":
        search_kwargs_final = {"search_window_size": search_params_dict.get("search_window_size", 32)}
    else: # IVF or other Faiss based that might use nprobe directly
        search_kwargs_final = {"nprobe": search_params_dict.get("nprobe", 100)}


    logger.info(f"Warmup ({warmup_runs} queries)...")
    for i in range(min(warmup_runs, len(queries_tensor))):
        q_vec = queries_tensor[i].unsqueeze(0).float()
        if index_type == "Quake":
            idx.search(q_vec, k_val, **search_params_dict)
        else:
            idx.search(q_vec, k_val, **search_kwargs_final)

    trial_aggregated_metrics = []

    for t_idx in range(num_trials):
        per_query_timings_in_trial = []
        recalls_in_trial = []

        for q_idx, q_vec_single in enumerate(queries_tensor):
            q_vec_single = q_vec_single.unsqueeze(0).float()

            search_result = None
            if index_type == "Quake":
                search_result = idx.search(q_vec_single, k_val, **search_params_dict)
            else:
                search_result = idx.search(q_vec_single, k_val, **search_kwargs_final)

            timing_info_child = getattr(search_result, "timing_info", None)
            query_metrics = get_timing_details(timing_info_child, is_parent=False)

            if index_type == "Quake" and timing_info_child: # Quake specific parent info
                timing_info_parent = getattr(timing_info_child, "parent_info", None)
                query_metrics.update(get_timing_details(timing_info_parent, is_parent=True))
            else: # For non-Quake or if no parent_info
                query_metrics.update(get_timing_details(None, is_parent=True))


            if gt_tensor is not None:
                ids_this_query = search_result.ids
                if ids_this_query.ndim == 2 and ids_this_query.shape[0] == 1: # Ensure it's [1, k_found]
                    recall_val = common_utils.quake_compute_recall(
                        ids_this_query,
                        gt_tensor[q_idx].unsqueeze(0),
                        k_val
                    ).item()
                    recalls_in_trial.append(recall_val)

            per_query_timings_in_trial.append(query_metrics)

        # Aggregate for this trial
        trial_df = pd.DataFrame(per_query_timings_in_trial)
        mean_metrics_for_trial = trial_df.mean().to_dict()
        if recalls_in_trial:
            mean_metrics_for_trial["recall"] = np.mean(recalls_in_trial)

        trial_aggregated_metrics.append(mean_metrics_for_trial)
        logger.info(f"  Trial {t_idx+1}/{num_trials}: "
                    f"child_total_ms={mean_metrics_for_trial.get('child_total_time_ns', np.nan)/1e6:.2f}, "
                    f"parent_total_ms={mean_metrics_for_trial.get('parent_total_time_ns', np.nan)/1e6:.2f}"
                    f"recall@{k_val}={mean_metrics_for_trial.get('recall', np.nan):.4f}")


    # Aggregate across trials
    final_metrics_df = pd.DataFrame(trial_aggregated_metrics)
    results_summary = {}
    for col in final_metrics_df.columns:
        # Convert times from ns to ms for mean, keep std in ns then convert if needed, or just report mean in ms.
        if col.endswith("_ns"):
            results_summary[f"{col.replace('_ns', '_ms')}_mean"] = final_metrics_df[col].mean() / 1e6
            results_summary[f"{col.replace('_ns', '_ms')}_std"] = final_metrics_df[col].std() / 1e6
        elif col == "recall":
            results_summary[f"recall_at_{k_val}_mean"] = final_metrics_df[col].mean()
            results_summary[f"recall_at_{k_val}_std"] = final_metrics_df[col].std()
        else: # other metrics if any
            results_summary[f"{col}_mean"] = final_metrics_df[col].mean()
            results_summary[f"{col}_std"] = final_metrics_df[col].std()

    return results_summary


def run_experiment(cfg_path_str: str, output_dir_str: str) -> None:
    config = common_utils.load_config(cfg_path_str)
    dataset_cfg = config["dataset"]
    k_val = dataset_cfg["k"]
    num_queries = dataset_cfg["num_queries"]

    exp_params_cfg = config # Top level items like trials, warmup
    num_trials = exp_params_cfg.get("trials", 5)
    warmup_runs = exp_params_cfg.get("warmup", 10)

    indexes_config_list = config["indexes"]
    output_csv_name = config.get("output", {}).get("results_csv", "numa_single_query_summary.csv")
    force_overwrite_results = config.get("overwrite", False)
    # force_rebuild_indexes = config.get("force_rebuild", False) # If we add a global force_rebuild

    main_output_dir = Path(output_dir_str).expanduser().absolute()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset '{dataset_cfg['name']}' for {num_queries} queries.")
    base_vectors, query_vectors, gt_vectors = common_utils.load_data(
        dataset_cfg["name"],
        dataset_cfg.get("path", ""), # Allow path in dataset_cfg
        nq_override=num_queries
    )

    # Shared Quake base index build logic (specific to this experiment script)
    quake_base_build_params_shared = None
    quake_base_index_file = main_output_dir / "Quake_base_index_single_query.bin" # Distinct name

    for idx_c in indexes_config_list:
        if idx_c["index"] == "Quake":
            temp_params = dict(idx_c.get("build_params", {}))
            # These are stripped for the *base shared* build
            temp_params.pop("num_workers", None)
            temp_params.pop("use_numa", None)
            temp_params.pop("parent_num_workers", None) # Not relevant for single level base
            quake_base_build_params_shared = temp_params
            break # Assume all Quake configs share the same base parameters for nc, metric etc.

    if quake_base_build_params_shared:
        # Check if force_rebuild_indexes is globally set, or if file doesn't exist
        # For now, let's assume force_rebuild_indexes comes from a potential global config key
        # if 'force_rebuild_indexes' not in config: config['force_rebuild_indexes'] = False # Default
        needs_shared_build = not quake_base_index_file.exists() # or config['force_rebuild_indexes']
        if needs_shared_build:
            logger.info(f"Building shared Quake base index at {quake_base_index_file}...")
            # For Quake, build_params_dict for prepare_quake_index needs 'metric' and 'nlist' (nc)
            # Ensure these are present in quake_base_build_params_shared
            if 'metric' not in quake_base_build_params_shared:
                quake_base_build_params_shared['metric'] = dataset_cfg.get('metric', 'l2')
            if 'nc' not in quake_base_build_params_shared and 'nlist' not in quake_base_build_params_shared : # common mistake
                logger.error("Missing 'nc' (nlist) in Quake base build_params.")
                # Fallback or error, for now assume it is there if this block is reached
            else:
                if 'nc' in quake_base_build_params_shared and 'nlist' not in quake_base_build_params_shared:
                    quake_base_build_params_shared['nlist'] = quake_base_build_params_shared.pop('nc')

                common_utils.prepare_quake_index(
                    vecs=base_vectors,
                    build_params_dict=quake_base_build_params_shared,
                    index_file_path=quake_base_index_file,
                    force_rebuild=needs_shared_build, # force if it doesn't exist
                    load=False
                )
                logger.info(f"Shared Quake base index preparation complete at {quake_base_index_file}")
        else:
            logger.info(f"Using existing shared Quake base index from {quake_base_index_file}")


    all_experiment_rows = []

    for idx_conf in indexes_config_list:
        index_name = idx_conf["name"]
        index_type_str = idx_conf["index"]
        current_build_params = dict(idx_conf.get("build_params", {})) # Copy for modification
        current_search_params = dict(idx_conf.get("search_params", {}))

        IndexCls = INDEX_CLASSES.get(index_type_str)
        if IndexCls is None or (IndexCls == Scann and Scann is None):
            logger.warning(f"Index type '{index_type_str}' for '{index_name}' is unavailable. Skipping.")
            continue

        per_index_csv_path = main_output_dir / f"{index_name}_detailed_results.csv"

        if per_index_csv_path.exists() and not force_overwrite_results:
            logger.info(f"Results for {index_name} exist at {per_index_csv_path}. Loading cached.")
            try:
                # Assuming one row per config in these detailed CSVs
                cached_df = pd.read_csv(per_index_csv_path)
                if not cached_df.empty:
                    all_experiment_rows.append(cached_df.iloc[0].to_dict())
            except Exception as e:
                logger.error(f"Could not load cached results for {index_name}: {e}")
            continue

        logger.info(f"--- Processing index: {index_name} (Type: {index_type_str}) ---")

        idx_instance = None
        if index_type_str == "Quake":
            if not quake_base_index_file.exists():
                logger.error(f"Shared Quake base index {quake_base_index_file} not found for {index_name}. Skipping.")
                continue
            idx_instance = QuakeWrapper()
            # Load-time parameters for this specific Quake configuration
            load_time_workers = current_build_params.get("num_workers", 0)
            load_time_use_numa = current_build_params.get("use_numa", False)
            load_time_parent_workers = current_build_params.get("parent_num_workers", 0) # If applicable
            logger.info(f"Loading Quake index '{index_name}' from shared base with: "
                        f"workers={load_time_workers}, numa={load_time_use_numa}, parent_workers={load_time_parent_workers}")
            idx_instance.load(str(quake_base_index_file),
                              num_workers=load_time_workers,
                              use_numa=load_time_use_numa,
                              parent_num_workers=load_time_parent_workers)
        else: # For non-Quake indexes
            generic_idx_file_path = main_output_dir / f"{index_name}_index.bin"
            # Assume force_rebuild_indexes applies here too, or build if not exists
            # build_if_needed_flag = not generic_idx_file_path.exists() # or config['force_rebuild_indexes']
            # if build_if_needed_flag:
            build_and_save_generic_index(IndexCls, current_build_params.copy(), base_vectors, generic_idx_file_path)

            idx_instance = IndexCls()
            load_params_non_quake = current_build_params.copy()
            if index_type_str == "DiskANN": load_params_non_quake.pop("metric", None) # DiskANN doesn't take metric at load
            idx_instance.load(str(generic_idx_file_path), **load_params_non_quake)


        if idx_instance is None:
            logger.error(f"Failed to load or build index {index_name}. Skipping.")
            continue

        benchmark_results = benchmark_single_query_detailed(
            idx_instance, query_vectors, gt_vectors, k_val,
            current_search_params, index_type_str, num_trials, warmup_runs
        )

        # Combine config with results
        final_row_data = {"index_name": index_name, "index_type": index_type_str}
        final_row_data.update({f"build_{k}": v for k,v in current_build_params.items()})
        final_row_data.update({f"search_{k}": v for k,v in current_search_params.items()})
        final_row_data.update(benchmark_results)

        all_experiment_rows.append(final_row_data)
        common_utils.save_results_csv(pd.DataFrame([final_row_data]), per_index_csv_path)

    if not all_experiment_rows:
        logger.warning("No results collected. Skipping final CSV and plots.")
        return

    final_summary_df = pd.DataFrame(all_experiment_rows)
    unified_csv_path = main_output_dir / output_csv_name
    common_utils.save_results_csv(final_summary_df, unified_csv_path)
    logger.info(f"Unified results for NUMA single-query written to {unified_csv_path}")

    # Plotting (using the structure from the original script)
    plot_suffix = Path(output_csv_name).stem

    # Mean Latency (child_total_time_ms_mean) vs. build_num_workers
    if "child_total_time_ms_mean" in final_summary_df.columns and "build_num_workers" in final_summary_df.columns:
        plt.figure(figsize=(10, 6))
        for name_plot, group_data in final_summary_df.groupby("index_name"):
            sorted_group = group_data.sort_values(by="build_num_workers")
            yerr_val = sorted_group["child_total_time_ms_std"] if "child_total_time_ms_std" in sorted_group else None
            plt.errorbar(
                sorted_group["build_num_workers"], # Treat as categorical for distinct points
                sorted_group["child_total_time_ms_mean"],
                yerr=yerr_val,
                marker="o", capsize=5, label=name_plot, linestyle='-'
            )
        # plt.xscale("symlog", base=2) # May not be ideal if num_workers are not powers of 2 or include 0
        plt.xlabel("Build Param: Num Workers (Configuration)")
        plt.ylabel("Mean Single-Query Latency (ms) - Child Total")
        plt.title(f"NUMA Single-Query: Latency vs. Num Workers ({dataset_cfg['name']})")
        plt.legend(title="Index Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0,0,0.8,1]) # Adjust for legend
        plt.savefig(main_output_dir / f"{plot_suffix}_latency_vs_workers.png", dpi=150)
        logger.info(f"Latency plot saved to {main_output_dir / f'{plot_suffix}_latency_vs_workers.png'}")
        plt.close()

    recall_col_name = f"recall_at_{k_val}_mean"
    if recall_col_name in final_summary_df.columns and "build_num_workers" in final_summary_df.columns:
        plt.figure(figsize=(10, 6))
        for name_plot, group_data in final_summary_df.groupby("index_name"):
            sorted_group = group_data.sort_values(by="build_num_workers")
            plt.plot(
                sorted_group["build_num_workers"],
                sorted_group[recall_col_name],
                marker="o", label=name_plot, linestyle='-'
            )
        plt.xlabel("Build Param: Num Workers (Configuration)")
        plt.ylabel(f"Mean Recall@{k_val}")
        plt.title(f"NUMA Single-Query: Recall vs. Num Workers ({dataset_cfg['name']})")
        plt.legend(title="Index Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ylim(0, 1.05) # Typical recall range
        plt.tight_layout(rect=[0,0,0.8,1])
        plt.savefig(main_output_dir / f"{plot_suffix}_recall_vs_workers.png", dpi=150)
        logger.info(f"Recall plot saved to {main_output_dir / f'{plot_suffix}_recall_vs_workers.png'}")
        plt.close()

    logger.info("NUMA Single-Query experiment finished.")