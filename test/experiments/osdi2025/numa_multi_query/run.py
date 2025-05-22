# test/experiments/osdi2025/numa_multi_query/run.py

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

import test.experiments.osdi2025.experiment_utils as common_utils
from quake.utils import compute_recall
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
from quake.index_wrappers.quake import QuakeWrapper # type: ignore

try: from quake.index_wrappers.scann import Scann # type: ignore
except ImportError: Scann = None
try: from quake.index_wrappers.diskann import DiskANNDynamic # type: ignore
except ImportError: DiskANNDynamic = None
try: from quake.index_wrappers.vamana import Vamana # type: ignore
except ImportError: Vamana = None

INDEX_CLASSES = {
    "Quake": QuakeWrapper, "IVF": FaissIVF, "SCANN": Scann,
    "HNSW": FaissHNSW, "DiskANN": DiskANNDynamic, "SVS": Vamana,
}
logger = logging.getLogger("numa_multi_query_orchestrator")

def task_build_index(
        index_config: dict, dataset_cfg: dict, global_run_params: dict,
        index_file_path: Path
):
    index_name = index_config["name"]
    index_type_str = index_config["index"]
    original_omp_threads = os.environ.get("OMP_NUM_THREADS")

    print(f"[{index_name} BUILD_TASK] Process ID: {os.getpid()}. Target file: {index_file_path}")
    # Make a copy to safely pop omp_threads param
    current_build_params = dict(index_config.get("build_params", {}))
    omp_build_threads = str(current_build_params.pop("omp_num_threads_build", "1")) # Pop the param
    os.environ["OMP_NUM_THREADS"] = omp_build_threads
    print(f"[{index_name} BUILD_TASK] Set OMP_NUM_THREADS={omp_build_threads}.")

    base_vectors, _, _ = common_utils.load_data(
        dataset_cfg["name"], nq_override=1)

    IndexCls = INDEX_CLASSES.get(index_type_str)
    if IndexCls is None or (IndexCls == Scann and Scann is None): # type: ignore
        if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
        else: os.environ.pop("OMP_NUM_THREADS", None)
        return {"error": f"Index type '{index_type_str}' unavailable.", "index_file_path": str(index_file_path)}

    print(f"[{index_name} BUILD_TASK] Calling prepare_wrapper_index (build phase)...")
    # current_build_params now does NOT contain omp_num_threads_build
    idx_instance_for_build = common_utils.prepare_wrapper_index(
        IndexCls, index_file_path, base_vectors, current_build_params,
        global_run_params["force_rebuild_indices"], load=False
    )
    del base_vectors

    if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
    else: os.environ.pop("OMP_NUM_THREADS", None)

    if idx_instance_for_build is None :
        return {"error": f"Index build failed for {index_name}.", "index_file_path": str(index_file_path)}

    print(f"[{index_name} BUILD_TASK] Build/Save completed for {index_file_path}.")
    return {"status": "success", "index_file_path": str(index_file_path)}


def task_search_index(
        index_config: dict, dataset_cfg: dict, global_run_params: dict,
        index_file_path: Path
):
    index_name = index_config["name"]
    index_type_str = index_config["index"]
    original_omp_threads = os.environ.get("OMP_NUM_THREADS")

    print(f"[{index_name} SEARCH_TASK] Process ID: {os.getpid()}. Loading index from: {index_file_path}")
    # These params are for loading the index instance, or for search
    current_build_params_for_load = dict(index_config.get("build_params", {}))
    current_search_params = dict(index_config.get("search_params", {}))

    # Pop omp_num_threads_search before passing current_search_params to idx_instance.search
    omp_search_threads = str(current_search_params.pop("omp_num_threads_search", "1"))
    os.environ["OMP_NUM_THREADS"] = omp_search_threads
    print(f"[{index_name} SEARCH_TASK] Set OMP_NUM_THREADS={omp_search_threads}.")

    _, query_vectors, gt_vectors = common_utils.load_data(
        dataset_cfg["name"], nq_override=dataset_cfg.get("num_queries"))

    IndexCls = INDEX_CLASSES.get(index_type_str)
    if IndexCls is None or (IndexCls == Scann and Scann is None): # type: ignore
        if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
        else: os.environ.pop("OMP_NUM_THREADS", None)
        return {"error": f"Index type '{index_type_str}' unavailable for search."}

    if not Path(index_file_path).exists(): # Ensure Path object for exists()
        if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
        else: os.environ.pop("OMP_NUM_THREADS", None)
        return {"error": f"Index file {index_file_path} not found for search."}

    idx_instance = IndexCls()
    load_kwargs = {}
    # Params like num_workers for Quake load are typically in "build_params" section of YAML
    if "num_workers" in current_build_params_for_load: load_kwargs["num_workers"] = current_build_params_for_load.get("num_workers")
    if "use_numa" in current_build_params_for_load: load_kwargs["use_numa"] = current_build_params_for_load.get("use_numa")
    if "parent_num_workers" in current_build_params_for_load: load_kwargs["parent_num_workers"] = current_build_params_for_load.get("parent_num_workers")

    print(f"[{index_name} SEARCH_TASK] Loading index with kwargs: {load_kwargs}")
    idx_instance.load(str(index_file_path), **load_kwargs)

    k_val = global_run_params["k_val"]
    print(f"[{index_name} SEARCH_TASK] Warmup ({global_run_params['num_warmup']} iterations)...")
    # current_search_params now does NOT contain omp_num_threads_search
    for _ in range(global_run_params['num_warmup']):
        _ = idx_instance.search(query_vectors, k_val, **current_search_params)

    trial_latencies_ms = []
    trial_recalls = []
    print(f"[{index_name} SEARCH_TASK] Benchmarking ({global_run_params['num_trials']} trials)...")
    for i in range(global_run_params['num_trials']):
        # current_search_params now does NOT contain omp_num_threads_search
        search_result = idx_instance.search(query_vectors, k_val, **current_search_params)
        latency_ms = np.nan
        timing_info = getattr(search_result, "timing_info", None)
        if timing_info and hasattr(timing_info, "total_time_ns"): latency_ms = getattr(timing_info, "total_time_ns") / 1e6
        elif timing_info and hasattr(timing_info, "child_total_time_ns"): latency_ms = getattr(timing_info, "child_total_time_ns") / 1e6
        trial_latencies_ms.append(latency_ms)

        recall_val = np.nan
        if gt_vectors is not None and hasattr(search_result, "ids"):
            try: recall_val = float(compute_recall(search_result.ids, gt_vectors, k_val).mean())
            except Exception: pass
        trial_recalls.append(recall_val)

    if original_omp_threads is not None: os.environ["OMP_NUM_THREADS"] = original_omp_threads
    else: os.environ.pop("OMP_NUM_THREADS", None)

    print(f"[{index_name} SEARCH_TASK] Individual trial results (Latency, Recall@K={k_val}):")
    for i in range(len(trial_latencies_ms)):
        lat_str = f"{trial_latencies_ms[i]:.3f}ms" if not np.isnan(trial_latencies_ms[i]) else "N/A"
        rec_str = f"{trial_recalls[i]:.4f}" if not np.isnan(trial_recalls[i]) else "N/A"
        print(f"  Trial {i+1}: Latency={lat_str}, Recall={rec_str}")

    final_row = {"index_name_from_process": index_name}
    valid_latencies = [l for l in trial_latencies_ms if not np.isnan(l)]
    valid_recalls = [r for r in trial_recalls if not np.isnan(r)]
    final_row["mean_latency_ms"] = float(np.mean(valid_latencies)) if valid_latencies else np.nan
    final_row["std_latency_ms"] = float(np.std(valid_latencies)) if valid_latencies else np.nan
    final_row[f"mean_recall_at_{k_val}"] = float(np.mean(valid_recalls)) if valid_recalls else np.nan
    final_row[f"std_recall_at_{k_val}"] = float(np.std(valid_recalls)) if valid_recalls else np.nan

    n_threads_or_workers = int(omp_search_threads)
    if index_type_str == "Quake" and "num_workers" in current_build_params_for_load :
        n_threads_or_workers = current_build_params_for_load.get("num_workers", n_threads_or_workers)
    final_row["n_config_concurrency"] = n_threads_or_workers

    print(f"[{index_name} SEARCH_TASK] Benchmark task finished.")
    return final_row

def run_experiment(cfg_path_str: str, output_dir_str: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)7s|%(name)s| %(message)s", datefmt="%H:%M:%S")
    logger.info(f"Orchestrator: Config: {cfg_path_str}, Output: {output_dir_str}")

    cfg = common_utils.load_config(cfg_path_str)
    main_output_dir = Path(output_dir_str); main_output_dir.mkdir(parents=True, exist_ok=True)
    process_logs_dir = main_output_dir / "process_logs"; process_logs_dir.mkdir(parents=True, exist_ok=True)
    individual_index_files_dir = main_output_dir / "individual_indices";
    individual_index_files_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg["dataset"]
    global_run_params = {
        "num_trials": cfg.get("trials", 3), "num_warmup": cfg.get("warmup", 1), "k_val": dataset_cfg["k"],
        "force_overwrite_results": cfg.get("overwrite", False),
        "force_rebuild_indices": cfg.get("force_rebuild", False),
        "enable_glances": cfg.get("enable_glances_monitoring", False)
    }
    indexes_config_list = cfg["indexes"]
    output_csv_name = cfg.get("output", {}).get("results_csv", "numa_multi_query_results.csv") # Changed from single_query

    all_experiment_rows = []
    for idx_conf in indexes_config_list:
        index_name = idx_conf["name"]
        index_file_path = individual_index_files_dir / f"{index_name}_index.bin"
        per_index_final_results_csv = main_output_dir / f"{index_name}_final_results.csv"

        if per_index_final_results_csv.exists() and not global_run_params["force_overwrite_results"]:
            try:
                all_experiment_rows.append(pd.read_csv(per_index_final_results_csv).iloc[0].to_dict())
                logger.info(f"Loaded cached final results for {index_name}.")
                continue
            except Exception as e: logger.warning(f"Could not load cached final results for {index_name}: {e}. Re-running.")

        logger.info(f"Orchestrator: Submitting BUILD task for index: {index_name}")
        build_log_file = process_logs_dir / f"{index_name}_build_process.log"
        build_env_vars = idx_conf.get("build_env_vars", {})
        process_timeout = idx_conf.get("process_timeout", cfg.get("default_process_timeout", 7200))

        build_result = common_utils.run_operation_in_process(
            task_build_index, (idx_conf, dataset_cfg, global_run_params, index_file_path),
            env_vars=build_env_vars, log_file_path=build_log_file, process_name=f"BuildTask_{index_name}",
            timeout_seconds=process_timeout, enable_glances=global_run_params["enable_glances"]
        )

        if not (build_result["status"] == "success" and build_result.get("data", {}).get("status") == "success"):
            err_msg = build_result.get('message', build_result.get('data', {}).get('error', 'Build task failed'))
            logger.error(f"BUILD task for {index_name} failed. Log: {build_result.get('log_file')}. Error: {err_msg}")
            all_experiment_rows.append({"index": index_name, "error": f"Build Failed: {err_msg}", "mean_latency_ms": np.nan, f"mean_recall_at_{global_run_params['k_val']}": np.nan})
            continue

        logger.info(f"BUILD task for {index_name} succeeded. Index file: {build_result.get('data',{}).get('index_file_path')}")
        built_index_file_path = Path(build_result.get("data",{}).get("index_file_path", index_file_path)) # Use confirmed path

        logger.info(f"Orchestrator: Submitting SEARCH task for index: {index_name}")
        search_log_file = process_logs_dir / f"{index_name}_search_process.log"
        search_env_vars = idx_conf.get("search_env_vars", {})

        search_process_result = common_utils.run_operation_in_process(
            task_search_index, (idx_conf, dataset_cfg, global_run_params, built_index_file_path),
            env_vars=search_env_vars, log_file_path=search_log_file, process_name=f"SearchTask_{index_name}",
            timeout_seconds=process_timeout, enable_glances=global_run_params["enable_glances"]
        )

        if search_process_result["status"] == "success" and isinstance(search_process_result["data"], dict) and "error" not in search_process_result["data"]:
            logger.info(f"Search task {index_name} succeeded. Log: {search_process_result.get('log_file')}. Metrics: {search_process_result.get('metrics_file')}")
            result_data_row = search_process_result["data"]; result_data_row["index"] = index_name
            all_experiment_rows.append(result_data_row)
            common_utils.save_results_csv(pd.DataFrame([result_data_row]), per_index_final_results_csv)
        else:
            err_msg = search_process_result.get('message', search_process_result.get('data', {}).get('error','Search task failed'))
            logger.error(f"Search task {index_name} failed. Log: {search_process_result.get('log_file')}. Metrics: {search_process_result.get('metrics_file')}. Error: {err_msg}")
            all_experiment_rows.append({"index": index_name, "error": f"Search Failed: {err_msg}", "mean_latency_ms": np.nan, f"mean_recall_at_{global_run_params['k_val']}": np.nan})

    if not all_experiment_rows: logger.warning("No results. Skipping CSV/plots."); return
    final_df = pd.DataFrame(all_experiment_rows)
    if 'index' not in final_df.columns and 'index_name_from_process' in final_df.columns:
        final_df.rename(columns={'index_name_from_process': 'index'}, inplace=True)
    final_df['index'] = final_df['index'].fillna(final_df.get('index_name_from_process', "unknown_index"))

    unified_csv_path = main_output_dir / output_csv_name
    common_utils.save_results_csv(final_df, unified_csv_path); logger.info(f"Unified results: {unified_csv_path}")

    plot_suffix = Path(output_csv_name).stem
    plot_x_param = "n_config_concurrency"
    # Ensure 'index' column is valid before attempting groupby for plotting
    plot_df = final_df.dropna(subset=['index', plot_x_param]).copy()
    if not plot_df.empty:
        plot_df.loc[:, plot_x_param] = pd.to_numeric(plot_df[plot_x_param], errors='coerce').fillna(-1).astype(int)
        k_val_plot = global_run_params['k_val']

        if "mean_latency_ms" in plot_df.columns:
            try:
                plt.figure(figsize=(8,6));
                for name, group in plot_df.groupby("index"):
                    group = group.sort_values(by=plot_x_param)
                    plt.errorbar(group[plot_x_param], group["mean_latency_ms"], yerr=group.get("std_latency_ms"), marker="o", label=name)
                plt.xlabel("Configured Concurrency"); plt.ylabel("Mean Latency (ms)"); plt.title(f"Latency ({dataset_cfg['name']})")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout(rect=[0,0,0.8,1])
                plt.savefig(main_output_dir / f"{plot_suffix}_latency.png"); plt.close(); logger.info(f"Latency plot saved.")
            except Exception as e: logger.error(f"Latency plot error: {e}")

        recall_col = f"mean_recall_at_{k_val_plot}"
        if recall_col in plot_df.columns:
            try:
                plt.figure(figsize=(8,6));
                for name, group in plot_df.groupby("index"):
                    group = group.sort_values(by=plot_x_param)
                    plt.errorbar(group[plot_x_param], group[recall_col], yerr=group.get(f"std_recall_at_{k_val_plot}"), marker="o", label=name)
                plt.xlabel("Configured Concurrency"); plt.ylabel(f"Mean Recall@{k_val_plot}"); plt.title(f"Recall ({dataset_cfg['name']})")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.tight_layout(rect=[0,0,0.8,1]); plt.ylim(0, 1.05)
                plt.savefig(main_output_dir / f"{plot_suffix}_recall.png"); plt.close(); logger.info(f"Recall plot saved.")
            except Exception as e: logger.error(f"Recall plot error: {e}")
    else: logger.warning("Not enough data for plotting after filtering.")
    logger.info("Experiment orchestration finished.")