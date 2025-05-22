# File: test/experiments/osdi2025/common/experiment_utils.py
import time
import yaml
import logging
from pathlib import Path
import contextlib
import sys
from typing import Dict, Any, List, Callable, Union, Optional, Tuple
import os
import multiprocessing
import subprocess
import threading
import json
import traceback
import signal
from datetime import datetime, timezone # For Glances metrics collection

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
    if index_file_path.exists() and not force_rebuild:
        if load:
            logger.info(f"Loading index from {index_file_path}")
            idx.load(str(index_file_path), num_workers_load, use_numa, num_parent_workers_load)
    else:
        logger.info(f"Building index -> {index_file_path}")
        bp = IndexBuildParams()
        for key, value in build_params_dict.items():
            setattr(bp, key, value)

        idx.build(vecs, torch.arange(len(vecs)), bp)
        index_file_path.parent.mkdir(parents=True, exist_ok=True)
        idx.save(str(index_file_path))
        logger.info(f"Index saved to {index_file_path}")
    return idx


def prepare_wrapper_index(
        IndexClass: type,
        index_file_path: Path,
        base_vectors: torch.Tensor,
        build_params: Dict[str, Any],
        force_rebuild: bool = False,
        load: bool = True
) -> object: # Returns an instance of the IndexClass

    idx_instance = IndexClass()
    index_file_path.parent.mkdir(parents=True, exist_ok=True)
    build_params_copy = build_params.copy()

    if index_file_path.exists() and not force_rebuild:
        logger.info(f"Index file {index_file_path} exists and force_rebuild is False.")
        if load:
            logger.info(f"Loading index from {index_file_path}")
            load_kwargs = {}
            if "num_workers" in build_params_copy:
                load_kwargs["num_workers"] = build_params_copy.get("num_workers")
            if "use_numa" in build_params_copy:
                load_kwargs["use_numa"] = build_params_copy.get("use_numa")

            idx_instance.load(str(index_file_path), **load_kwargs)
            logger.info(f"Loaded index from {index_file_path} with load_kwargs: {load_kwargs}")
        else:
            logger.info(f"Index file {index_file_path} exists, but load=False. Returning new, non-loaded instance.")
    else:
        if force_rebuild and index_file_path.exists():
            logger.info(f"Force rebuilding index at {index_file_path}")
        else:
            logger.info(f"Building index at {index_file_path} (file did not exist).")

        idx_instance.build(base_vectors, **build_params_copy)
        idx_instance.save(str(index_file_path))
        logger.info(f"Built and saved index to {index_file_path}")

        if load:
            logger.info(f"Index built. Instance is considered ready as per build_params (load=True).")

    return idx_instance

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

def create_index_build_params(**attrs) -> IndexBuildParams:
    """Creates a Quake IndexBuildParams object from dictionary attributes."""
    bp = IndexBuildParams()
    if 'nc' in attrs and 'nlist' not in attrs: # Allow 'nc' as an alias for 'nlist'
        attrs['nlist'] = attrs.pop('nc')

    for k, v in attrs.items():
        if hasattr(bp, k):
            setattr(bp, k, v)
        else:
            logger.warning(f"Attribute '{k}' not found in IndexBuildParams. Ignoring.")
    return bp

def save_results_df(records: list, output_path: Path):
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}") # Already logs

@contextlib.contextmanager
def redirect_all_stdout_to_file(filepath: Union[Path, str], mode: str = 'a'):
    original_python_stdout = sys.stdout
    original_c_stdout_fd = -1
    saved_c_stdout_fd = -1
    target_file = None
    can_redirect_c_stdout = hasattr(sys.stdout, 'fileno')

    try:
        if can_redirect_c_stdout:
            original_c_stdout_fd = sys.stdout.fileno()
            saved_c_stdout_fd = os.dup(original_c_stdout_fd)
        target_file = open(filepath, mode)
        if can_redirect_c_stdout:
            os.dup2(target_file.fileno(), original_c_stdout_fd)
        sys.stdout = target_file
        yield
    except Exception:
        raise
    finally:
        if sys.stdout == target_file and target_file:
            sys.stdout.flush()
        if can_redirect_c_stdout and original_c_stdout_fd != -1 and saved_c_stdout_fd != -1:
            os.dup2(saved_c_stdout_fd, original_c_stdout_fd)
        sys.stdout = original_python_stdout
        if target_file:
            target_file.close()
        if saved_c_stdout_fd != -1:
            os.close(saved_c_stdout_fd)

_GLANCES_REFRESH_SEC = 1
_GLANCES_MB = 1_048_576.0

def start_glances_minimal(tmp_json_path: Path) -> subprocess.Popen:
    cmd = [
        "glances", "-q", f"-t{_GLANCES_REFRESH_SEC}", "--export", "json",
        "--export-json-file", str(tmp_json_path), "--disable-plugin", "all",
        "--enable-plugin", "cpu,mem,diskio,gpu",
        # Consider adding process monitoring if useful: "--enable-plugin", "processcount,processlist"
        # and then filter by PID if possible with glances export, though might be complex.
        # For now, system-wide A,B,C,D metrics.
    ]
    # print(f"Starting Glances: {' '.join(cmd)}") # Goes to main process stdout if called from there
    # If called from child, it'll go to child's log via redirection.
    # Use preexec_fn=os.setsid to create a new process group for Glances
    glances_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)
    time.sleep(2) # Allow Glances to start and write initial JSON
    if glances_process.poll() is not None: # Check if Glances started successfully
        stderr_output = glances_process.stderr.read() if glances_process.stderr else "No stderr."
        raise RuntimeError(f"Glances failed to start. Exit code: {glances_process.returncode}. Stderr: {stderr_output}")
    return glances_process

def stop_glances_minimal(glances_process: Optional[subprocess.Popen]):
    if glances_process and glances_process.poll() is None:
        try:
            os.killpg(os.getpgid(glances_process.pid), signal.SIGTERM) # Send SIGTERM to the entire process group
            glances_process.wait(timeout=3)
        except ProcessLookupError: pass # Process already dead
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(glances_process.pid), signal.SIGKILL) # Force kill
                glances_process.wait(timeout=1)
            except Exception: pass # Best effort
        except Exception: pass # Other errors during stop

def read_glances_json_minimal(tmp_json_path: Path) -> List[dict]:
    if not tmp_json_path.exists(): return []
    text = tmp_json_path.read_text(encoding="utf-8").strip()
    if not text: return []
    try: # Glances typically writes a list of JSON objects, or one object per file if snapshotting fast
        data = json.loads(text)
        return data if isinstance(data, list) else [data] # Ensure it's a list
    except json.JSONDecodeError: # Handle cases where file might not be complete JSON list
        records = []
        for line in text.splitlines(): # Try parsing line by line if not a single JSON doc
            try: records.append(json.loads(line))
            except json.JSONDecodeError: pass # Ignore malformed lines
        return records

def extract_metrics_minimal(glances_record: Dict[str, Any]) -> Dict[str, Any]:
    # Simplified extraction for A, B, C, D
    cpu = glances_record.get("cpu", {})
    mem = glances_record.get("mem", {})
    diskio_list = glances_record.get("diskio", [])
    gpu_list = glances_record.get("gpu", [])

    read_bytes = sum(d.get("read_bytes", 0) for d in diskio_list if isinstance(d, dict))
    write_bytes = sum(d.get("write_bytes", 0) for d in diskio_list if isinstance(d, dict))

    # Assuming first GPU, if any. User script used 'proc' or 'utilization'.
    gpu_util = np.nan
    if gpu_list and isinstance(gpu_list[0], dict):
        gpu_info = gpu_list[0]
        gpu_util = gpu_info.get("proc", gpu_info.get("utilization", np.nan))

    return {
        "cpu_percent": cpu.get("total", cpu.get("user", 0) + cpu.get("system", 0)), # 'total' is often available
        "mem_percent": mem.get("percent", np.nan),
        "disk_read_mb_per_s": read_bytes / _GLANCES_MB / _GLANCES_REFRESH_SEC, # Approximate rate
        "disk_write_mb_per_s": write_bytes / _GLANCES_MB / _GLANCES_REFRESH_SEC, # Approximate rate
        "gpu_percent": gpu_util,
    }
# --- End Glances Helpers ---

def _glances_metric_collector_thread(
        glances_tmp_json_path: Path,
        metrics_output_ndjson_path: Path,
        stop_event: threading.Event
):
    # print(f"Glances metrics collector thread started. Output: {metrics_output_ndjson_path}") # Debug
    with open(metrics_output_ndjson_path, "w") as mof:
        while not stop_event.wait(timeout=_GLANCES_REFRESH_SEC): # Check event periodically
            try:
                glances_data_list = read_glances_json_minimal(glances_tmp_json_path)
                # Glances --export-json-file overwrites the file. So we read the whole content.
                # If it's a list, it might be multiple snapshots if glances is very fast relative to our read.
                # Usually, it's one snapshot.
                for record in glances_data_list: # Process each snapshot found
                    extracted = extract_metrics_minimal(record)
                    ts = datetime.now(timezone.utc).isoformat()
                    mof.write(json.dumps({"timestamp": ts, **extracted}) + "\n")
                mof.flush()
            except Exception as e:
                # This print will go to the main process log for the child process
                print(f"Glances metrics collector thread error: {e}", file=sys.stderr) # Use stderr
    # print(f"Glances metrics collector thread stopped for {metrics_output_ndjson_path}") # Debug

def _process_target_wrapper(
        target_func: Callable, log_file_path: Path, env_vars: Optional[Dict[str, str]],
        result_queue: multiprocessing.Queue, process_name_for_log: str,
        enable_glances_monitoring: bool, args: Tuple, kwargs: Dict[str, Any]
):
    current_process_name = multiprocessing.current_process().name
    glances_proc: Optional[subprocess.Popen] = None
    glances_tmp_json_file = log_file_path.with_suffix(".glances-temp.json")
    metrics_output_file = log_file_path.with_suffix(".glances.ndjson")
    metrics_thread: Optional[threading.Thread] = None
    stop_glances_collection_event = threading.Event()

    try:
        if env_vars: # Set environment variables first
            for k, v in env_vars.items(): os.environ[str(k)] = str(v)

        if enable_glances_monitoring:
            try:
                print(f"[{current_process_name}] Starting Glances monitoring... Temp JSON: {glances_tmp_json_file}")
                glances_proc = start_glances_minimal(glances_tmp_json_file)
                metrics_thread = threading.Thread(
                    target=_glances_metric_collector_thread,
                    args=(glances_tmp_json_file, metrics_output_file, stop_glances_collection_event),
                    daemon=True
                )
                metrics_thread.start()
                print(f"[{current_process_name}] Glances monitoring and metrics collector thread started.")
            except Exception as ge:
                print(f"[{current_process_name}] WARNING: Failed to start Glances: {ge}", file=sys.stderr)
                glances_proc = None # Ensure it's None if start failed
                metrics_thread = None

        # with redirect_all_stdout_to_file(log_file_path):
        print(f"--- Process {current_process_name} started. Main log: {log_file_path} ---")
        if enable_glances_monitoring and glances_proc:
            print(f"--- Glances metrics being collected to: {metrics_output_file} ---")

        result_data = target_func(*args, **kwargs)
        result_queue.put({"status": "success", "data": result_data, "log_file": str(log_file_path), "metrics_file": str(metrics_output_file) if glances_proc else None})
        print(f"--- Process {current_process_name} finished successfully. ---")

    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"Process {current_process_name} Exception: {e}\n{tb_str}"
        try:
            with open(log_file_path, 'a') as f_err:
                f_err.write(f"\n--- PROCESS ERROR ({current_process_name}) ---\n{error_message}\n")
        except: pass
        result_queue.put({"status": "error", "message": str(e), "traceback": tb_str, "log_file": str(log_file_path), "metrics_file": str(metrics_output_file) if glances_proc else None})
    finally:
        if metrics_thread and metrics_thread.is_alive():
            stop_glances_collection_event.set()
            metrics_thread.join(timeout=5) # Wait for thread to finish
        if glances_proc:
            stop_glances_minimal(glances_proc)
        if enable_glances_monitoring and glances_tmp_json_file.exists(): # Cleanup temp file
            try: glances_tmp_json_file.unlink()
            except OSError: pass


def run_operation_in_process(
        target_func: Callable, args_tuple: Tuple = (), kwargs_dict: Optional[Dict[str, Any]] = None,
        env_vars: Optional[Dict[str, str]] = None, log_file_path: Union[Path, str] = None,
        process_name: Optional[str] = None, timeout_seconds: Optional[float] = None,
        enable_glances: bool = False # New flag to control Glances
) -> Dict[str, Any]:
    if kwargs_dict is None: kwargs_dict = {}
    effective_process_name = process_name or target_func.__name__

    if log_file_path is None:
        log_file_path = Path(f"./{effective_process_name}_{os.getpid()}_process.log")
    log_file_path = Path(log_file_path); log_file_path.parent.mkdir(parents=True, exist_ok=True)

    result_queue = multiprocessing.Queue(1)
    process = multiprocessing.Process(
        target=_process_target_wrapper,
        args=(target_func, log_file_path, env_vars, result_queue, effective_process_name, enable_glances, args_tuple, kwargs_dict),
        name=effective_process_name
    )
    process.start()
    result = None
    try:
        result = result_queue.get(timeout=timeout_seconds)
    except multiprocessing.queues.Empty: # type: ignore
        result = {"status": "error", "message": "Process timed out", "log_file": str(log_file_path)}
        if process.is_alive():
            process.terminate(); process.join(5)
            if process.is_alive(): process.kill() # type: ignore
    except Exception as q_e:
        result = {"status": "error", "message": f"Queue error: {q_e}", "log_file": str(log_file_path)}
    process.join()
    return result if result is not None else {"status": "error", "message": "Process ended without result", "log_file": str(log_file_path)}
