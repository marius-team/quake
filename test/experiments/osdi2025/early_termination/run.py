#!/usr/bin/env python3
"""
APS vs. early-termination baselines (Simplified)
Generates CSV + Markdown table with:
    RecallAchieved | nprobe | QueryLatencyMs | TuningTimeMs
for recall targets 0.80, 0.90, 0.99
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import test.experiments.osdi2025.experiment_utils as common_utils

from quake import QuakeIndex, SearchParams, IndexBuildParams
from quake.utils import compute_recall

# LAET (mandatory for this experiment)
try:
    from .laet import LAETPipeline
except ImportError as e:
    raise ImportError(
        "LAET is required for this experiment but could not be imported. "
        "Install the laet package or fix PYTHONPATH."
    ) from e

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)7s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("experiment")

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
    best_a = a_max
    lo, hi = a_min, a_max
    best_stats: list[tuple[float, float, float]] = []

    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        current_per_query_stats: list[tuple[float, float, float]] = []

        for i_q, q_vec in enumerate(queries):
            sp = common_utils.create_search_params(
                nprobe=-1, k=k, recall_target=target_recall,
                use_auncel=True, auncel_a=mid, auncel_b=.5,
                recompute_threshold=recompute_thr, initial_search_fraction=init_frac,
            )
            partitions_scanned, rec, time_ms = common_utils.run_search_trial(
                idx, q_vec, gt[i_q], k, sp
            )
            current_per_query_stats.append((partitions_scanned, rec, time_ms))

        mean_recall = np.mean([r for _, r, _ in current_per_query_stats])
        if mean_recall >= target_recall:
            best_a, best_stats = mid, current_per_query_stats
            hi = mid * 0.5
        else:
            lo = mid * 1.5
        if abs(hi - lo) < 1e-4 or hi < lo : # Added small tolerance and check
            break
    return best_a, best_stats


def binary_search_best_nprobe(
        idx: QuakeIndex,
        queries_tensor: torch.Tensor, # Can be single or multiple queries
        gt_tensor: torch.Tensor,    # Corresponding ground truth
        k_val: int,
        target_recall_val: float,
        max_nprobe_val: int,
        use_spann_flag: bool = False,
        spann_eps_val: float = 1.5,
) -> int:
    lo, hi, best = 1, max_nprobe_val, max_nprobe_val
    is_single_query = queries_tensor.shape[0] == 1

    while lo <= hi:
        mid = (lo + hi) // 2
        sp = common_utils.create_search_params(
            k=k_val, nprobe=mid, use_spann=use_spann_flag,
            spann_eps=spann_eps_val if use_spann_flag else 0.0
        )

        if is_single_query:
            # For single query, run_search_trial is fine for recall check
            _, rec, _ = common_utils.run_search_trial(idx, queries_tensor[0], gt_tensor[0], k_val, sp)
            mean_rec_val = rec
        else:
            # For multiple queries, need to aggregate recall
            ids_all_list = []
            for i_q in range(queries_tensor.shape[0]):
                # Directly call idx.search for id extraction for compute_recall over batch
                res_search = idx.search(queries_tensor[i_q].unsqueeze(0), sp)
                if res_search.ids.numel():
                    ids_all_list.append(res_search.ids)
            if not ids_all_list:
                mean_rec_val = 0.0
            else:
                mean_rec_val = compute_recall(torch.cat(ids_all_list, 0), gt_tensor, k_val).mean().item()

        if mean_rec_val >= target_recall_val:
            best, hi = mid, mid - 1
        else:
            lo = mid + 1
    return best

def print_markdown(df: pd.DataFrame) -> None:
    print("\n" + "-" * 80)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("-" * 80 + "\n")


def run_experiment(cfg_path_str: str, output_dir_str: str) -> None:
    cfg = common_utils.load_config(cfg_path_str)
    out_dir = Path(output_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = cfg["experiment"]
    dataset_cfg = cfg["dataset"]
    idx_build_cfg = cfg["index"]
    paths_cfg = cfg["paths"]

    k_val = exp_cfg["k"]
    recall_targets_list: List[float] = exp_cfg["recall_targets"]
    table_targets_set = {0.80, 0.90, 0.99}

    vecs, queries_all, gt_all = common_utils.load_data(
        dataset_cfg["name"], dataset_cfg.get("path", ""), exp_cfg.get("nq")
    )
    logger.info(f"Dataset loaded – {len(vecs):,} base vecs, {len(queries_all)} queries.")

    idx_file_path = Path(paths_cfg["index_dir"]) / f"{dataset_cfg['name']}_quake_ivf{idx_build_cfg['nlist']}.index"
    quake_idx_instance = common_utils.prepare_quake_index(
        vecs,
        {"nlist": idx_build_cfg["nlist"], "metric": idx_build_cfg["metric"], "num_workers": exp_cfg.get("n_workers", 0)},
        idx_file_path,
        force_rebuild=cfg["overwrite"].get("index", False),
        num_workers_load=exp_cfg.get("n_workers",0) # Pass num_workers for loading context
    )
    max_nprobe_val = quake_idx_instance.nlist()

    # LAET SPECIFIC CONFIG
    laet_cfg_dict = cfg["laet_config"]
    data_paths_cfg_laet = {
        "base_vectors": laet_cfg_dict["base_vectors_path_for_laet_train"],
        "train_query_vectors": laet_cfg_dict["train_query_vectors_path_for_laet_train"],
        "gt_train": laet_cfg_dict["gt_train_path_for_laet_train"],
        "gt_format": laet_cfg_dict.get("gt_format_for_laet_train", "ivecs"),
        "training_vectors_for_faiss_index": laet_cfg_dict["faiss_training_vectors_path_for_laet_train"],
    }
    generation_params_cfg_laet = {
        "num_train_queries_to_process": laet_cfg_dict.get("num_laet_training_samples", 10000),
        "k_gt_load_for_target": laet_cfg_dict.get("laet_train_k_gt_load", 1),
        "k_search_for_target": laet_cfg_dict.get("laet_train_k_search", 10),
        "max_nprobe_scan": laet_cfg_dict.get("laet_train_max_nprobe", 128),
        "param_scan_step": laet_cfg_dict.get("laet_train_param_scan_step", 1),
    }
    faiss_params_cfg_laet = {
        "faiss_index_key_for_gen": laet_cfg_dict.get("faiss_index_key_for_laet_train_gen", f"IVF{max_nprobe_val},Flat")
    }
    training_params_cfg_laet = {
        "log_target_training": laet_cfg_dict.get("train_log_target", False),
        "test_size": laet_cfg_dict.get("train_gbdt_test_size", 0.1),
        # ... other GBDT params from the LAET repo
        "num_leaves": laet_cfg_dict.get("train_gbdt_num_leaves", 31),
        "learning_rate": laet_cfg_dict.get("train_gbdt_lr", 0.05),
        "n_estimators": laet_cfg_dict.get("train_gbdt_n_estimators", 100),
    }

    laet_pipeline_instance = LAETPipeline(
        vector_dim=queries_all.shape[1],
        laet_artifacts_config={
            "base_dir": paths_cfg["laet_artifacts_base_dir"],
            "dataset_name_for_files": laet_cfg_dict.get("dataset_name_for_files", dataset_cfg["name"].upper()),
            "index_key_for_files": laet_cfg_dict.get("index_key_for_files", f"IVF{max_nprobe_val}_Flat"),
            "model_subdir": laet_cfg_dict.get("model_subdir", "laet_gbdt_models"),
        },
    )
    t_laet_train_start_ns = time.perf_counter_ns()
    laet_pipeline_instance.get_or_train_model(
        overwrite_model=False, # Or cfg["overwrite"].get("laet_model", False)
        data_paths_config=data_paths_cfg_laet,
        generation_params_config=generation_params_cfg_laet,
        faiss_params_config=faiss_params_cfg_laet,
        training_params_config=training_params_cfg_laet,
    )
    laet_training_total_ns = time.perf_counter_ns() - t_laet_train_start_ns
    logger.info(f"LAET model ready (train/load time: {laet_training_total_ns/1e6:.1f} ms).")

    experiment_records: List[Dict] = []

    for method_name in cfg["methods"]:
        for recall_target_val in recall_targets_list:
            query_results_list = []  # list of (nprobe_val, recall_val, query_time_ms)
            current_tuning_ns = 0

            if method_name == "Oracle":
                for i_q, q_vec in enumerate(queries_all):
                    gt_vec = gt_all[i_q]
                    t_tune_start_ns = time.perf_counter_ns()
                    # Pass single query and gt for Oracle's per-query tuning
                    best_np_val = binary_search_best_nprobe(
                        quake_idx_instance, q_vec.unsqueeze(0), gt_vec.unsqueeze(0), k_val, recall_target_val, max_nprobe_val
                    )
                    current_tuning_ns += (time.perf_counter_ns() - t_tune_start_ns)

                    sp_final = common_utils.create_search_params(k=k_val, nprobe=best_np_val)
                    _, rec_val, time_ms_val = common_utils.run_search_trial(
                        quake_idx_instance, q_vec, gt_vec, k_val, sp_final
                    )
                    query_results_list.append((best_np_val, rec_val, time_ms_val))

            elif method_name in ("FixedNProbe", "SPANN"):
                t_tune_start_ns = time.perf_counter_ns()
                best_np_val = binary_search_best_nprobe(
                    quake_idx_instance, queries_all, gt_all, k_val, recall_target_val, max_nprobe_val,
                    use_spann_flag=(method_name == "SPANN"), spann_eps_val=1.5
                )
                current_tuning_ns = time.perf_counter_ns() - t_tune_start_ns

                for i_q, q_vec in enumerate(queries_all):
                    sp_final = common_utils.create_search_params(
                        k=k_val, nprobe=best_np_val,
                        use_spann=(method_name == "SPANN"),
                        spann_eps=1.5 if method_name == "SPANN" else 0.0
                    )
                    _, rec_val, time_ms_val = common_utils.run_search_trial(
                        quake_idx_instance, q_vec, gt_all[i_q], k_val, sp_final
                    )
                    query_results_list.append((best_np_val, rec_val, time_ms_val))

            elif method_name == "APS":
                for i_q, q_vec in enumerate(queries_all):
                    sp_aps = common_utils.create_search_params(
                        nprobe=-1, k=k_val, recall_target=recall_target_val,
                        recompute_threshold=exp_cfg["recompute_ratio"],
                        use_precomputed=exp_cfg["use_precompute"],
                        initial_search_fraction=exp_cfg["initial_search_fraction"]
                    )
                    nprobe_scanned, rec_val, time_ms_val = common_utils.run_search_trial(
                        quake_idx_instance, q_vec, gt_all[i_q], k_val, sp_aps
                    )
                    query_results_list.append((nprobe_scanned, rec_val, time_ms_val))

            elif method_name == "Auncel":
                t_tune_start_ns = time.perf_counter_ns()
                best_a_val, query_results_list = tune_auncel_a( # tune_auncel_a returns the list directly
                    quake_idx_instance, queries_all, gt_all, k_val, recall_target_val,
                    recompute_thr=exp_cfg["recompute_ratio"],
                    init_frac=exp_cfg["initial_search_fraction"]
                )
                current_tuning_ns = time.perf_counter_ns() - t_tune_start_ns
                logger.info(f"Auncel @ RT={recall_target_val:.2f} → selected a = {best_a_val:.4f}")

            elif method_name == "LAET":
                t_infer_start_ns = time.perf_counter_ns()
                mean_np, mean_rec, mean_ms = laet_pipeline_instance.run_inference_for_quake_experiment(
                    quake_idx=quake_idx_instance,
                    queries_torch=queries_all,
                    gt_torch=gt_all,
                    target_recall=recall_target_val,
                    k_search=k_val,
                    base_quake_sp=common_utils.create_search_params(k=k_val), # Pass k here
                )
                # For LAET, tuning_ns includes GBDT training + multiplier search
                current_tuning_ns = laet_training_total_ns + (time.perf_counter_ns() - t_infer_start_ns)
                query_results_list.append((mean_np, mean_rec, mean_ms)) # Store averages

            else:
                logger.warning(f"Unknown method '{method_name}' – skipping.")
                continue

            if not query_results_list: continue

            # logger.info(f"Method={method} RecallTarget={rt}, Recall={mean_recall:.4f} (std={std_recall:.4f}), "
            #             f"Time={mean_time_ms:.2f}ms (std={std_time_ms:.2f}ms), NProbe={mean_nprobe:.1f} (std={std_nprobe:.1f})")

            logger.info(f"Method={method_name} RecallTarget={recall_target_val:.2f}, "
                        f"Recall={np.mean([r for _, r, _ in query_results_list]):.4f}, "
                        f"Query Latency={np.mean([t for _, _, t in query_results_list]):.2f}ms, "
                        f"NProbe={np.mean([n for n, _, _ in query_results_list]):.1f}, "
                        f"Tuning Time={current_tuning_ns / 1e6:.1f}ms")

            arr = np.asarray(query_results_list, dtype=float)
            if arr.shape[0] == 1 and method_name == "LAET": # LAET already returns mean values
                mean_nprobe_val, mean_recall_val, mean_q_ms_val = arr[0,0], arr[0,1], arr[0,2]
                std_nprobe_val, std_recall_val, std_q_ms_val = np.nan, np.nan, np.nan # Stdev not applicable for pre-averaged
            else:
                mean_nprobe_val = np.nanmean(arr[:,0])
                std_nprobe_val = np.nanstd(arr[:,0])
                mean_recall_val = np.nanmean(arr[:,1])
                std_recall_val = np.nanstd(arr[:,1])
                mean_q_ms_val = np.nanmean(arr[:,2])
                std_q_ms_val = np.nanstd(arr[:,2])


            experiment_records.append(dict(
                Method=method_name, RecallTarget=recall_target_val,
                RecallAchieved=mean_recall_val, nprobe=mean_nprobe_val,
                QueryLatencyMs=mean_q_ms_val, TuningTimeMs=current_tuning_ns / 1e6,
                StdRecall=std_recall_val, StdNprobe=std_nprobe_val, StdQueryLatencyMs=std_q_ms_val
            ))

    df_results = pd.DataFrame(experiment_records)
    df_output_final = df_results[df_results["RecallTarget"].isin(table_targets_set)].copy().sort_values(
        ["Method", "RecallTarget"]
    )

    csv_output_path = out_dir / f"aps_et_results_{dataset_cfg['name']}.csv"
    common_utils.save_results_csv(df_output_final, csv_output_path) # Use common save
    print_markdown(df_output_final)

    if cfg.get("plot", {}).get("enable", False):
        plot_path = out_dir / f"et_latency_recall_plot_{dataset_cfg['name']}.png"
        common_utils.plot_recall_performance(
            df_results, # Plot all results, not just table_targets
            x_metric="RecallTarget",
            y_metrics=["QueryLatencyMs", "RecallAchieved", "nprobe"],
            y_labels=["Mean Query Latency (ms)", "Mean Recall Achieved", "Mean NProbe"],
            y_scales=["log", "linear", "log"], # nprobe often benefits from log scale
            plot_title_prefix=f"{dataset_cfg['name']} Early Termination",
            output_path=plot_path,
            recall_targets_for_line=recall_targets_list
        )