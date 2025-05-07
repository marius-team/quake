#!/usr/bin/env python3
"""
Recall-Target Runner for Approximate Nearest Neighbor Search Methods.
Supports: Oracle, FixedNProbe, APS, and LAET (Learned Adaptive Early Termination).
"""

import time
import yaml
import logging
from pathlib import Path
import argparse

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'quake' and 'laet' are in the PYTHONPATH or same directory
try:
    from quake import QuakeIndex, IndexBuildParams, SearchParams
    from quake.datasets.ann_datasets import load_dataset
    from quake.utils import compute_recall
except ImportError as e:
    print(f"Error importing Quake library components: {e}. Ensure 'quake' is installed and in PYTHONPATH.")
    raise

try:
    # Adjust this import based on your project structure.
    # If laet.py is in the same directory as this script, it should be:
    # from laet import LAETPipeline
    # Your current script uses a relative import:
    from .laet import LAETPipeline
except ImportError as e:
    # Fallback if relative import fails (e.g., script run as top-level)
    try:
        from laet import LAETPipeline
        print("Note: Imported LAETPipeline using 'from laet import LAETPipeline'.")
    except ImportError:
        print(f"Error importing LAETPipeline: {e}. Ensure 'laet.py' is correctly placed and import path is valid.")
        raise


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(cfg_path: str, output_dir_str: str):
    """
    Runs the ANN experimentation pipeline.

    Args:
        cfg_path (str): Path to the YAML configuration file.
        output_dir_str (str): Path to the directory where results will be saved.
    """
    # 1) Load config + setup output dir
    logger.info(f"Loading configuration from: {cfg_path}")
    try:
        cfg = yaml.safe_load(Path(cfg_path).read_text())
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {cfg_path}")
        return
    except Exception as e:
        logger.error(f"Error loading YAML configuration from {cfg_path}: {e}")
        return

    out_path = Path(output_dir_str)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {out_path}")

    # 2) Load data
    data_dir = cfg["dataset"].get("path", "")
    dataset_name = cfg["dataset"]["name"]
    logger.info(f"Loading dataset: {dataset_name} from path: '{data_dir if data_dir else 'default'}'")
    try:
        vecs, queries_torch, gt_torch = load_dataset(dataset_name, data_dir)
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        return

    nq_config = cfg["experiment"].get("nq", queries_torch.shape[0])
    queries_torch, gt_torch = queries_torch[:nq_config], gt_torch[:nq_config]
    logger.info(f"Using {queries_torch.shape[0]} queries for the experiment.")

    # 3) Build or load QuakeIndex (used by all methods)
    idx_quake = QuakeIndex()

    build_params = IndexBuildParams()
    build_params.nlist = cfg["index"]["nlist"]
    build_params.metric = cfg["index"]["metric"]
    build_params.num_workers = cfg["experiment"].get("n_workers", 0)

    idx_base_dir = Path(cfg["paths"].get("index_dir", out_path / "indexes"))
    idx_base_dir.mkdir(parents=True, exist_ok=True)
    idx_path_quake = idx_base_dir / f"{dataset_name}_quake_ivf{build_params.nlist}.index"

    if not idx_path_quake.exists() or cfg["overwrite"].get("index", False):
        logger.info(f"Building QuakeIndex at {idx_path_quake}...")
        try:
            idx_quake.build(vecs, torch.arange(len(vecs)), build_params)
            idx_quake.save(str(idx_path_quake))
            logger.info(f"QuakeIndex built and saved to {idx_path_quake}.")
        except Exception as e:
            logger.error(f"Failed to build or save QuakeIndex: {e}", exc_info=True)
            return
    else:
        logger.info(f"Loading existing QuakeIndex from {idx_path_quake}...")
        try:
            idx_quake.load(str(idx_path_quake), 0)
            logger.info(f"QuakeIndex loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load QuakeIndex from {idx_path_quake}: {e}", exc_info=True)
            return

    current_index_nlist = idx_quake.nlist() if callable(getattr(idx_quake, 'nlist', None)) else getattr(idx_quake, 'nlist', build_params.nlist)
    if current_index_nlist != build_params.nlist:
        logger.warning(f"QuakeIndex nlist ({current_index_nlist}) differs from config ({build_params.nlist}). Using index's nlist: {current_index_nlist}.")
    max_nlist_val = current_index_nlist


    # 4) Initialize LAETPipeline if "LAET" is in methods
    laet_pipeline_instance = None
    if "LAET" in cfg["methods"]:
        if not idx_quake:
            logger.error("LAET method requires QuakeIndex, but it's not available. Skipping LAET.")
        else:
            laet_main_cfg = cfg.get("laet_config", {})
            laet_paths_cfg = cfg.get("paths", {}) # For laet_artifacts_base_dir

            # Prepare config for LAETPipeline constructor
            pipeline_init_config = {
                "base_dir": laet_paths_cfg.get("laet_artifacts_base_dir", "laet_pipeline_output"),
                "training_data_subdir": laet_main_cfg.get("training_data_subdir", "laet_training_data"),
                "model_subdir": laet_main_cfg.get("model_subdir", "laet_gbdt_models"),
                "dataset_name_for_files": laet_main_cfg.get("dataset_name_for_files", dataset_name.upper()),
                "index_key_for_files": laet_main_cfg.get("index_key_for_files", f"IVF{max_nlist_val}_Flat"),
                "model_was_trained_on_log_target": laet_main_cfg.get("train_log_target", False) # Used for model naming if trained by pipeline
            }

            laet_pipeline_instance = LAETPipeline(vector_dim=queries_torch.shape[1],
                                                  laet_artifacts_config=pipeline_init_config)

            # Prepare configs for get_or_train_model
            # These paths should point to your primary dataset files for LAET GBDT training
            data_paths_for_laet_training = {
                "base_vectors": laet_main_cfg.get("base_vectors_path_for_laet_train", f"data/{dataset_name}_base.fvecs"),
                "train_query_vectors": laet_main_cfg.get("train_query_vectors_path_for_laet_train", f"data/{dataset_name}_learn.fvecs"),
                "gt_train": laet_main_cfg.get("gt_train_path_for_laet_train", f"data/{dataset_name}_learn_groundtruth.ivecs"),
                "gt_format": laet_main_cfg.get("gt_format_for_laet_train", "ivecs"),
                "training_vectors_for_faiss_index": laet_main_cfg.get("faiss_training_vectors_path_for_laet_train", f"data/{dataset_name}_learn.fvecs")
            }
            generation_params_for_laet_training = {
                "num_train_queries_to_process": laet_main_cfg.get("num_laet_training_samples", 10000),
                "k_gt_load_for_target": laet_main_cfg.get("laet_train_k_gt_load", 1),
                "k_search_for_target": laet_main_cfg.get("laet_train_k_search", 10),
                "max_nprobe_scan": laet_main_cfg.get("laet_train_max_nprobe", 128),
                "max_efsearch_scan": laet_main_cfg.get("laet_train_max_efsearch", 256),
                "param_scan_step": laet_main_cfg.get("laet_train_param_scan_step", 1)
            }
            faiss_params_for_laet_training = {
                "faiss_index_key_for_gen": laet_main_cfg.get("faiss_index_key_for_laet_train_gen", f"IVF{max_nlist_val},Flat")
            }
            gbdt_training_params_for_laet = {
                "log_target_training": laet_main_cfg.get("train_log_target", False), # This sets log_target_prediction in pipeline
                "test_size": laet_main_cfg.get("train_gbdt_test_size", 0.1),
                "num_leaves": laet_main_cfg.get("train_gbdt_num_leaves", 31),
                "learning_rate": laet_main_cfg.get("train_gbdt_lr", 0.05),
                "n_estimators": laet_main_cfg.get("train_gbdt_n_estimators", 100),
                "feature_fraction": laet_main_cfg.get("train_gbdt_feature_fraction", 0.9),
                "bagging_fraction": laet_main_cfg.get("train_gbdt_bagging_fraction", 0.8),
                "bagging_freq": laet_main_cfg.get("train_gbdt_bagging_freq", 5)
            }

            laet_pipeline_instance.get_or_train_model(
                overwrite_model=cfg.get("overwrite", {}).get("laet_model", False),
                data_paths_config=data_paths_for_laet_training,
                generation_params_config=generation_params_for_laet_training,
                faiss_params_config=faiss_params_for_laet_training,
                training_params_config=gbdt_training_params_for_laet
            )
            logger.info("LAET GBDT model is ready (loaded or trained).")

                # 5) Run each method × recall_target
    records = []
    k_metric = cfg["experiment"]["k"]

    for method in cfg["methods"]:
        for rt_metric in cfg["experiment"]["recall_targets"]:
            logger.info(f"Running Method: {method}, Target Recall: {rt_metric:.3f}")

            mean_param_val, mean_recall_val, mean_time_ms_val = np.nan, np.nan, np.nan
            std_param_val, std_recall_val, std_time_ms_val = np.nan, np.nan, np.nan
            per_query_data_list = []

            if method == "LAET":
                if laet_pipeline_instance and idx_quake and laet_pipeline_instance.gbdt_model:
                    quake_sp_for_laet = SearchParams()
                    # k and nprobe will be set inside run_inference_for_quake_experiment

                    bs_multiplier_range = tuple(laet_main_cfg.get("inference_multiplier_range", [0.1, 50.0]))
                    bs_steps = laet_main_cfg.get("inference_num_bs_steps", 20)

                    try:
                        mean_param_val, mean_recall_val, mean_time_ms_val = \
                            laet_pipeline_instance.run_inference_for_quake_experiment(
                                quake_idx=idx_quake, # Corrected from quake_index to idx_quake
                                queries_torch=queries_torch,
                                gt_torch=gt_torch,
                                target_recall=rt_metric, # Corrected from target_recall_metric
                                k_search=k_metric,    # Corrected from k_search_quake
                                base_quake_sp=quake_sp_for_laet, # Corrected from base_quake_search_params
                                num_bs_steps=bs_steps,
                                multiplier_range=bs_multiplier_range
                            )
                        std_param_val, std_recall_val, std_time_ms_val = 0.0, 0.0, 0.0
                        logger.info(f"LAET @ RT={rt_metric:.3f} -> AvgParam={mean_param_val:.2f}, AvgRecall={mean_recall_val:.4f}, AvgTime={mean_time_ms_val:.3f}ms")
                    except Exception as e:
                        logger.error(f"Error during LAET inference run: {e}", exc_info=True)
                        mean_param_val, mean_recall_val, mean_time_ms_val = np.nan, np.nan, np.nan
                else:
                    logger.warning(f"LAET method specified but pipeline/model or QuakeIndex not ready. Skipping for RT={rt_metric}.")
                    # Metrics remain NaN by default

            elif method in ["Oracle", "FixedNProbe", "APS"]:
                if not idx_quake:
                    logger.warning(f"QuakeIndex not available for method {method}. Skipping.")
                    continue

                if method == "Oracle":
                    for i, q_single_torch in enumerate(queries_torch):
                        lo, hi, best_param_oracle = 1, max_nlist_val, max_nlist_val
                        while lo <= hi:
                            mid_param = (lo + hi) // 2
                            sp_oracle_bs = SearchParams()
                            sp_oracle_bs.nprobe = mid_param
                            sp_oracle_bs.k = k_metric
                            res_oracle_bs = idx_quake.search(q_single_torch.unsqueeze(0), sp_oracle_bs)
                            rec_oracle_bs = compute_recall(res_oracle_bs.ids, gt_torch[i].unsqueeze(0), k_metric).item()
                            if rec_oracle_bs >= rt_metric:
                                best_param_oracle = mid_param
                                hi = mid_param - 1
                            else:
                                lo = mid_param + 1

                        t_start_ns = time.perf_counter_ns()
                        sp_oracle_final = SearchParams()
                        sp_oracle_final.nprobe = best_param_oracle
                        sp_oracle_final.k = k_metric
                        res_oracle_final = idx_quake.search(q_single_torch.unsqueeze(0), sp_oracle_final)
                        elapsed_ns = time.perf_counter_ns() - t_start_ns
                        rec_oracle_final = compute_recall(res_oracle_final.ids, gt_torch[i].unsqueeze(0), k_metric).item()
                        per_query_data_list.append((best_param_oracle, rec_oracle_final, elapsed_ns / 1_000_000.0))

                elif method == "FixedNProbe":
                    lo_fixed, hi_fixed, best_overall_param_fixed = 1, max_nlist_val, max_nlist_val
                    while lo_fixed <= hi_fixed:
                        mid_param_fixed = (lo_fixed + hi_fixed) // 2
                        sp_fixed_bs = SearchParams()
                        sp_fixed_bs.nprobe = mid_param_fixed
                        sp_fixed_bs.k = k_metric
                        sp_fixed_bs.recall_target = -1

                        ids_list_fixed = []
                        for q_bs_fixed in queries_torch:
                            res_bs_fixed = idx_quake.search(q_bs_fixed.unsqueeze(0), sp_fixed_bs)
                            if res_bs_fixed.ids is not None and res_bs_fixed.ids.numel() > 0:
                                ids_list_fixed.append(res_bs_fixed.ids)

                        if not ids_list_fixed: avg_rec_fixed = 0.0
                        else: avg_rec_fixed = compute_recall(torch.cat(ids_list_fixed, 0), gt_torch, k_metric).mean().item()

                        if avg_rec_fixed >= rt_metric:
                            best_overall_param_fixed = mid_param_fixed
                            hi_fixed = mid_param_fixed - 1
                        else: lo_fixed = mid_param_fixed + 1

                    for i, q_single_torch in enumerate(queries_torch):
                        t_start_ns = time.perf_counter_ns()
                        sp_fixed_final = SearchParams()
                        sp_fixed_final.nprobe = best_overall_param_fixed
                        sp_fixed_final.k = k_metric
                        sp_fixed_final.recall_target = -1
                        res_fixed_final = idx_quake.search(q_single_torch.unsqueeze(0), sp_fixed_final)
                        elapsed_ns = time.perf_counter_ns() - t_start_ns
                        rec_fixed_final = compute_recall(res_fixed_final.ids, gt_torch[i].unsqueeze(0), k_metric).item()
                        per_query_data_list.append((best_overall_param_fixed, rec_fixed_final, elapsed_ns / 1_000_000.0))

                elif method == "APS":
                    for i, q_single_torch in enumerate(queries_torch):
                        sp_aps = SearchParams()
                        sp_aps.nprobe = -1
                        sp_aps.k = k_metric
                        sp_aps.recall_target = rt_metric
                        sp_aps.recompute_threshold = cfg["experiment"].get("recompute_ratio", 0.00001)
                        sp_aps.use_precomputed = cfg["experiment"].get("use_precompute", True)
                        sp_aps.initial_search_fraction = cfg["experiment"].get("initial_search_fraction", 0.1)

                        t_start_ns = time.perf_counter_ns()
                        res_aps = idx_quake.search(q_single_torch.unsqueeze(0), sp_aps)
                        elapsed_ns = time.perf_counter_ns() - t_start_ns
                        rec_aps = compute_recall(res_aps.ids, gt_torch[i].unsqueeze(0), k_metric).item()
                        param_aps = res_aps.timing_info.partitions_scanned if hasattr(res_aps, 'timing_info') and hasattr(res_aps.timing_info, 'partitions_scanned') else np.nan
                        per_query_data_list.append((param_aps, rec_aps, elapsed_ns / 1_000_000.0))

                if per_query_data_list:
                    arr = np.array(per_query_data_list, dtype=float)
                    if arr.size > 0:
                        mean_param_val, mean_recall_val, mean_time_ms_val = np.nanmean(arr, axis=0)
                        std_param_val, std_recall_val, std_time_ms_val = np.nanstd(arr, axis=0)
                    logger.info(f"{method} @ RT={rt_metric:.3f} -> AvgParam={mean_param_val:.2f}, AvgRecall={mean_recall_val:.4f}, AvgTime={mean_time_ms_val:.3f}ms")
            else:
                if method != "LAET":
                    logger.warning(f"Unknown method: {method}. Skipping.")
                    continue

            records.append({
                "Method": method, "RecallTarget": rt_metric,
                "mean_nprobe": mean_param_val, "std_nprobe": std_param_val,
                "mean_recall": mean_recall_val, "std_recall": std_recall_val,
                "mean_time_ms": mean_time_ms_val, "std_time_ms": std_time_ms_val,
            })

    # 6) Save CSV
    df_results = pd.DataFrame(records)
    csv_filename = cfg["paths"].get("results_file_prefix", "experiment_summary") + f"_{dataset_name}.csv"
    out_csv_path = out_path / csv_filename
    df_results.to_csv(out_csv_path, index=False, float_format='%.4f')
    logger.info(f"Results summary saved to: {out_csv_path}")

    # 7) Plotting
    if not df_results.empty:
        plot_filename = cfg["paths"].get("plot_file_prefix", "recall_comparison_plots") + f"_{dataset_name}.png"
        plot_output_dir = Path(cfg["paths"].get("plot_dir", output_dir_str))
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        plot_path_full = plot_output_dir / plot_filename

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        unique_methods_in_results = df_results["Method"].unique()

        for m_name in unique_methods_in_results:
            grp = df_results[df_results["Method"] == m_name].sort_values(by="RecallTarget")
            if grp.empty: continue

            axes[0].errorbar(grp["RecallTarget"], grp["mean_time_ms"], yerr=grp["std_time_ms"],
                             marker="o", capsize=3, label=m_name, alpha=0.8, errorevery=1, elinewidth=1)
            axes[1].errorbar(grp["RecallTarget"], grp["mean_recall"], yerr=grp["std_recall"],
                             marker="o", capsize=3, label=m_name, alpha=0.8, errorevery=1, elinewidth=1)
            axes[2].errorbar(grp["RecallTarget"], grp["mean_nprobe"], yerr=grp["std_nprobe"],
                             marker="o", capsize=3, label=m_name, alpha=0.8, errorevery=1, elinewidth=1)

        axes[0].set_ylabel("Mean Query Time (ms)")
        axes[0].set_title(f"Recall Target vs. Query Time ({dataset_name})")
        axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.7)

        recall_target_line_plot = np.array(cfg["experiment"]["recall_targets"])
        axes[1].plot(recall_target_line_plot, recall_target_line_plot, "k--", label="Target Recall", alpha=0.7)
        axes[1].set_ylabel("Achieved Mean Recall")
        axes[1].set_title(f"Recall Target vs. Achieved Recall ({dataset_name})")
        axes[1].legend(); axes[1].grid(True, ls=":", alpha=0.7)
        min_r, max_r = (df_results["mean_recall"].min() if pd.notna(df_results["mean_recall"].min()) else 0.0,
                        df_results["mean_recall"].max() if pd.notna(df_results["mean_recall"].max()) else 1.0)
        axes[1].set_ylim(bottom=max(0.0, min_r - 0.05), top=min(1.01, max_r + 0.05))

        axes[2].set_xlabel("Recall Target")
        axes[2].set_ylabel("Mean Search Parameter (nprobe/equiv.)")
        axes[2].set_yscale("log"); axes[2].set_title(f"Recall Target vs. Search Parameter ({dataset_name})")
        axes[2].legend(); axes[2].grid(True, which="both", ls=":", alpha=0.7)

        plt.tight_layout(); plt.savefig(plot_path_full); plt.close(fig)
        logger.info(f"Plots saved to: {plot_path_full}")
    else:
        logger.info("No data in results DataFrame, skipping plotting.")
