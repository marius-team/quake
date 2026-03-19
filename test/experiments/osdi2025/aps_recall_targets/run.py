#!/usr/bin/env python3
"""
Adaptive‐Partition‐Scanning Recall‐Target Runner (Simplified)
Supports: Oracle, FixedNProbe, APS
"""
import logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from test.experiments.osdi2025 import experiment_utils as common_utils
from quake import SearchParams # Keep direct import for clarity if used extensively

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_experiment(cfg_path: str, output_dir_str: str):
    cfg = common_utils.load_config(cfg_path)
    out_dir = Path(output_dir_str)

    dataset_cfg = cfg["dataset"]
    exp_cfg = cfg["experiment"]
    idx_build_cfg = cfg["index"]
    paths_cfg = cfg["paths"]

    vecs, queries, gt = common_utils.load_data(
        dataset_cfg["name"], dataset_cfg.get("path", ""), exp_cfg["nq"]
    )

    idx_path = Path(paths_cfg["index_dir"]) / f"{dataset_cfg['name']}_ivf{idx_build_cfg['nlist']}.index"
    quake_idx = common_utils.prepare_quake_index(
        vecs,
        {"nlist": idx_build_cfg["nlist"], "metric": idx_build_cfg["metric"], "num_workers": exp_cfg["n_workers"]},
        idx_path,
        force_rebuild=cfg["overwrite"].get("index", False)
    )
    logger.info(f"Index ready (workers={exp_cfg['n_workers']})")

    records = []
    k = exp_cfg["k"]
    max_nlist_val = quake_idx.nlist()

    for method in cfg["methods"]:
        for rt in exp_cfg["recall_targets"]:
            per_query_stats = [] # (nprobe, recall, time_ms)

            if method == "Oracle":
                for i, q_vec in enumerate(queries):
                    lo, hi, best_np = 1, max_nlist_val, max_nlist_val
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        sp = common_utils.create_search_params(nprobe=mid, k=k, recall_target=-1)
                        _, rec, _ = common_utils.run_search_trial(quake_idx, q_vec, gt[i], k, sp)
                        if rec >= rt:
                            best_np, hi = mid, mid - 1
                        else:
                            lo = mid + 1
                    sp_final = common_utils.create_search_params(nprobe=best_np, k=k, recall_target=-1)
                    _, final_rec, final_time_ms = common_utils.run_search_trial(quake_idx, q_vec, gt[i], k, sp_final)
                    per_query_stats.append((best_np, final_rec, final_time_ms))

            elif method == "FixedNProbe":
                lo, hi, best_global_np = 1, max_nlist_val, max_nlist_val
                while lo <= hi:
                    mid = (lo + hi) // 2
                    sp = common_utils.create_search_params(nprobe=mid, k=k, recall_target=-1)
                    current_ids = []
                    for q_vec_fixed in queries: # Use all queries for tuning this
                        res_ids = quake_idx.search(q_vec_fixed.unsqueeze(0), sp).ids
                        current_ids.append(res_ids)
                    avg_rec = common_utils.quake_compute_recall(torch.cat(current_ids,0), gt, k).mean().item()
                    if avg_rec >= rt:
                        best_global_np, hi = mid, mid - 1
                    else:
                        lo = mid + 1

                sp_final = common_utils.create_search_params(nprobe=best_global_np, k=k, recall_target=-1)
                for i, q_vec in enumerate(queries):
                    _, final_rec, final_time_ms = common_utils.run_search_trial(quake_idx, q_vec, gt[i], k, sp_final)
                    per_query_stats.append((best_global_np, final_rec, final_time_ms))

            elif method == "APS":
                sp = common_utils.create_search_params(
                    nprobe=-1, k=k, recall_target=rt,
                    recompute_threshold=exp_cfg["recompute_ratio"],
                    use_precomputed=exp_cfg["use_precompute"],
                    initial_search_fraction=exp_cfg["initial_search_fraction"],
                    aps_flush_period_us=exp_cfg["aps_flush_period_us"]
                )
                # for i, q_vec in enumerate(queries):
                #     res = quake_idx.search(q_vec.unsqueeze(0), sp)
                #     nprobe_scan = res.timing_info.partitions_scanned
                #     final_rec = common_utils.quake_compute_recall(res.ids, gt[i].unsqueeze(0), k).item()
                #     final_time_ms = res.timing_info.total_time_ns / 1e6
                #     boundary_time_ms = res.timing_info.boundary_time_ns / 1e6 if hasattr(res.timing_info, 'boundary_time_ns') else 0.0
                #     aps_time_ms = res.timing_info.aps_time_ns / 1e6 if hasattr(res.timing_info, 'aps_time_ns') else 0.0
                #
                #     logger.info(f"APS recall: {final_rec:.4f} (target={rt}), nprobe_scanned={nprobe_scan}, boundary_time={boundary_time_ms:.2f}ms, aps_time={aps_time_ms:.2f}ms, total_time={final_time_ms:.2f}ms")
                #     per_query_stats.append((nprobe_scan, final_rec, final_time_ms))

                res = quake_idx.search(queries, sp)
                nprobe_scan = res.timing_info.partitions_scanned
                final_recall = common_utils.quake_compute_recall(res.ids, gt, k)
                final_time_ms = res.timing_info.total_time_ns / 1e6
                boundary_time_ms = res.timing_info.boundary_distance_time_ns / 1e6
                aps_time_ms = res.timing_info.aps_time_ns / 1e6

                logger.info(f"APS recall: {final_recall.mean():.4f} (target={rt}), nprobe_scanned={nprobe_scan}, "
                            f"boundary_time={boundary_time_ms:.2f}ms, aps_time={aps_time_ms:.2f}ms, total_time={final_time_ms:.2f}ms")
                per_query_stats.append((nprobe_scan, final_recall.mean().item(), final_time_ms))

            else:
                raise ValueError(f"Unknown method: {method}")


            arr = np.array(per_query_stats, dtype=float)
            mean_nprobe = arr[:,0].mean()
            std_nprobe = arr[:,0].std()
            mean_recall = arr[:,1].mean()
            std_recall = arr[:,1].std()
            mean_time_ms = arr[:,2].mean()
            std_time_ms = arr[:,2].std()

            logger.info(f"Method={method} RecallTarget={rt}, Recall={mean_recall:.4f} (std={std_recall:.4f}), "
                        f"Time={mean_time_ms:.2f}ms (std={std_time_ms:.2f}ms), NProbe={mean_nprobe:.1f} (std={std_nprobe:.1f})")

            records.append({
                "Method": method, "RecallTarget": rt,
                "mean_nprobe": mean_nprobe, "std_nprobe": std_nprobe,
                "mean_recall": mean_recall, "std_recall": std_recall,
                "mean_time_ms": mean_time_ms, "std_time_ms": std_time_ms,
            })

    results_df = pd.DataFrame(records)
    common_utils.save_results_csv(results_df, out_dir / "aps_recall_targets_summary.csv")

    plot_path = out_dir / "aps_recall_vs_metrics.png"
    common_utils.plot_recall_performance(
        results_df,
        x_metric="RecallTarget",
        y_metrics=["mean_time_ms", "mean_recall", "mean_nprobe"],
        y_labels=["Mean Query Time (ms)", "Mean Recall Achieved", "Mean NProbe"],
        y_scales=["linear", "linear", "log"],
        plot_title_prefix=f"{dataset_cfg['name']} APS Experiment",
        output_path=plot_path,
        recall_targets_for_line=exp_cfg["recall_targets"]
    )