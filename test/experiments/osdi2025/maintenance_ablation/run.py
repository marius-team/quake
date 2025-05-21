#!/usr/bin/env python3
"""
Quake Maintenance Policy Ablation Study Runner
───────────────────────────────────────────────
This experiment evaluates different maintenance policy configurations within Quake
under a dynamic workload involving inserts, deletes, and queries.
It generates detailed reports including:
1. Per-index CSV logs of operations.
2. A 9-panel unified plot comparing various metrics across configurations.
3. A stacked bar chart breaking down cumulative time per operation type.
4. A summary table (CSV and Markdown) of key performance indicators.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any # Added Any for type hinting

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# Common utilities
import test.experiments.osdi2025.experiment_utils as common_utils

# Quake specific imports
from quake.index_wrappers.quake import QuakeWrapper # This experiment only uses QuakeWrapper

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OP_STYLE = {
    "query":    dict(ls="-", marker="o", mfc="none", ms=4, lw=1.2),
    "insert":   dict(ls="-", marker="s", mfc="none", ms=4, lw=1.2),
    "delete":   dict(ls="-", marker="^", mfc="none", ms=4, lw=1.2),
    "maintain": dict(ls="None", marker="X", mfc="black", ms=5),
}
LAT_OPS = ["query", "insert", "delete", "maintain"]
IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M = 0, 1, 2, 3
IDX_PART, IDX_RES, IDX_REC, IDX_TOT, IDX_SPL = 4, 5, 6, 7, 8


def unified_plot(cfg: Dict[str, Any], out_dir: Path) -> None:
    styles = cfg["plot"].get("styles", {})
    fig, axs2d = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
    axs = axs2d.flatten()

    idx_handles: List[Line2D] = []
    for j, idx_cfg in enumerate(cfg.get("indexes", [])): # Added .get for safety
        nm, st = idx_cfg["name"], styles.get(idx_cfg["name"], {})
        idx_handles.append(Line2D([0], [0],
                                  color=st.get("color", f"C{j % 10}"),
                                  marker=st.get("marker", "o"),
                                  ls="", markersize=6, label=nm))

    max_ops_overall = 0

    for j, idx_cfg in enumerate(cfg.get("indexes", [])):
        name = idx_cfg["name"]
        st = styles.get(name, {})
        colour = st.get("color", f"C{j % 10}")
        marker = st.get("marker", "o")

        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[unified_plot] Results CSV %s missing – skipped for this index.", csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                log.warning(f"[unified_plot] Empty CSV: {csv_path}")
                continue
            if 'operation_number' in df.columns and not df['operation_number'].empty:
                max_ops_overall = max(max_ops_overall, df.operation_number.max())
        except pd.errors.EmptyDataError:
            log.warning(f"[unified_plot] Could not read or empty CSV: {csv_path}")
            continue

        # Latency plots
        for op, ax_idx_val in zip(LAT_OPS, [IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M]):
            ax = axs[ax_idx_val]
            y_val_series = None
            if op == "maintain":
                if 'maintenance_time_ms' in df.columns:
                    sub = df[df.maintenance_time_ms.fillna(0) > 0]
                    y_val_series = sub.maintenance_time_ms
            elif 'operation_type' in df.columns and 'latency_ms' in df.columns:
                sub = df[df.operation_type == op]
                y_val_series = sub.latency_ms

            if y_val_series is not None and not y_val_series.empty:
                ax.plot(sub.operation_number, y_val_series, color=colour, **OP_STYLE[op])

        # Other metric plots
        plot_specs = [
            (IDX_PART, 'n_list'), (IDX_RES, 'n_resident'),
            (IDX_REC, 'recall', lambda d: (d.operation_type == "query") & d.recall.notna()),
            (IDX_TOT, 'total_cumulative_time',
             lambda d: pd.Series(np.cumsum(d.latency_ms.fillna(0) + d.maintenance_time_ms.fillna(0)), name='total_cumulative_time')),
            (IDX_SPL, ['n_splits', 'n_deletes']) # Special handling for splits/deletes
        ]

        for spec_item in plot_specs:
            ax_idx_val = spec_item[0]
            metric_name_or_list = spec_item[1]
            condition_func = spec_item[2] if len(spec_item) > 2 else lambda d: d[metric_name_or_list].notna()

            ax = axs[ax_idx_val]

            if ax_idx_val == IDX_TOT: # Cumulative time calculation
                if 'latency_ms' in df.columns and 'maintenance_time_ms' in df.columns:
                    total_ms_series = np.cumsum(df.latency_ms.fillna(0) + df.maintenance_time_ms.fillna(0))
                    if not total_ms_series.empty:
                        ax.plot(df.operation_number, total_ms_series, color=colour, marker=marker, lw=1.2)
                continue

            if ax_idx_val == IDX_SPL: # Splits and deletes
                if 'operation_number' in df.columns:
                    spl_series = df.n_splits.fillna(0) if 'n_splits' in df.columns else pd.Series(0, index=df.index)
                    del_series = df.n_deletes.fillna(0) if 'n_deletes' in df.columns else pd.Series(0, index=df.index)
                    if not df.empty:
                        if (spl_series > 0).any(): ax.step(df.operation_number, np.cumsum(spl_series), where="post", color=colour, ls="--", lw=1.2)
                        if (del_series > 0).any(): ax.step(df.operation_number, np.cumsum(del_series), where="post", color=colour, ls=":",  lw=1.2)
                continue

            # General case for other plots
            if isinstance(metric_name_or_list, str) and metric_name_or_list in df.columns:
                sub_df = df[condition_func(df)]
                if not sub_df.empty:
                    ax.plot(sub_df.operation_number, sub_df[metric_name_or_list], color=colour, marker=marker, lw=1.2)

    titles = {
        IDX_LAT_Q: "Latency – Query", IDX_LAT_I: "Latency – Insert",
        IDX_LAT_D: "Latency – Delete", IDX_LAT_M: "Latency – Maintain",
        IDX_PART:  "# Partitions", IDX_RES:   "Resident Vectors",
        IDX_REC:   "Recall", IDX_TOT:   "Running Total Time (ms)",
        IDX_SPL:   "Cumulative Splits / Deletes",
    }
    y_labels_map = {
        IDX_PART: "Count", IDX_RES: "# Vectors", IDX_REC: "Recall",
        IDX_TOT: "Cumulative Time (ms)", IDX_SPL: "Count"
    }
    for i, ax_val in enumerate(axs):
        ax_val.set_title(titles[i], fontsize=11)
        ax_val.grid(True, which="both", ls=":", alpha=0.7)
        ax_val.set_axisbelow(True)
        if i in y_labels_map: ax_val.set_ylabel(y_labels_map[i])
        # Set x-label only for the bottom row of plots
        if i // 3 == 2 : ax_val.set_xlabel("Operation #")

        is_latency_plot = i in [IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M]
        if is_latency_plot: ax_val.set_ylabel("Latency (ms)")


    if max_ops_overall > 0:
        for ax_val in axs:
            ax_val.set_xlim(left=0, right=max_ops_overall)

    fig.legend(idx_handles, [h.get_label() for h in idx_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.035),
               ncol=min(4, len(idx_handles)),
               fontsize=9, title="Index Configuration", frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_plot_path = out_dir / "unified_plot.png"
    plt.savefig(out_plot_path, dpi=150)
    log.info("Unified plot saved to %s", out_plot_path)
    plt.close(fig)


def make_time_breakdown(cfg: Dict[str, Any], out_dir: Path) -> None:
    categories = ["Search", "Insert", "Delete", "Maintain", "Total"]
    totals_data, recall_data = {}, {}
    cmap = get_cmap("tab10")

    for j, idx_cfg in enumerate(cfg.get("indexes", [])):
        name = idx_cfg["name"]
        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[time_breakdown] %s missing – skipped", csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty: raise pd.errors.EmptyDataError
        except pd.errors.EmptyDataError:
            log.warning(f"[time_breakdown] Empty or unreadable CSV: {csv_path}")
            continue

        totals_data[name] = [
            df[df.operation_type == "query"].latency_ms.sum() if 'operation_type' in df.columns else 0,
            df[df.operation_type == "insert"].latency_ms.sum() if 'operation_type' in df.columns else 0,
            df[df.operation_type == "delete"].latency_ms.sum() if 'operation_type' in df.columns else 0,
            df.maintenance_time_ms.fillna(0).sum() if 'maintenance_time_ms' in df.columns else 0,
        ]
        totals_data[name].append(sum(totals_data[name]))

        query_df = df[df.operation_type == "query"] if 'operation_type' in df.columns else pd.DataFrame()
        recall_data[name] = query_df.recall.mean() if not query_df.empty and 'recall' in query_df else np.nan


    if not totals_data:
        log.info("[time_breakdown] No data available for time breakdown plot.")
        return

    n_idx, n_cats = len(totals_data), len(categories)
    x_indices = np.arange(n_cats)
    bar_width = 0.8 / max(1, n_idx)

    fig, ax = plt.subplots(figsize=(max(10, n_idx * 1.5 + 2), 6))
    for j, (name, values) in enumerate(totals_data.items()):
        style = cfg.get("plot", {}).get("styles", {}).get(name, {})
        color = style.get("color", cmap(j % cmap.N))
        x_offset = x_indices + (j - (n_idx - 1) / 2) * bar_width
        ax.bar(x_offset, values, width=bar_width, label=name, color=color, edgecolor="black")
        current_recall = recall_data.get(name)
        if pd.notna(current_recall) and values[0] > 0:
            ax.text(x_offset[0], values[0] * 1.01, f"R={current_recall:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90, color='dimgrey') # Standard color for text

    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Cumulative Time (ms)", fontsize=10)
    ax.set_title("Time Breakdown per Index Configuration", fontsize=12)
    ax.legend(title="Index Configuration", frameon=False, fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    out_plot_path = out_dir / "time_breakdown.png"
    plt.savefig(out_plot_path, dpi=150)
    log.info("Time breakdown plot saved to %s", out_plot_path)
    plt.close(fig)


def produce_summary_table(cfg: Dict[str, Any], out_dir: Path) -> None:
    rows = []
    for idx_cfg in cfg.get("indexes", []):
        name = idx_cfg["name"]
        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[summary_table] %s missing – skipped", csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty: raise pd.errors.EmptyDataError
        except pd.errors.EmptyDataError:
            log.warning(f"[summary_table] Empty or unreadable CSV: {csv_path}")
            continue

        search_time = df[df.operation_type == "query"].latency_ms.sum() if 'operation_type' in df.columns else 0
        insert_time = df[df.operation_type == "insert"].latency_ms.sum() if 'operation_type' in df.columns else 0
        delete_time = df[df.operation_type == "delete"].latency_ms.sum() if 'operation_type' in df.columns else 0
        maintain_time = df.maintenance_time_ms.fillna(0).sum() if 'maintenance_time_ms' in df.columns else 0
        total_time = search_time + insert_time + delete_time + maintain_time

        query_df = df[df.operation_type == "query"] if 'operation_type' in df.columns else pd.DataFrame()
        recall_val = query_df.recall.mean() if not query_df.empty and 'recall' in query_df else np.nan

        n_list_series = df.n_list.dropna() if 'n_list' in df.columns else pd.Series(dtype=float) # ensure series exists
        n_partitions_val = n_list_series.iloc[-1] if not n_list_series.empty else np.nan

        rows.append(dict(
            Index=name, Search=int(search_time), Insert=int(insert_time),
            Delete=int(delete_time), Maintain=int(maintain_time), Total=int(total_time),
            Recall=f"{recall_val:.4f}" if pd.notna(recall_val) else "—",
            Partitions=int(n_partitions_val) if pd.notna(n_partitions_val) else "—"
        ))

    if not rows:
        log.info("[summary_table] No data for summary table.")
        return

    summary_df = pd.DataFrame(rows).sort_values("Index")
    summary_csv_path = out_dir / "summary_table.csv"
    common_utils.save_results_csv(summary_df, summary_csv_path) # Use common util

    summary_md_path = out_dir / "summary_table.md"
    md_content = tabulate(summary_df, headers="keys", tablefmt="github", showindex=False)
    summary_md_path.write_text(md_content + "\n")
    log.info("Summary table (Markdown) saved to %s", summary_md_path)
    log.info("\n%s", md_content)


def run_experiment(cfg_path_str: str, output_dir_str: str) -> None:
    cfg = common_utils.load_config(cfg_path_str)
    main_output_dir = Path(output_dir_str).expanduser()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    current_mode = cfg.get("mode", "run")
    log.info(f"Running Maintenance Ablation experiment in mode: {current_mode}")

    workload_actual_dir = main_output_dir # Workload files are stored at the top level of main_output_dir

    # --- Phase 1: Dataset and Workload Generation ---
    if current_mode in {"build", "run"}:
        workload_actual_dir = common_utils.generate_dynamic_workload(
            dataset_main_cfg=cfg["dataset"],
            workload_generator_cfg=cfg["workload_generator"],
            global_output_dir=main_output_dir, # Pass main_output_dir as the place to store workload files
            overwrite_workload=cfg["overwrite"].get("workload", False)
        )

    # --- Phase 2: Index Evaluation ---
    if current_mode == "run":
        log.info("Starting index evaluation phase...")
        # For this specific experiment, the index class is always QuakeWrapper
        index_class_map = {"Quake": QuakeWrapper}

        for index_conf in cfg.get("indexes", []):
            common_utils.evaluate_index_on_dynamic_workload(
                index_config=index_conf,
                index_class_mapping=index_class_map,
                workload_data_dir=workload_actual_dir, # Use the returned/set workload_actual_dir
                experiment_main_output_dir=main_output_dir,
                overwrite_idx_results=cfg["overwrite"].get("results", False),
                do_maintenance_flag=True # This experiment always does maintenance
            )

    # --- Phase 3: Global Artifact Generation ---
    if current_mode in {"run", "plot"}:
        log.info("Generating global plots and summary tables...")
        any_results_exist = any(
            (main_output_dir / index_conf["name"] / "results.csv").exists()
            for index_conf in cfg.get("indexes", [])
        )

        if any_results_exist:
            unified_plot(cfg, main_output_dir)
            make_time_breakdown(cfg, main_output_dir)
            produce_summary_table(cfg, main_output_dir)
        else:
            log.warning("No results found to generate plots or summary tables. Skipping this step.")

    log.info("Maintenance Ablation experiment finished for mode: %s", current_mode)