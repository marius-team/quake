#!/usr/bin/env python3
"""
Quake maintenance–ablation runner
─────────────────────────────────
Extends the original “kick-the-tires” driver with:

1.  *summary_table.csv*      – cumulative times + final partition count
2.  *summary_table.md*       – Markdown, via **tabulate**
3.  *time_breakdown.png*     – stacked bars (unchanged)
4.  *unified_plot.png*       – nine-panel diagnostic (unchanged)

Per-index artefacts (under <output>/<index>/):
    results.csv              – per-operation log
    time_breakdown.png       – per-index stacked bars
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# ── quake wrappers ------------------------------------------------------------
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake import QuakeWrapper
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── constants -----------------------------------------------------------------
OP_STYLE = {
    "query":    dict(ls="-", marker="o", mfc="none", ms=4, lw=1.2),
    "insert":   dict(ls="-", marker="s", mfc="none", ms=4, lw=1.2),
    "delete":   dict(ls="-", marker="^", mfc="none", ms=4, lw=1.2),
    "maintain": dict(ls="None", marker="X", mfc="black", ms=5),
}

LAT_OPS = ["query", "insert", "delete", "maintain"]
IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M = 0, 1, 2, 3
IDX_PART, IDX_RES, IDX_REC, IDX_TOT, IDX_SPL = 4, 5, 6, 7, 8

# ══════════════════════════════════════════════════════════════════════════════
# Plot: unified nine-panel diagnostic
# ══════════════════════════════════════════════════════════════════════════════
def unified_plot(cfg: Dict, out_dir: Path) -> None:
    styles = cfg["plot"].get("styles", {})
    fig, axs2d = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
    axs = axs2d.flatten()

    # legend handles -----------------------------------------------------------
    idx_handles: List[Line2D] = []
    for j, idx in enumerate(cfg["indexes"]):
        nm, st = idx["name"], styles.get(idx["name"], {})
        idx_handles.append(Line2D([0], [0],
                                  color=st.get("color", f"C{j}"),
                                  marker=st.get("marker", "o"),
                                  ls="", markersize=6, label=nm))

    # traces -------------------------------------------------------------------
    for j, idx in enumerate(cfg["indexes"]):
        name = idx["name"]
        st = styles.get(name, {})
        colour = st.get("color", f"C{j}")
        marker = st.get("marker", "o")

        csv = out_dir / name / "results.csv"
        if not csv.exists():
            log.warning("[unified_plot] %s missing – skipped.", csv)
            continue
        df = pd.read_csv(csv)

        # latency --------------------------------------------------------------
        for op, ax_idx in zip(LAT_OPS, [IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M]):
            ax = axs[ax_idx]
            if op == "maintain":
                sub = df[df.maintenance_time_ms.fillna(0) > 0]
                y = sub.maintenance_time_ms
            else:
                sub = df[df.operation_type == op]
                y = sub.latency_ms
            if not sub.empty:
                ax.plot(sub.operation_number, y, color=colour, **OP_STYLE[op])

        # partitions -----------------------------------------------------------
        ax = axs[IDX_PART]
        sub = df[df.n_list.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.n_list,
                    color=colour, marker=marker, lw=1.2)

        # resident vectors -----------------------------------------------------
        ax = axs[IDX_RES]
        sub = df[df.n_resident.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.n_resident,
                    color=colour, marker=marker, lw=1.2)

        # recall ---------------------------------------------------------------
        ax = axs[IDX_REC]
        sub = df[(df.operation_type == "query") & df.recall.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.recall,
                    color=colour, marker=marker, lw=1.2)

        # running total time ---------------------------------------------------
        ax = axs[IDX_TOT]
        total_ms = df.latency_ms.fillna(0) + df.maintenance_time_ms.fillna(0)
        ax.plot(df.operation_number, np.cumsum(total_ms),
                color=colour, marker=marker, lw=1.2)

        # cumulative splits / deletes -----------------------------------------
        ax = axs[IDX_SPL]
        spl, dlt = df.n_splits.fillna(0), df.n_deletes.fillna(0)
        if (spl > 0).any():
            ax.step(df.operation_number, np.cumsum(spl),
                    where="post", color=colour, ls="--", lw=1.2)
        if (dlt > 0).any():
            ax.step(df.operation_number, np.cumsum(dlt),
                    where="post", color=colour, ls=":",  lw=1.2)

    # titles / grids -----------------------------------------------------------
    titles = {
        IDX_LAT_Q: "Latency – Query",
        IDX_LAT_I: "Latency – Insert",
        IDX_LAT_D: "Latency – Delete",
        IDX_LAT_M: "Latency – Maintain",
        IDX_PART:  "# Partitions",
        IDX_RES:   "Resident Vectors",
        IDX_REC:   "Recall",
        IDX_TOT:   "Running Total Time",
        IDX_SPL:   "Cumulative Splits / Deletes",
    }
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    axs[IDX_PART].set_ylabel("count")
    axs[IDX_RES].set_ylabel("# vectors")
    axs[IDX_REC].set_ylabel("recall")
    axs[IDX_TOT].set_ylabel("ms")
    axs[IDX_SPL].set_ylabel("count")

    for idx in [IDX_PART, IDX_RES, IDX_REC, IDX_TOT, IDX_SPL]:
        axs[idx].set_xlabel("Operation #")

    all_ops = max((pd.read_csv(out_dir / idx["name"] / "results.csv").operation_number.max()
                   for idx in cfg["indexes"]
                   if (out_dir / idx["name"] / "results.csv").exists()),
                  default=None)
    if all_ops is not None:
        for ax in axs:
            ax.set_xlim(left=0, right=all_ops)

    fig.legend(idx_handles,
               [h.get_label() for h in idx_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=min(4, len(idx_handles)),
               fontsize=9, title="Index", frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "unified_plot.png"
    plt.savefig(out_path, dpi=150)
    log.info("[unified_plot] → %s", out_path)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Plot: stacked-bar time breakdown
# ══════════════════════════════════════════════════════════════════════════════
def make_time_breakdown(cfg: Dict, out_dir: Path) -> None:
    categories = ["Search", "Insert", "Delete", "Maintain", "Total"]
    idx_names = [idx["name"] for idx in cfg["indexes"]]
    totals, recalls = {}, {}
    cmap = get_cmap("tab10")

    for j, idx in enumerate(cfg["indexes"]):
        name = idx["name"]; csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[time_breakdown] missing %s – skipped", csv_path)
            continue
        df = pd.read_csv(csv_path)
        totals[name] = [
            df[df.operation_type == "query"].latency_ms.sum(),
            df[df.operation_type == "insert"].latency_ms.sum(),
            df[df.operation_type == "delete"].latency_ms.sum(),
            df.maintenance_time_ms.fillna(0).sum(),
        ]
        totals[name].append(sum(totals[name]))
        recalls[name] = df[df.operation_type == "query"].recall.mean()

    n_idx, n_cats = len(totals), len(categories)
    x = np.arange(n_cats); width = 0.8 / max(1, n_idx)

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, (name, values) in enumerate(totals.items()):
        style = cfg["plot"].get("styles", {}).get(name, {})
        color = style.get("color", cmap(j % 10))
        x_off = x + (j - (n_idx - 1) / 2) * width
        ax.bar(x_off, values, width=width, label=name,
               color=color, edgecolor="black")
        ax.text(x_off[0], values[0] * 1.01,
                f"R={recalls[name]:.3f}", ha="center",
                va="bottom", fontsize=7, rotation=90, color=color)

    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Cumulative Time (ms)")
    ax.set_title("Time Breakdown per Index")
    ax.legend(title="Index", frameon=False)
    plt.tight_layout()

    out_path = out_dir / "time_breakdown.png"
    plt.savefig(out_path, dpi=150)
    log.info("[time_breakdown] → %s", out_path)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Summary table (CSV + Markdown)
# ══════════════════════════════════════════════════════════════════════════════
def produce_summary_table(cfg: Dict, out_dir: Path) -> None:
    """
    Write summary_table.csv and summary_table.md with columns:
        Index | Search | Insert | Delete | Maintain | Total | Recall | Partitions
    Times are cumulative milliseconds; Recall is mean over all queries.
    """
    rows = []
    for idx in cfg["indexes"]:
        name     = idx["name"]
        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[summary_table] %s missing – skipped", csv_path)
            continue

        df = pd.read_csv(csv_path)
        search   = df[df.operation_type == "query" ].latency_ms.sum()
        insert   = df[df.operation_type == "insert"].latency_ms.sum()
        delete   = df[df.operation_type == "delete"].latency_ms.sum()
        maintain = df.maintenance_time_ms.fillna(0).sum()
        total    = search + insert + delete + maintain
        recall   = df[df.operation_type == "query"].recall.mean()  # ← NEW
        n_part   = (df.n_list.dropna().iloc[-1]
                    if not df.n_list.dropna().empty else None)

        rows.append(dict(Index=name,
                         Search=int(search),
                         Insert=int(insert),
                         Delete=int(delete),
                         Maintain=int(maintain),
                         Total=int(total),
                         Recall=f"{recall:.4f}" if pd.notna(recall) else "—",
                         Partitions=n_part))

    if not rows:
        log.warning("[summary_table] no data – nothing written")
        return

    df = pd.DataFrame(rows).sort_values("Index")

    csv_path = out_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    log.info("[summary_table] → %s", csv_path)

    md_path = out_dir / "summary_table.md"
    md = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    md_path.write_text(md + "\n")
    log.info("[summary_table] → %s", md_path)
    log.info("\n%s", md)
# ══════════════════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(cfg_path: str, output_dir: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)

    # 1. dataset & workload ----------------------------------------------------
    if cfg["mode"] in {"build", "run"}:
        ds = cfg["dataset"]
        vecs, queries, _ = load_dataset(ds["name"], ds.get("path", ""))
        gen = DynamicWorkloadGenerator(
            workload_dir               = out,
            base_vectors               = vecs,
            metric                     = ds["metric"],
            insert_ratio               = cfg["workload_generator"]["insert_ratio"],
            delete_ratio               = cfg["workload_generator"]["delete_ratio"],
            query_ratio                = cfg["workload_generator"]["query_ratio"],
            update_batch_size          = cfg["workload_generator"]["update_batch_size"],
            query_batch_size           = cfg["workload_generator"]["query_batch_size"],
            number_of_operations       = cfg["workload_generator"]["number_of_operations"],
            initial_size               = cfg["workload_generator"]["initial_size"],
            cluster_size               = cfg["workload_generator"]["cluster_size"],
            cluster_sample_distribution= cfg["workload_generator"]["cluster_sample_distribution"],
            queries                    = queries,
            query_cluster_sample_distribution=cfg["workload_generator"]["query_cluster_sample_distribution"],
            seed                       = cfg["workload_generator"]["seed"])
        if not gen.workload_exists() or cfg["overwrite"].get("workload", False):
            log.info("Generating workload …")
            gen.generate_workload()
        else:
            log.info("Existing workload found – generation skipped.")

    # 2. evaluate --------------------------------------------------------------
    if cfg["mode"] == "run":
        for idx in cfg["indexes"]:
            name, res = idx["name"], out / idx["name"]
            res.mkdir(parents=True, exist_ok=True)
            csv = res / "results.csv"
            if csv.exists() and not cfg["overwrite"].get("results", False):
                log.info("%s: cached results – skipped.", name); continue

            eval = WorkloadEvaluator(workload_dir=out, output_dir=res)
            cls  = {"Quake": QuakeWrapper, "IVF": FaissIVF}[idx["index"]]
            eval.evaluate_workload(
                name          = name,
                index         = cls(),
                build_params  = idx.get("build_params", {}),
                search_params = idx.get("search_params", {}),
                m_params      = idx.get("maintenance_params", {}),
                do_maintenance=True,
            )

    # 3. global artefacts ------------------------------------------------------
    if cfg["mode"] in {"run", "plot"}:
        unified_plot(cfg, out)
        make_time_breakdown(cfg, out)
        produce_summary_table(cfg, out)