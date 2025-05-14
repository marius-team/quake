#!/usr/bin/env python3
"""
Quake maintenance–ablation runner
─────────────────────────────────
• Builds / runs / plots exactly like the original “kick-the-tires” script.
• Per-operation CSV rows now contain
      n_splits   n_deletes   maintenance_time_ms
  (written by WorkloadEvaluator below).
• After all indices finish, this script also produces
      <output>/<index>/time_breakdown.png
  – a stacked-bar chart of cumulative time spent in
    Search / Insert / Delete / Maintenance / Total.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
import numpy as np, pandas as pd, yaml

# quake wrappers ----------------------------------------------------------------
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake    import QuakeWrapper
from quake.workload_generator      import DynamicWorkloadGenerator, WorkloadEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


OP_STYLE = {
    "query"   : dict(ls="-",   marker="o", mfc="none", ms=4, lw=1.2),
    "insert"  : dict(ls="-",   marker="s", mfc="none", ms=4, lw=1.2),
    "delete"  : dict(ls="-",   marker="^", mfc="none", ms=4, lw=1.2),
    "maintain": dict(ls="None", marker="X", mfc="black", ms=5),   # points only
}

LAT_OPS = ["query", "insert", "delete", "maintain"]
IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M = 0, 1, 2, 3
IDX_PART,  IDX_RES,   IDX_REC,   IDX_TOT, IDX_SPL = 4, 5, 6, 7, 8


def unified_plot(cfg: Dict, out_dir: Path) -> None:
    styles = cfg["plot"].get("styles", {})

    fig, axs2d = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
    axs = axs2d.flatten()

    # ── index legend handles (colour / marker) ────────────────────────────────
    idx_handles: List[Line2D] = []
    for j, idx in enumerate(cfg["indexes"]):
        nm, st = idx["name"], styles.get(idx["name"], {})
        idx_handles.append(
            Line2D([0], [0],
                   color   = st.get("color", f"C{j}"),
                   marker  = st.get("marker", "o"),
                   ls      = "",
                   markersize = 6,
                   label   = nm))

    # ── plot each index ──────────────────────────────────────────────────────
    for j, idx in enumerate(cfg["indexes"]):
        name   = idx["name"]
        st     = styles.get(name, {})
        colour = st.get("color", f"C{j}")
        marker = st.get("marker", "o")

        csv = out_dir / name / "results.csv"
        if not csv.exists():
            print(f"[unified_plot] warning: {csv} missing – skipped")
            continue

        df = pd.read_csv(csv)

        # latency panels ------------------------------------------------------
        for op, ax_idx in zip(LAT_OPS,
                              [IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M]):
            ax = axs[ax_idx]
            if op == "maintain":
                sub = df[df.maintenance_time_ms.fillna(0) > 0]
                y   = sub.maintenance_time_ms
            else:
                sub = df[df.operation_type == op]
                y   = sub.latency_ms
            if sub.empty:
                continue
            ax.plot(sub.operation_number, y,
                    color=colour,
                    **OP_STYLE[op])

        # partitions ----------------------------------------------------------
        ax = axs[IDX_PART]
        sub = df[df.n_list.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.n_list,
                    color=colour, marker=marker, lw=1.2)

        # resident vectors ----------------------------------------------------
        ax = axs[IDX_RES]
        sub = df[df.n_resident.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.n_resident,
                    color=colour, marker=marker, lw=1.2)

        # recall --------------------------------------------------------------
        ax = axs[IDX_REC]
        sub = df[(df.operation_type == "query") & df.recall.notna()]
        if not sub.empty:
            ax.plot(sub.operation_number, sub.recall,
                    color=colour, marker=marker, lw=1.2)

        # running total time --------------------------------------------------
        ax = axs[IDX_TOT]
        total_ms = df.latency_ms.fillna(0) + df.maintenance_time_ms.fillna(0)
        ax.plot(df.operation_number, np.cumsum(total_ms),
                color=colour, marker=marker, lw=1.2)

        # cumulative splits / deletes ----------------------------------------
        ax = axs[IDX_SPL]
        spl = df.n_splits.fillna(0)
        dlt = df.n_deletes.fillna(0)
        if (spl > 0).any():
            ax.step(df.operation_number, np.cumsum(spl),
                    where="post", color=colour, ls="--", lw=1.2)
        if (dlt > 0).any():
            ax.step(df.operation_number, np.cumsum(dlt),
                    where="post", color=colour, ls=":", lw=1.2)

    # ── titles / labels / grids ──────────────────────────────────────────────
    titles = {
        IDX_LAT_Q: "Latency – Query",
        IDX_LAT_I: "Latency – Insert",
        IDX_LAT_D: "Latency – Delete",
        IDX_LAT_M: "Latency – Maintain",
        IDX_PART : "# Partitions",
        IDX_RES  : "Resident Vectors",
        IDX_REC  : "Recall",
        IDX_TOT  : "Running Total Time",
        IDX_SPL  : "Cumulative Splits / Deletes",
    }
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], fontsize=11)
        ax.grid(alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    axs[IDX_PART].set_ylabel("count")
    axs[IDX_RES ].set_ylabel("# vectors")
    axs[IDX_REC ].set_ylabel("recall")
    axs[IDX_TOT ].set_ylabel("ms")
    axs[IDX_SPL ].set_ylabel("count")

    # x-labels only on bottom row
    for idx in [IDX_PART, IDX_RES, IDX_REC, IDX_TOT, IDX_SPL]:
        axs[idx].set_xlabel("Operation #")

    # common x-limit so vertical grids align
    all_ops = max((pd.read_csv(out_dir/idx["name"]/ "results.csv").operation_number.max()
                   for idx in cfg["indexes"]
                   if (out_dir/idx["name"]/ "results.csv").exists()), default=None)
    if all_ops is not None:
        for ax in axs:
            ax.set_xlim(left=0, right=all_ops)

    # ── single legend (indexes only) ─────────────────────────────────────────
    fig.legend(idx_handles,
               [h.get_label() for h in idx_handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.03),
               ncol=min(4, len(idx_handles)),
               fontsize=9, title="Index", frameon=False)

    # save --------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.97])     # space for top legend
    out_path = out_dir / "unified_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"[unified_plot] → {out_path}")
    plt.close()

# ───────────────────────────────────────────────────────────────────────────────

def make_time_breakdown(cfg: Dict, out_dir: Path) -> None:
    """
    Create a side-by-side bar chart showing cumulative time spent in
    Search / Insert / Delete / Maintenance / Total for each index.

    Recall values are annotated above each Search bar.
    Output file: <out_dir>/time_breakdown.png
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
    import numpy as np
    import pandas as pd

    categories = ["Search", "Insert", "Delete", "Maintain", "Total"]
    idx_names  = [idx["name"] for idx in cfg["indexes"]]
    n_idx      = len(idx_names)
    n_cats     = len(categories)

    # Totals and recall tracking
    totals = {name: [0]*n_cats for name in idx_names}
    recalls = {}

    for idx in cfg["indexes"]:
        name     = idx["name"]
        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            print(f"[make_time_breakdown] warning: no CSV for {name} – skipped")
            continue

        df = pd.read_csv(csv_path)
        totals[name][0] = df[df.operation_type == "query" ].latency_ms.sum()              # Search
        totals[name][1] = df[df.operation_type == "insert"].latency_ms.sum()              # Insert
        totals[name][2] = df[df.operation_type == "delete"].latency_ms.sum()              # Delete
        totals[name][3] = df.maintenance_time_ms.fillna(0).sum()                          # Maintain
        totals[name][4] = sum(totals[name][:4])                                            # Total
        recalls[name]   = df[df.operation_type == "query"].recall.mean()                  # Recall

    # Plotting
    x = np.arange(n_cats)
    width = 0.8 / max(1, n_idx)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = get_cmap("tab10")

    for j, name in enumerate(idx_names):
        y = totals[name]
        style = cfg["plot"].get("styles", {}).get(name, {})
        color = style.get("color", cmap(j % 10))

        x_offset = x + (j - (n_idx - 1) / 2) * width
        bars = ax.bar(x_offset, y, width=width, label=name, color=color, edgecolor="black")

        # Add recall above Search bar (index 0)
        recall_val = recalls.get(name)
        if recall_val is not None:
            ax.text(x_offset[0], y[0] + 0.01 * max(y), f"Recall={recall_val:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Cumulative Time (ms)")
    ax.set_title("Time Breakdown per Index")
    ax.legend(title="Index", frameon=False)
    plt.tight_layout()

    out_path = out_dir / "time_breakdown.png"
    plt.savefig(out_path, dpi=150)
    print(f"[make_time_breakdown] → {out_path}")
    plt.close()

# ───────────────────────────────────────────────────────────────────────────────
def run_experiment(cfg_path:str, output_dir:str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir).expanduser();  out.mkdir(parents=True, exist_ok=True)

    # 1 dataset / workload ---------------------------------------------------
    if cfg["mode"] in {"build","run"}:
        ds = cfg["dataset"]
        vecs, queries, _ = load_dataset(ds["name"], ds.get("path",""))
        gen = DynamicWorkloadGenerator(
            workload_dir               =out,
            base_vectors               =vecs,
            metric                     =ds["metric"],
            insert_ratio               =cfg["workload_generator"]["insert_ratio"],
            delete_ratio               =cfg["workload_generator"]["delete_ratio"],
            query_ratio                =cfg["workload_generator"]["query_ratio"],
            update_batch_size          =cfg["workload_generator"]["update_batch_size"],
            query_batch_size           =cfg["workload_generator"]["query_batch_size"],
            number_of_operations       =cfg["workload_generator"]["number_of_operations"],
            initial_size               =cfg["workload_generator"]["initial_size"],
            cluster_size               =cfg["workload_generator"]["cluster_size"],
            cluster_sample_distribution=cfg["workload_generator"]["cluster_sample_distribution"],
            queries                    =queries,
            query_cluster_sample_distribution=cfg["workload_generator"]["query_cluster_sample_distribution"],
            seed                       =cfg["workload_generator"]["seed"])
        if not gen.workload_exists() or cfg["overwrite"].get("workload",False):
            log.info("Generating workload …");  gen.generate_workload()
        else:
            log.info("Existing workload found – generation skipped.")

    # 2 evaluation -----------------------------------------------------------
    if cfg["mode"]=="run":
        for idx in cfg["indexes"]:
            name, res = idx["name"], out/idx["name"]
            csv = res/"results.csv"
            if csv.exists() and not cfg["overwrite"].get("results",False):
                log.info("%s: cached results – skipped.", name);  continue
            res.mkdir(parents=True, exist_ok=True)
            eval = WorkloadEvaluator(workload_dir=out, output_dir=res)
            cls  = {"Quake":QuakeWrapper,"IVF":FaissIVF}[idx["index"]]
            eval.evaluate_workload(
                name           =name,
                index          =cls(),
                build_params   =idx.get("build_params",{}),
                search_params  =idx.get("search_params",{}),
                m_params=idx.get("maintenance_params",{}),
                do_maintenance=True,
            )

    # 3 global plot ----------------------------------------------------------
    if cfg["mode"] in {"run","plot"}:
        unified_plot(cfg, out)
        make_time_breakdown(cfg, out)
