#!/usr/bin/env python3
"""
Wikidata Workload Experiment Runner
─────────────────────────────────────
This experiment runs a Wikidata workload across multiple index configurations.
It generates detailed reports including:
1. Per-index CSV logs of operations.
2. A 9-panel unified plot comparing various metrics across configurations.
3. A stacked bar chart breaking down cumulative time per operation type.
4. A summary table (CSV and Markdown) of key performance indicators.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# Common utilities
import test.experiments.osdi2025.experiment_utils as common_utils

# evaluator and wrappers
from .wiki_workload import WikidataWorkloadEvaluator
from quake.index_wrappers.quake import QuakeWrapper
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
try: from quake.index_wrappers.scann import Scann
except ImportError: Scann = None
try: from quake.index_wrappers.diskann import DiskANNDynamic
except ImportError: DiskANNDynamic = None
try: from quake.index_wrappers.vamana import Vamana
except ImportError: Vamana = None

INDEX_CLASSES: Dict[str, Any] = {
    "Quake": QuakeWrapper,
    "IVF": FaissIVF,
    "HNSW": FaissHNSW,
    "SCANN": Scann,
    "DiskANN": DiskANNDynamic,
    "SVS": Vamana,
}

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
    styles = cfg.get("plot", {}).get("styles", {})
    fig, axs2d = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
    axs = axs2d.flatten()

    idx_handles: List[Line2D] = []
    for j, idx_cfg in enumerate(cfg.get("indexes", [])):
        nm = idx_cfg["name"]
        st = styles.get(nm, {})
        idx_handles.append(Line2D([0], [0],
                                  color=st.get("color", f"C{j % 10}"),
                                  marker=st.get("marker", "o"),
                                  ls="", markersize=6, label=nm))

    max_ops_overall = 0
    for j, idx_cfg in enumerate(cfg.get("indexes", [])):
        name = idx_cfg["name"]
        colour = styles.get(name, {}).get("color", f"C{j % 10}")
        marker = styles.get(name, {}).get("marker", "o")

        csv_path = out_dir / name / "results.csv"
        if not csv_path.exists():
            log.warning("[unified_plot] missing %s", csv_path)
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty: continue
            max_ops_overall = max(max_ops_overall, df.operation_number.max())
        except Exception:
            log.warning("[unified_plot] failed to read %s", csv_path)
            continue

        # latency
        for op, ax_idx in zip(LAT_OPS, [IDX_LAT_Q, IDX_LAT_I, IDX_LAT_D, IDX_LAT_M]):
            ax = axs[ax_idx]
            if op != "maintain":
                sub = df[df.operation_type == op]
                if not sub.empty:
                    ax.plot(sub.operation_number, sub.latency_ms, color=colour, **OP_STYLE[op])

        # partitions, resident blank
        # recall
        rec_ix = IDX_REC
        sub = df[df.operation_type=="query"]
        if "recall" in sub:
            axs[rec_ix].plot(sub.operation_number, sub.recall, color=colour, marker=marker, lw=1.2)
        # cumulative time
        tot_ix = IDX_TOT
        cum = np.cumsum(df.latency_ms.fillna(0))
        axs[tot_ix].plot(df.operation_number, cum, color=colour, marker=marker, lw=1.2)

    # titles and labels
    titles = {
        IDX_LAT_Q: "Latency – Query", IDX_LAT_I: "Insert", IDX_LAT_D: "Delete", IDX_LAT_M: "Maintain",
        IDX_PART: "# Partitions", IDX_RES: "Resident", IDX_REC: "Recall",
        IDX_TOT: "Cum. Time (ms)", IDX_SPL: "Splits/Deletes"
    }
    for i, ax in enumerate(axs):
        ax.set_title(titles.get(i, ""))
        ax.grid(True, ls=":", alpha=0.7)
        if i//3==2: ax.set_xlabel("Op #")
    if max_ops_overall>0:
        for ax in axs: ax.set_xlim(0, max_ops_overall)

    fig.legend(idx_handles, [h.get_label() for h in idx_handles],
               loc="upper center", ncol=min(4,len(idx_handles)), frameon=False)
    plt.tight_layout(rect=[0,0.03,1,0.97])
    path = out_dir / "unified_plot.png"
    plt.savefig(path, dpi=150)
    log.info("Saved unified_plot to %s", path)
    plt.close(fig)


def make_time_breakdown(cfg: Dict[str, Any], out_dir: Path) -> None:
    categories = ["Query","Insert","Delete","Total"]
    data = {}
    for idx_cfg in cfg.get("indexes", []):
        name = idx_cfg["name"]
        df = pd.read_csv(out_dir/name/"results.csv")
        data[name] = [
            df[df.operation_type=="query"].latency_ms.sum(),
            df[df.operation_type=="insert"].latency_ms.sum(),
            df[df.operation_type=="delete"].latency_ms.sum(),
        ]
        data[name].append(sum(data[name]))
    fig, ax = plt.subplots(figsize=(8,6))
    cmap = get_cmap("tab10")
    n = len(data)
    x = np.arange(len(categories))
    w = 0.8/n
    for j,(name,vals) in enumerate(data.items()):
        ax.bar(x + j*w, vals, width=w, label=name, color=cmap(j))
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Cumulative ms")
    ax.legend(frameon=False)
    path = out_dir/"time_breakdown.png"
    plt.savefig(path, dpi=150)
    log.info("Saved time_breakdown to %s", path)
    plt.close(fig)


def produce_summary_table(cfg: Dict[str, Any], out_dir: Path) -> None:
    rows = []
    for idx_cfg in cfg.get("indexes", []):
        name = idx_cfg["name"]
        df = pd.read_csv(out_dir/name/"results.csv")
        q = df[df.operation_type=="query"]
        rows.append({
            "Index": name,
            "MeanLatency": float(q.latency_ms.mean()),
            "MeanRecall": float(q.recall.mean()) if "recall" in q else None,
        })
    if not rows: return
    sdf = pd.DataFrame(rows)
    common_utils.save_results_csv(sdf, out_dir/"summary_table.csv")
    md = tabulate(sdf, headers="keys", tablefmt="github", showindex=False)
    print(md)


def run_experiment(cfg_path_str: str, output_dir_str: str) -> None:
    cfg = common_utils.load_config(cfg_path_str)
    out = Path(output_dir_str).expanduser(); out.mkdir(parents=True, exist_ok=True)
    mode = cfg.get("mode","run")
    log.info("Mode: %s", mode)

    workload_dir = cfg["workload_dir"]

    if mode in {"run"}:  # Phase 2: evaluation
        for idx_cfg in cfg.get("indexes", []):
            name = idx_cfg["name"]
            key  = idx_cfg["index"]
            Cls = INDEX_CLASSES.get(key)
            if Cls is None:
                log.warning("Unknown index %s", key); continue
            wrapper = Cls()
            maint = idx_cfg.get("maintenance_params")
            do_maint = maint is not None
            ev = WikidataWorkloadEvaluator(workload_dir, out/name)
            ev.evaluate_workload(
                name=name,
                index=wrapper,
                build_params=idx_cfg.get("build_params",{}),
                search_params=idx_cfg.get("search_params",{}),
                do_maintenance=do_maint,
                m_params=maint,
                batch=bool(idx_cfg.get("batch",False)),
            )
    if mode in {"run","plot"}:  # Phase 3
        unified_plot(cfg, out)
        make_time_breakdown(cfg, out)
        produce_summary_table(cfg, out)