#!/usr/bin/env python3
"""
Unified experiment runner

* Reads a single YAML config (`configs/experiment.yaml`).
* Modes
    - **build** –‑ only (re‑)generate the workload.
    - **run**   –‑ generate workload (if needed), evaluate every index that
                  does not already have a CSV of prior results (unless
                  `overwrite.results: true`), then emit the unified plot.
    - **plot**  –‑ skip evaluation; just regenerate the unified plot.
* A results CSV (`<output>/<index‑name>/results.csv`) is written after every
  evaluation; its presence is what lets us skip reruns.
* Plot styling (colour / marker / linestyle) is entirely configurable from
  the YAML under `plot.styles`.
"""

import logging, shutil, time, yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

# ── quake imports ──────────────────────────────────────────────────────────
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.utils import compute_recall
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("runner")

INDEX_CLASSES = {
    "Quake": QuakeWrapper,
    "IVF":   FaissIVF,
}

LAT_STYLE = {"insert": "--", "delete": "-.", "query": "-"}


# ──────────────────────────────────────────────────────────────────────────
def unified_plot(cfg: Dict[str, Any], out_dir: Path) -> None:
    styles = cfg.get("plot", {}).get("styles", {})
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_lat, ax_part = axs[0]
    ax_res, ax_rec  = axs[1]

    legend_idx: List[Line2D] = []

    for idx in cfg["indexes"]:
        name   = idx["name"]
        csv    = out_dir / name / "results.csv"
        if not csv.exists():
            log.warning("Skipping plot for %s (no CSV).", name)
            continue
        df   = pd.read_csv(csv)
        styl = styles.get(name, {})
        col, mark = styl.get("color"), styl.get("marker")

        # latency
        for op in ("insert", "delete", "query"):
            sub = df[df.operation_type == op]
            if not sub.empty:
                ax_lat.plot(sub.operation_number, sub.latency_ms,
                            color=col, linestyle=LAT_STYLE[op])
        # partitions
        if "n_list" in df.columns:
            ax_part.plot(df.operation_number, df.n_list, color=col, marker=mark)
        # resident
        if "n_resident" in df.columns:
            ax_res.plot(df.operation_number, df.n_resident, color=col, marker=mark)
        # recall
        q = df[(df.operation_type == "query") & df.recall.notna()]
        if not q.empty:
            ax_rec.plot(q.operation_number, q.recall, color=col, marker=mark)

        legend_idx.append(Line2D([0], [0], color=col, label=name))

    ax_lat.set_yscale("log"); ax_lat.set_title("Latency"); ax_lat.set_xlabel("Op #")
    ax_part.set_title("#Partitions"); ax_part.set_xlabel("Op #")
    ax_res.set_title("Resident"); ax_res.set_xlabel("Op #")
    ax_rec.set_title("Recall"); ax_rec.set_xlabel("Op #"); ax_rec.set_ylim(0, 1)

    style_handles = [Line2D([0], [0],
                            color="k", ls=LAT_STYLE[o], label=o.title())
                     for o in ("insert", "delete", "query")]
    ax_lat.legend(handles=style_handles, title="Op type", fontsize=8)
    fig.legend(handles=legend_idx, loc="upper center", ncol=max(1, len(legend_idx)),
               title="Index", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    path = out_dir / "unified_plot.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"Unified plot → {path}")


# ──────────────────────────────────────────────────────────────────────────
def _should(overwrite: bool, path: Path) -> bool:
    return overwrite or not path.exists()


# ──────────────────────────────────────────────────────────────────────────
def _evaluate_index(cfg: Dict[str, Any], out_root: Path, batch: bool,
                    overwrite_results: bool) -> None:
    workload_dir = out_root         # workload lives in the output root
    for idx in cfg["indexes"]:
        name   = idx["name"]
        res_dir = out_root / name
        csv     = res_dir / "results.csv"
        if not _should(overwrite_results, csv):
            print(f"{name}: cached results -> skip evaluation"); continue

        cls_key = idx["index"]
        cls     = INDEX_CLASSES.get(cls_key)
        if cls is None:
            log.error("Unknown index wrapper %s – skip %s", cls_key, name); continue
        res_dir.mkdir(parents=True, exist_ok=True)

        evaluator = WorkloadEvaluator(workload_dir=workload_dir, output_dir=res_dir)
        evaluator.evaluate_workload(
            name          = name,
            index         = cls(),
            build_params  = idx.get("build_params", {}),
            search_params = idx.get("search_params", {}),
            do_maintenance= idx.get("maintenance_params") is not None,
            m_params      = idx.get("maintenance_params"),
            batch         = batch,
        )


# ──────────────────────────────────────────────────────────────────────────
def _build_workload(cfg: Dict[str, Any], out_root: Path,
                    overwrite_workload: bool) -> None:

    print("overwrite_workload:", overwrite_workload, "out_root:", out_root)

    if not _should(overwrite_workload, out_root / "runbook.json"):
        print("Workload exists – skip generation"); return

    ds_cfg = cfg["dataset"]
    vecs, qvecs, _ = load_dataset(ds_cfg["name"], ds_cfg.get("path", ""))
    if ds_cfg["metric"] == "l2":   # ensure normalisation for L2 if desired
        pass
    wg_cfg = cfg["workload_generator"]
    gen = DynamicWorkloadGenerator(
        workload_dir=out_root,
        base_vectors=vecs,
        metric=ds_cfg["metric"],
        insert_ratio=wg_cfg["insert_ratio"],
        delete_ratio=wg_cfg["delete_ratio"],
        query_ratio =wg_cfg["query_ratio"],
        number_of_operations=wg_cfg["number_of_operations"],
        initial_size=wg_cfg["initial_size"],
        cluster_size=wg_cfg["cluster_size"],
        update_batch_size=wg_cfg["update_batch_size"],
        query_batch_size =wg_cfg["query_batch_size"],
        cluster_sample_distribution=wg_cfg["cluster_sample_distribution"],
        queries=qvecs,
        query_cluster_sample_distribution=wg_cfg["query_cluster_sample_distribution"],
        seed=wg_cfg["seed"],
    )
    print("Generating workload …"); gen.generate_workload()


# ──────────────────────────────────────────────────────────────────────────
def run_experiment(cfg_src: str | Dict[str, Any], output_dir: str | Path) -> None:
    cfg = yaml.safe_load(Path(cfg_src).read_text()) if isinstance(cfg_src, (str, Path)) else cfg_src
    out_root = Path(output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    mode      = cfg.get("mode", "run")
    overwrite_results = cfg.get("overwrite_results", False)
    overwrite_workload = cfg.get("overwrite_workload", False)
    batch     = cfg.get("batch", False)

    if mode in {"build", "run"}:
        _build_workload(cfg, out_root, overwrite_workload)
    if mode == "run":
        _evaluate_index(cfg, out_root, batch, overwrite_results)
    if mode in {"run", "plot"}:
        unified_plot(cfg, out_root)