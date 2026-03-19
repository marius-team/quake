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

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import yaml

# ── quake imports ──────────────────────────────────────────────────────────────
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake import QuakeWrapper
from quake.utils import compute_recall
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

logger = logging.getLogger(__name__)


LAT_STYLES = {"insert": "--", "delete": "-.", "query": "-"}

def unified_plot(cfg: Dict, output_dir: Path) -> None:
    """Produce a 4‑panel plot with:
       • one figure‑level legend (colour → index)
       • latency‑panel legend (linestyle → op‑type)"""
    styles = cfg["plot"].get("styles", {})

    # ── collect colours / markers per index for global legend ────────────────
    index_handles = []
    for idx_cfg in cfg["indexes"]:
        name = idx_cfg["name"]
        st = styles.get(name, {})
        index_handles.append(
            Line2D(
                [0],
                [0],
                color=st.get("color", None),
                marker=None,
                linestyle="-",
                label=name,
            )
        )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ── panel references ─────────────────────────────────────────────────────
    ax_lat, ax_part = axs[0]
    ax_res, ax_rec = axs[1]

    # ── plot per index ───────────────────────────────────────────────────────
    for idx_cfg in cfg["indexes"]:
        name = idx_cfg["name"]
        csv = output_dir / name / "results.csv"
        if not csv.exists():
            logger.warning("Skipping %s (no CSV).", name); continue

        df = pd.read_csv(csv)
        st = styles.get(name, {})
        color, marker = st.get("color", None), st.get("marker", None)

        # A – latency
        for op in ("insert", "delete", "query"):
            sub = df[df["operation_type"] == op]
            if not sub.empty:
                ax_lat.plot(
                    sub["operation_number"], sub["latency_ms"],
                    color=color, linestyle=LAT_STYLES[op], marker=None)

        ax_lat.set_yscale("log")

        # B – partitions
        sub = df[df["n_list"].notna()]
        if not sub.empty:
            ax_part.plot(sub["operation_number"], sub["n_list"],
                         color=color, marker=marker)

        # C – resident vectors
        sub = df[df["n_resident"].notna()]
        if not sub.empty:
            ax_res.plot(sub["operation_number"], sub["n_resident"],
                        color=color, marker=marker)

        # D – query recall
        sub = df[(df["operation_type"] == "query") & df["recall"].notna()]
        if not sub.empty:
            ax_rec.plot(sub["operation_number"], sub["recall"],
                        color=color, marker=marker)

    # ── format panels (titles, axes) ─────────────────────────────────────────
    ax_lat.set_title("Operation Latency")
    ax_lat.set_xlabel("Operation Number"); ax_lat.set_ylabel("Latency (ms)")

    ax_part.set_title("Partitions per Operation")
    ax_part.set_xlabel("Operation Number"); ax_part.set_ylabel("Number of Partitions")

    ax_res.set_title("Resident Set Size")
    ax_res.set_xlabel("Operation Number"); ax_res.set_ylabel("Resident Vectors")

    ax_rec.set_title("Query Recall")
    ax_rec.set_xlabel("Operation Number"); ax_rec.set_ylabel("Recall")

    # legends
    # latency‑style legend (insert/delete/query)
    style_handles = [
        Line2D([0], [0], color="k", linestyle=LAT_STYLES[op], label=op.capitalize())
        for op in ("insert", "delete", "query")
    ]
    ax_lat.legend(handles=style_handles, title="Operation Type", fontsize=8)

    # global colour legend (index names)
    fig.legend(
        handles=index_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(index_handles)),
        title="Index",
    )

    plt.tight_layout()
    out_path = output_dir / "unified_plot.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved unified plot → {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# main execution
# ───────────────────────────────────────────────────────────────────────────────
def run_experiment(cfg_path: str, output_dir: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode = cfg.get("mode", "run")
    overwrite = cfg.get("overwrite", {})
    overwrite_workload = overwrite.get("workload", False)
    overwrite_results = overwrite.get("results", False)

    # ── build / run both need the dataset loaded ───────────────────────────────
    if mode in {"build", "run"}:
        data_cfg = cfg["dataset"]
        vectors, queries, _ = load_dataset(data_cfg["name"], data_cfg.get("path", ""))
        metric = data_cfg["metric"]
        print(f"Dataset loaded: vectors {vectors.shape}, queries {queries.shape}")

    # ── workload generation (build or run) ─────────────────────────────────────
    if mode in {"build", "run"}:
        wg_cfg = cfg["workload_generator"]
        generator = DynamicWorkloadGenerator(
            workload_dir=out,
            base_vectors=vectors,
            metric=metric,
            insert_ratio=wg_cfg["insert_ratio"],
            delete_ratio=wg_cfg["delete_ratio"],
            query_ratio=wg_cfg["query_ratio"],
            update_batch_size=wg_cfg["update_batch_size"],
            query_batch_size=wg_cfg["query_batch_size"],
            number_of_operations=wg_cfg["number_of_operations"],
            initial_size=wg_cfg["initial_size"],
            cluster_size=wg_cfg["cluster_size"],
            cluster_sample_distribution=wg_cfg["cluster_sample_distribution"],
            queries=queries,
            query_cluster_sample_distribution=wg_cfg["query_cluster_sample_distribution"],
            seed=wg_cfg["seed"],
        )

        if generator.workload_exists() and not overwrite_workload:
            print("Workload already exists – skipping generation.")
        else:
            print("Generating workload …")
            generator.generate_workload()

    # ── evaluation pass (run mode only) ────────────────────────────────────────
    if mode == "run":
        for idx_cfg in cfg["indexes"]:
            name = idx_cfg["name"]
            index_type = idx_cfg["index"]
            res_dir = out / name
            csv_path = res_dir / "results.csv"
            need_eval = overwrite_results or not csv_path.exists()

            if need_eval:
                print(f"\nIndex {name}: running evaluation.")
                res_dir.mkdir(parents=True, exist_ok=True)
                evaluator = WorkloadEvaluator(workload_dir=out, output_dir=res_dir)

                index = {"Quake": QuakeWrapper, "IVF": FaissIVF}.get(index_type)
                if index is None:
                    raise ValueError(f"Unknown index type: {index_type}")

                results = evaluator.evaluate_workload(
                    name=name,
                    index=index(),
                    build_params=idx_cfg["build_params"],
                    search_params=idx_cfg["search_params"],
                    do_maintenance=True,
                )
            else:
                print(f"\nIndex {name}: using cached results ({csv_path}).")

    # ── generate / regenerate the unified plot ─────────────────────────────────
    if mode in {"run", "plot"}:
        unified_plot(cfg, out)