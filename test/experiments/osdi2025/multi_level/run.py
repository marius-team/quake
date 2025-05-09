from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

# ── quake imports ──────────────────────────────────────────────────────────────
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake import QuakeWrapper
from quake.utils import compute_recall  # recall@k

logger = logging.getLogger("experiment")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ───────────────────────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────────────────────
def _prepare_index(
        idx_cls,
        vecs: torch.Tensor,
        metric: str,
        build_params: Dict,
        index_path: Path,
        rebuild: bool,
) -> Tuple[object, float]:
    """
    Build–or–load an index and return (index, build_time_seconds).
    build_time_seconds == 0 when loaded.
    """
    idx = idx_cls()
    if index_path.exists() and not rebuild:
        t0 = time.time()
        idx.load(str(index_path), num_workers=build_params["num_workers"])
        return idx, 0.0

    logger.info("Building index → %s …", index_path)
    t0 = time.time()
    idx.build(vecs, metric=metric, **build_params)
    t_build = time.time() - t0
    idx.save(str(index_path))
    logger.info("Index saved (%.1fs).", t_build)
    return idx, t_build


def _dist_ids(res):
    """Handle `(D, I)` tuples OR Quake SearchResult objects."""
    if isinstance(res, tuple) and len(res) == 2:
        return res
    if hasattr(res, "distances") and hasattr(res, "indices"):
        return res.distances, res.indices
    raise TypeError(f"Unknown search result type: {type(res).__name__}")


def _benchmark_index(
        idx,
        queries: torch.Tensor,
        ground: np.ndarray,
        search_params: Dict,
        trials: int,
        warmup: int,
) -> Dict:
    bs = search_params.pop("batch_size", None)
    k = search_params["k"]

    # warm-up
    for _ in range(warmup):
        if bs is None:
            idx.search(queries, **search_params)
        else:
            for i in range(0, len(queries), bs):
                idx.search(queries[i : i + bs], **search_params)

    qps_vals: List[float] = []
    lat_ms:   List[float] = []

    for _ in range(trials):
        t0 = time.time()
        if bs is None:
            res = idx.search(queries, **search_params)
            D, I = res.distances, res.ids
        else:
            all_I, all_D = [], []
            for i in range(0, len(queries), bs):
                res = idx.search(queries[i : i + bs], **search_params)
                D_, I_ = res.distances, res.ids
                all_I.append(I_); all_D.append(D_)
            I = np.concatenate(all_I); D = np.concatenate(all_D)
        dt = time.time() - t0
        qps_vals.append(len(queries) / dt)
        lat_ms.append(dt * 1e3)

    recall = compute_recall(I, ground[:, :k], k)
    recall = 0.0 if recall is None else recall.mean().item()

    return dict(
        recall           = float(recall),
        qps_mean         = float(np.mean(qps_vals)),
        qps_std          = float(np.std(qps_vals)),
        latency_mean_ms  = float(np.mean(lat_ms)),
        trials           = trials,
        warmup           = warmup,
        batch_size       = bs or len(queries),
    )


def _expand_targets(sp: Dict) -> List[Dict]:
    """
    Produce one search-parameter dict per recall_target value.

    If `recall_targets` is absent (or a single float) we fall back to a list
    containing exactly one dict, preserving the original behaviour.
    """
    if "recall_targets" in sp:
        tgs = sp.pop("recall_targets")
        if not isinstance(tgs, (list, tuple)):
            tgs = [tgs]
        return [{**sp, "recall_target": t} for t in tgs]
    elif "nprobes" in sp:
        tgs = sp.pop("nprobes")
        if not isinstance(tgs, (list, tuple)):
            tgs = [tgs]
        return [{**sp, "nprobe": t} for t in tgs]
    else:
        return [sp]


def _plot_unified(cfg: Dict, root: Path) -> None:
    styles = cfg["plot"]["styles"]
    fig, ax = plt.subplots(figsize=(6, 4))

    for icfg in cfg["indexes"]:
        name = icfg["name"]
        csv  = root / name / "results.csv"
        if not csv.exists():
            logger.warning("No results for %s – skipped in plot.", name)
            continue

        df = pd.read_csv(csv).sort_values("recall")
        sty = styles.get(name, {})
        ax.plot(
            df["qps_mean"], df["recall"],
            label     = name,
            marker    = sty.get("marker", "o"),
            linestyle = sty.get("linestyle", "-"),
            linewidth = 1.4,
            markersize= 6,
            color     = sty.get("color", None),
        )

    ax.set_xscale("log", base=10)
    ax.set_xlabel("Queries per second (log)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall curves over varied recall_target")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    out_png = root / "recall_vs_qps.png"
    fig.savefig(out_png, dpi=180)
    logger.info("Plot written to %s", out_png)


# ───────────────────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────────────────
def run_experiment(cfg_path: str, output_dir: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    mode           = cfg.get("mode", "run")            # build | run | plot
    overwrite      = cfg.get("overwrite", {})
    rebuild_index  = bool(overwrite.get("index",   False))
    rewrite_csv    = bool(overwrite.get("results", False))

    # dataset ------------------------------------------------------------------
    ds = cfg["dataset"]
    logger.info("Loading dataset %s …", ds["name"])
    vecs, queries, ground = load_dataset(ds["name"], ds.get("path", ""))
    metric   = ds["metric"]
    nq       = ds["nq"]
    ground   = ground[:nq]
    queries  = queries[:nq]

    idx_map = {"Quake": QuakeWrapper, "IVF": FaissIVF}

    # build / run --------------------------------------------------------------
    if mode in {"build", "run"}:
        for icfg in cfg["indexes"]:
            name     = icfg["name"]
            res_dir  = out / name; res_dir.mkdir(exist_ok=True, parents=True)
            idx_bin  = res_dir / "index.bin"
            csv_file = res_dir / "results.csv"
            idx_cls  = idx_map[icfg["index"]]

            # BUILD once =======================================================
            idx, build_t = _prepare_index(
                idx_cls, vecs, metric, icfg["build_params"], idx_bin, rebuild_index
            )
            if mode == "build":
                continue  # build-only mode stops here

            # RUN for every recall_target ======================================
            if csv_file.exists() and not rewrite_csv:
                existing_df = pd.read_csv(csv_file)
            else:
                existing_df = pd.DataFrame()

            for sp in _expand_targets(icfg["search_params"].copy()):
                tgt = sp.get("recall_target", None)
                if tgt is None:
                    tgt = sp.get("nprobe")
                # if (
                #         not rewrite_csv
                #         and "recall_target" in existing_df.columns
                #         and ((existing_df.recall_target - tgt).abs() < 1e-9).any()
                # ):
                #     logger.info("[%s] target %.3g cached – skipping run.", name, tgt)
                #     continue

                stats = _benchmark_index(
                    idx           = idx,
                    queries       = queries,
                    ground        = ground,
                    search_params = sp,
                    trials        = cfg.get("trials", 5),
                    warmup        = cfg.get("warmup", 1),
                )
                stats.update(build_time_s = build_t, recall_target = tgt)
                existing_df = pd.concat(
                    [existing_df, pd.DataFrame([stats])], ignore_index=True
                )
                logger.info("[%s] target %.3g done.", name, tgt)

            existing_df.to_csv(csv_file, index=False)
            logger.info("[%s] results written to %s", name, csv_file)

    # plot ---------------------------------------------------------------------
    if mode in {"run", "plot"}:
        _plot_unified(cfg, out)

