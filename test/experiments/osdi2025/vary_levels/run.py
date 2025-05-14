#!/usr/bin/env python3
"""
experiment_levels.py – benchmark a single Quake/IVF index with optional
parent levels, each level having its own search parameters.

* Build the **child** index once (saved as base_index.bin).
* For every level described in `levels:`:
    – Reload the base index from disk.
    – Optionally add a parent layer with `nc` centroids.
    – Run searches using the per-level overrides.
* Results for each level go to ./OUT_DIR/L<nc>/results.csv
* A combined QPS-vs-Recall plot is written to OUT_DIR/recall_vs_qps.png
"""

from __future__ import annotations
import argparse, logging, time
from pathlib import Path
from typing  import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quake
import torch, yaml

from quake.datasets.ann_datasets   import load_dataset
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.quake     import QuakeWrapper
from quake.utils                   import compute_recall

logger = logging.getLogger("experiment")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ───────────────────── helper routines ─────────────────────
def _prepare_index(idx_cls, vecs, metric, build_params,
                   path: Path, rebuild: bool) -> Tuple[object, float]:
    """Build or load the *child* index; return (index, build_time_s)."""
    idx = idx_cls()
    if path.exists() and not rebuild:
        t0 = time.time()
        idx.load(str(path), num_workers=build_params["num_workers"])
        return idx, 0.0
    logger.info("Building index → %s …", path)
    t0 = time.time()
    idx.build(vecs, metric=metric, **build_params)
    t_build = time.time() - t0
    idx.save(str(path))
    logger.info("Index saved (%.1fs).", t_build)
    return idx, t_build


def _benchmark_index(idx, queries, ground, search_params,
                     trials: int, warmup: int) -> Dict:
    """Run `trials` timed searches; return stats dict."""
    sp = search_params.copy()              # don’t modify caller’s dict
    bs = sp.pop("batch_size", None)
    k  = sp["k"]

    # warm-up
    for _ in range(warmup):
        if bs is None:
            idx.search(queries, **sp)
        else:
            for i in range(0, len(queries), bs):
                idx.search(queries[i:i+bs], **sp)

    qps, lat_ms = [], []
    for _ in range(trials):
        t0 = time.time()
        if bs is None:
            I = idx.search(queries, **sp).ids
        else:
            I = np.concatenate([
                idx.search(queries[i:i+bs], **sp).ids
                for i in range(0, len(queries), bs)
            ])
        dt = time.time() - t0
        qps.append(len(queries) / dt)
        lat_ms.append(dt * 1e3)

    recall = compute_recall(I, ground[:, :k], k)
    recall = 0.0 if recall is None else recall.mean().item()

    return dict(
        recall          = float(recall),
        qps_mean        = float(np.mean(qps)),
        qps_std         = float(np.std(qps)),
        latency_ms      = float(np.mean(lat_ms)),
        trials          = trials,
        warmup          = warmup,
        batch_size      = bs or len(queries),
    )


def _expand_targets(sp: Dict) -> List[Dict]:
    """Yield one search-param dict per recall_target OR per nprobe value."""
    if "recall_targets" in sp:
        tgs = sp.pop("recall_targets")
        if not isinstance(tgs, (list, tuple)): tgs = [tgs]
        return [{**sp, "recall_target": t} for t in tgs]
    if "nprobes" in sp:
        tgs = sp.pop("nprobes")
        if not isinstance(tgs, (list, tuple)): tgs = [tgs]
        return [{**sp, "nprobe": t} for t in tgs]
    return [sp]


def _plot(cfg: Dict, root: Path) -> None:
    """Generate combined QPS-vs-Recall plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    styles  = cfg.get("plot", {}).get("styles", {})
    for lv in cfg["levels"]:
        nc   = lv["nc"]
        name = f"L{nc}"
        csv  = root / name / "results.csv"
        if not csv.exists():
            logger.warning("No results for level %s – skipped.", nc)
            continue
        df  = pd.read_csv(csv).sort_values("recall")
        sty = styles.get(name, {})
        ax.plot(
            df["qps_mean"], df["recall"],
            label      = f"{nc:,} centroids" if nc else "single-level",
            marker     = sty.get("marker", "o"),
            linestyle  = sty.get("linestyle", "-"),
            linewidth  = 1.4,
            markersize = 6,
            color      = sty.get("color", None),
        )
    # ax.set_xscale("log", base=10)
    ax.set_xlabel("Queries per second (log)")
    ax.set_ylabel("Recall@k")
    ax.set_title("QPS vs Recall for different parent levels")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "recall_vs_qps.png", dpi=180)
    logger.info("Plot written to %s", root / "recall_vs_qps.png")

# ─────────────────────────── main ──────────────────────────
def run_experiment(cfg_path: str, output_dir: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # dataset ---------------------------------------------------------------
    ds = cfg["dataset"]
    logger.info("Loading dataset %s …", ds["name"])
    vecs, queries, ground = load_dataset(ds["name"], ds.get("path", ""))
    nq = ds["nq"]
    queries, ground = queries[:nq], ground[:nq]
    metric = ds["metric"]

    # build child index once -----------------------------------------------
    icfg      = cfg["index"]
    idx_cls   = {"Quake": QuakeWrapper, "IVF": FaissIVF}[icfg["type"]]
    base_bin  = out / "base_index.bin"
    base_idx, build_t = _prepare_index(
        idx_cls, vecs, metric,
        icfg["build_params"],
        base_bin,
        rebuild=cfg["overwrite"].get("index", False),
    )

    # iterate over parent levels -------------------------------------------
    for lv in cfg["levels"]:
        nc       = lv["nc"]
        name     = f"L{nc}"
        res_dir  = out / name
        res_dir.mkdir(parents=True, exist_ok=True)
        csv_file = res_dir / "results.csv"

        if csv_file.exists() and not cfg["overwrite"].get("results", False):
            logger.info("[%s] results cached – skipping.", name)
            continue

        # fresh load of the base index for this level
        idx = idx_cls()
        idx.load(str(base_bin), num_workers=icfg["build_params"].get("num_workers", 1))
        if nc:  # add parent layer when nc > 0
            build_params = quake.IndexBuildParams()
            build_params.nlist = nc
            idx.index.add_level(build_params)

        # child search params (base defaults + per-level overrides)
        child_sp = icfg["search_params"].copy()
        child_sp.update(lv.get("search_params", {}))

        # optional parent search params
        parent_sp = lv.get("parent_search_params", None)

        df = pd.DataFrame()
        for sp in _expand_targets(child_sp.copy()):
            if parent_sp:               # attach parent section if any
                sp["parent"] = parent_sp
            stats = _benchmark_index(
                idx, queries, ground, sp,
                trials = cfg.get("trials", 5),
                warmup = cfg.get("warmup", 1),
            )
            stats.update(
                build_time_s = build_t if nc == 0 else 0.0,
                parent_nc    = nc,
                recall_target= sp.get("recall_target", sp.get("nprobe", np.nan)),
            )
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
            logger.info("[%s] target %.3g done.", name, stats["recall_target"])

        df.to_csv(csv_file, index=False)
        logger.info("[%s] results written to %s", name, csv_file)

    _plot(cfg, out)
