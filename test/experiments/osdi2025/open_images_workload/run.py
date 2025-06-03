from __future__ import annotations

import json, logging, time, shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import quake
import test.experiments.osdi2025.experiment_utils as common_utils
from quake.utils import to_path, compute_recall
from quake.index_wrappers.quake import QuakeWrapper
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.index_wrappers.faiss_hnsw import FaissHNSW
try:    from quake.index_wrappers.scann import Scann
except ImportError: Scann = None          # type: ignore
try:    from quake.index_wrappers.diskann import DiskANNDynamic
except ImportError: DiskANNDynamic = None # type: ignore
try:    from quake.index_wrappers.vamana import Vamana
except ImportError: Vamana = None         # type: ignore

INDEX_CLASSES: Dict[str, Any] = {
    "Quake":   QuakeWrapper,
    "IVF":     FaissIVF,
    "HNSW":    FaissHNSW,
    "SCANN":   Scann,
    "DiskANN": DiskANNDynamic,
    "SVS":     Vamana,
}

log = logging.getLogger("openimages")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

WINDOW_SIZE = 2_000_000
K           = 100

STYLE = {
    "query":  dict(ls="-", marker="o", ms=4, lw=1.1, mfc="none"),
    "insert": dict(ls="-", marker="s", ms=4, lw=1.1, mfc="none"),
    "delete": dict(ls="-", marker="^", ms=4, lw=1.1, mfc="none"),
}

class OpenImagesEvaluator:
    def __init__(self, workload_dir: Union[str, Path], out_dir: Union[str, Path],
                 *, dataset_dir: Union[str, Path]):
        self.workload_dir = to_path(workload_dir)
        self.runbook_path = self.workload_dir / "runbook.json"
        self.ops_dir      = self.workload_dir / "operations"
        self.out_dir      = to_path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

        ds = to_path(dataset_dir)
        self._BASE    = torch.load(ds / "BASE.pt")
        self._IDS     = torch.load(ds / "IDS.pt")
        self._QUERIES = torch.load(ds / "QUERIES.pt")
        self._BASE    /= self._BASE.norm(dim=1, keepdim=True)
        self._QUERIES /= self._QUERIES.norm(dim=1, keepdim=True)

        self.current_start, self.current_end = 0, WINDOW_SIZE
        self.current_ids  = self._IDS[:WINDOW_SIZE]
        self.current_base = self._BASE[:WINDOW_SIZE]

        max_id = int(self._IDS.max())
        self.inv_map = torch.full((max_id + 1,), -1, dtype=torch.long)
        self.inv_map[self.current_ids] = torch.arange(len(self.current_ids))

    # ------------------------------------------------------------------ helpers
    def _init_index(self, name: str, wrapper, build_p: Dict[str, Any],
                    m_p: Optional[Dict[str, Any]]):
        idx_dir  = self.workload_dir / "init_indexes"
        idx_file = idx_dir / f"{name}.index"
        idx_dir.mkdir(parents=True, exist_ok=True)

        if not idx_file.exists():
            wrapper.build(self.current_base, ids=self.current_ids, **build_p)
            wrapper.save(idx_file)
        else:
            wrapper.load(idx_file,
                         num_workers=build_p.get("num_workers", 0),
                         use_numa   =build_p.get("use_numa", True),
                         parent     =build_p.get("parent"))

        if isinstance(wrapper, QuakeWrapper) and m_p:
            mp = quake.MaintenancePolicyParams()
            for k, v in m_p.items(): setattr(mp, k, v)
            wrapper.index.initialize_maintenance_policy(mp)
        return wrapper

    # ------------------------------------------------------------------ core
    def evaluate(self, *, name: str, wrapper, build_params: Dict[str, Any],
                 search_params: Dict[str, Any], do_maintenance: bool,
                 m_params: Optional[Dict[str, Any]], batch: bool,
                 max_q: int = 1000) -> None:

        wrapper = self._init_index(name, wrapper, build_params, m_params)
        runbook = json.load(open(self.runbook_path))
        totals  = dict(query=0.0, insert=0.0, delete=0.0, maintenance=0.0)
        rows: List[Dict[str, Any]] = []

        recall = None
        for key, op in runbook["operations"].items():
            op_id = int(key); typ = op["type"]; t0 = time.perf_counter()

            if typ == "delete":
                n = op["size"]; ids_del = self.current_ids[:n]
                wrapper.remove(ids_del)
                self.current_start += n
                self.current_ids  = self._IDS[self.current_start:self.current_end]
                self.current_base = self._BASE[self.current_start:self.current_end]
                self.inv_map[ids_del] = -1

            elif typ == "insert":
                n = op["size"]; old_end = self.current_end; self.current_end += n
                vecs = self._BASE[old_end:self.current_end]; ids = self._IDS[old_end:self.current_end]
                wrapper.add(vecs, ids=ids)
                self.current_ids  = self._IDS[self.current_start:self.current_end]
                self.current_base = self._BASE[self.current_start:self.current_end]
                if int(ids.max()) >= self.inv_map.size(0):
                    bigger = torch.full((int(ids.max()) + 1,), -1, dtype=torch.long)
                    bigger[:len(self.inv_map)] = self.inv_map; self.inv_map = bigger
                self.inv_map[ids] = torch.arange(len(self.current_ids)-n, len(self.current_ids))

            elif typ == "query":
                q_idx = torch.load(self.ops_dir / f"{op_id}.pt")
                queries = self._QUERIES[q_idx]
                if len(queries) > max_q:
                    perm = torch.randperm(len(queries))[:max_q]
                    queries, q_idx = queries[perm], q_idx[perm]
                if batch:
                    pred_ids = wrapper.search(queries, **search_params).ids
                else:
                    parts = [wrapper.search(q.unsqueeze(0), **search_params).ids for q in queries]
                    pred_ids = torch.cat(parts)
                gt_ids = torch.load(self.ops_dir / f"{op_id}_gt_ids.pt")[:len(pred_ids)]
                recall = compute_recall(pred_ids, gt_ids, search_params.get("k", K)).mean().item()
                op["recall"] = recall
            else:
                raise ValueError(typ)

            lat_ms  = 1e3 * (time.perf_counter() - t0)
            maint_ms = spl = dele = spl_ms = del_ms = ref_ms = 0.0
            if do_maintenance:
                t1 = time.perf_counter(); info = wrapper.maintenance()
                maint_ms = 1e3 * (time.perf_counter() - t1)
                spl, dele = info.n_splits, info.n_deletes
                spl_ms, del_ms = info.split_time_us/1000.0, info.delete_time_us/1000.0
                ref_ms = info.refinement_time_us/1000.0

            totals[typ] += lat_ms; totals["maintenance"] += maint_ms
            state = wrapper.index_state()

            row = dict(
                operation_number      = op_id,
                operation_type        = typ,
                latency_ms            = lat_ms,
                maintenance_latency_ms= maint_ms,
                n_resident            = state.get("n_total", len(self.current_ids)),
                nlist                 = state.get("n_list", np.nan),
                n_splits              = spl,
                n_deletes             = dele,
                split_time_ms         = spl_ms,
                delete_time_ms        = del_ms,
                refinement_time_ms    = ref_ms,
                recall                = recall,
            )
            print(row)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / "results.csv", index=False)
        self._plots(df, totals, name)

    # ------------------------------------------------------------------ plots
    def _plots(self, df: pd.DataFrame, tot: Dict[str,float], title: str):
        fig, axs = plt.subplots(2,2, figsize=(12,9))
        for t in ("query","insert","delete"):
            sub = df[df.operation_type==t]
            if not sub.empty:
                axs[0,0].plot(sub.operation_number, sub.latency_ms, label=t, **STYLE[t])
        axs[0,0].set_title("Latency"); axs[0,0].legend(); axs[0,0].grid(ls=":")
        if "nlist" in df:
            axs[0,1].plot(df.operation_number, df.nlist, marker="o", lw=1.1)
            axs[0,1].set_title("#Partitions"); axs[0,1].grid(ls=":")
        axs[1,0].plot(df.operation_number, df.n_resident, marker="o", lw=1.1)
        axs[1,0].set_title("Resident"); axs[1,0].grid(ls=":")
        rec = df[(df.operation_type=="query") & df.recall.notna()]
        if not rec.empty:
            axs[1,1].plot(rec.operation_number, rec.recall, marker="o", lw=1.1)
            axs[1,1].set_title("Recall"); axs[1,1].set_ylim(0,1); axs[1,1].grid(ls=":")
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(self.out_dir / "four_panel.png", dpi=150); plt.close(fig)

        lbl = ["Query","Insert","Delete","Maintenance","Total"]
        val = [tot.get("query",0), tot.get("insert",0),
               tot.get("delete",0), tot.get("maintenance",0)]
        val.append(sum(val))
        plt.bar(lbl, val); plt.title(f"Time breakdown – {title}")
        plt.ylabel("ms"); plt.tight_layout()
        plt.savefig(self.out_dir / "time_breakdown.png", dpi=150); plt.close()

# ──────────────────────────────────────────────────────────────────────────
def run_experiment(cfg_path_str: Dict[str, Any], output_root: Union[str, Path]) -> None:
    cfg = common_utils.load_config(cfg_path_str)
    wdir  = Path(cfg["workload_dir"]).expanduser()
    dset  = Path(cfg["dataset_dir"]).expanduser()
    out   = Path(output_root).expanduser()
    batch = cfg.get("batch", False)
    out.mkdir(parents=True, exist_ok=True)

    for idx in cfg["indexes"]:
        name   = idx["name"]; wrapper_key = idx["index"]
        out_dir = out / name

        if out_dir.exists() and not cfg.get("overwrite", False):
            log.info("Skip %s – results exist (overwrite=False)", name); continue
        if out_dir.exists() and cfg.get("overwrite", False):
            shutil.rmtree(out_dir)

        cls = INDEX_CLASSES.get(wrapper_key)
        if cls is None:
            log.warning("Unknown wrapper %s – skip %s", wrapper_key, name); continue

        ev = OpenImagesEvaluator(wdir, out_dir, dataset_dir=dset)
        ev.evaluate(
            name   = name,
            wrapper= cls(),
            build_params = idx.get("build_params", {}),
            search_params= idx.get("search_params", {}),
            do_maintenance = idx.get("maintenance_params") is not None,
            m_params       = idx.get("maintenance_params"),
            batch          = batch,
        )
