#!/usr/bin/env python3
"""
metrics_trace.py – metrics tracing, logging, and optional plotting/saving.
Direct, precise, minimal; consolidated output directory.
"""

import argparse, subprocess, os, signal, time, json, logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, TextIO, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── Configuration ──────────────────────────
REFRESH_SEC     = 1
HISTORY_SECONDS = 3600
PANELS = [
    ("CPU %", ["cpu_pct"]),
    ("Memory %", ["mem_pct"]),
    ("Disk MB/s", ["disk_read_mb", "disk_write_mb"]),
    ("GPU %", ["gpu_util"]),
]
_MB = 1_048_576.0

# ────────────────────── Metric Extraction ──────────────────────────
def extract(j: Dict[str, Any]) -> Dict[str, float]:
    cpu = j.get("cpu", {})
    mem = j.get("mem", {})
    # Disk IO
    diskio = j.get("diskio", [])
    read = write = 0.0
    if isinstance(diskio, list):
        for d in diskio:
            if not isinstance(d, dict): continue
            read  += d.get("rate/read_bytes", d.get("read_bytes",  0))
            write += d.get("rate/write_bytes", d.get("write_bytes", 0))
    elif isinstance(diskio, dict):
        vals = diskio.values()
        if any(isinstance(v, dict) for v in vals):
            for v in vals:
                if not isinstance(v, dict): continue
                read  += v.get("rate/read_bytes", v.get("read_bytes",  0))
                write += v.get("rate/write_bytes", v.get("write_bytes", 0))
        else:
            for k, v in diskio.items():
                if k.endswith(".read_bytes") or k.endswith(".read_bytes_rate_per_sec"):  read  += v
                if k.endswith(".write_bytes") or k.endswith(".write_bytes_rate_per_sec"): write += v
    read  /= _MB; write /= _MB
    # GPU utilization
    gpu_src = j.get("gpu") or {}
    gpu_util = np.nan
    if isinstance(gpu_src, list) and gpu_src and isinstance(gpu_src[0], dict):
        gpu_util = gpu_src[0].get("proc") or gpu_src[0].get("utilization") or np.nan
    elif isinstance(gpu_src, dict):
        nested = [v for v in gpu_src.values() if isinstance(v, dict)]
        if nested:
            gpu_util = nested[0].get("proc") or nested[0].get("utilization") or np.nan
        else:
            for k, v in gpu_src.items():
                if k.endswith(".proc") or k.endswith(".gpu_proc"): gpu_util = v; break
    return {
        "cpu_pct":       cpu.get("user", 0) + cpu.get("system", 0),
        "mem_pct":       mem.get("percent", np.nan),
        "disk_read_mb":  read,
        "disk_write_mb": write,
        "gpu_util":      gpu_util,
    }

# ───────────────────────── Utilities ──────────────────────────────
def start_glances(tmp_json: Path) -> subprocess.Popen:
    cmd = [
        "glances", "-q", f"-t{REFRESH_SEC}",
        "--export", "json", "--export-json-file", str(tmp_json),
        "--disable-plugin", "all", "--enable-plugin", "cpu,mem,diskio,gpu",
    ]
    logging.info("Starting Glances: %s", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         text=True, preexec_fn=os.setsid)
    time.sleep(1.5)
    if p.poll() is not None:
        raise RuntimeError(p.stderr.read())
    return p


def stop_glances(p: subprocess.Popen | None):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(3)
        except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL)


def read_all(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text: return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        out = []
        for line in text.splitlines():
            try: out.append(json.loads(line))
            except: pass
        return out


def parse_timestamp(r: dict) -> datetime:
    raw = r.get("timestamp") or r.get("now") or r.get("ts")
    if isinstance(raw, (int, float)):
        return pd.to_datetime(raw, unit="s", utc=True).to_pydatetime()
    dt = pd.to_datetime(raw, utc=True, errors="coerce")
    return dt.to_pydatetime() if not pd.isna(dt) else datetime.now(timezone.utc)


def new_fig(n: int):
    cols = 2 if n>2 else 1; rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3*rows), sharex=True)
    return fig, axes.flatten() if n>1 else [axes]


def redraw(fig, axes, df: pd.DataFrame):
    for ax in axes: ax.clear()
    for ax, (title, cols) in zip(axes, PANELS):
        if set(cols).issubset(df.columns):
            df[cols].plot(ax=ax, lw=1.2)
            ax.set_title(title, fontsize=9); ax.grid(True, ls="--", alpha=0.5)
    fig.tight_layout(); fig.canvas.draw_idle()

# ───────────────────────── Loops ──────────────────────────────────
def live_loop_plot(tmp_json: Path, log_fp: TextIO, history: Deque[dict], out_dir: Path):
    fig, axes = new_fig(len(PANELS)); plt.ion(); fig.show()
    last_ts = None
    try:
        while True:
            recs = read_all(tmp_json); updated=False
            for r in recs:
                ts = parse_timestamp(r)
                if last_ts is None or ts>last_ts:
                    row={"ts":ts}|extract(r)
                    history.append(row); last_ts=ts; updated=True
                    log_fp.write(json.dumps({"ts":str(ts),**extract(r)})+"\n"); log_fp.flush()
            if updated:
                df=pd.DataFrame(history).set_index("ts"); redraw(fig,axes,df)
                if out_dir:
                    fig.savefig(out_dir/"live.png")
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff(); plt.close('all')


def live_loop_log(tmp_json: Path, log_fp: TextIO, history: Deque[dict]):
    try:
        while True:
            for r in read_all(tmp_json):
                ts=parse_timestamp(r)
                row={"ts":ts}|extract(r)
                history.append(row)
                log_fp.write(json.dumps({"ts":str(ts),**extract(r)})+"\n"); log_fp.flush()
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        pass


def load_static(logfile: Path, history: Deque[dict], out_dir: Path, plot: bool):
    for r in read_all(logfile):
        ts=parse_timestamp(r)
        if not history or ts>history[-1]["ts"]:
            if "cpu_pct" in r and "disk_read_mb" in r:
                row={"ts":ts}
                row.update({k:r.get(k,0) for _,cols in PANELS for k in cols})
            else:
                row={"ts":ts}|extract(r)
            history.append(row)
    if plot:
        df=pd.DataFrame(history).set_index("ts")
        fig,axes=new_fig(len(PANELS)); redraw(fig,axes,df)
        fig.savefig(out_dir/"static.png"); plt.show()

# ───────────────────────── Main ────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description="Metrics tracing with consolidated out-dir")
    ap.add_argument("-o","--out-dir", default="outputs", type=Path,
                    help="Directory for logs, glances JSON, and plots")
    ap.add_argument("--live", action="store_true", help="Run live logging")
    ap.add_argument("--plot", action="store_true", help="Enable plotting")
    ap.add_argument("--debug",action="store_true",help="Verbose logging")
    args=ap.parse_args()

    lvl=logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=lvl,format="%(asctime)s %(levelname)s %(message)s",datefmt="%H:%M:%S")

    out_dir=args.out_dir.expanduser()
    out_dir.mkdir(parents=True,exist_ok=True)
    tmp_json=out_dir/"system_metrics.tmp.json"
    logfile=out_dir/"metrics.log.ndjson"

    history:Deque[dict]=deque(maxlen=HISTORY_SECONDS//REFRESH_SEC)
    proc=None
    try:
        if args.live:
            proc=start_glances(tmp_json)
            with logfile.open("a",encoding="utf-8") as log_fp:
                if args.plot:
                    live_loop_plot(tmp_json,log_fp,history,out_dir)
                else:
                    live_loop_log(tmp_json,log_fp,history)
        else:
            load_static(logfile, history, out_dir, args.plot)
    finally:
        stop_glances(proc)

if __name__=="__main__": main()
