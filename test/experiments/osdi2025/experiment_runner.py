#!/usr/bin/env python3
"""
Master runner for OSDI2025 vector‑search experiments.
"""
import argparse
from pathlib import Path

from test.experiments.osdi2025.kick_the_tires.run import run_experiment as run_kick_the_tires
from test.experiments.osdi2025.numa_single_query.run import run_experiment as run_numa_single
from test.experiments.osdi2025.numa_multi_query.run  import run_experiment as run_numa_multi
from test.experiments.osdi2025.aps_recall_targets.run import run_experiment as run_aps_recall
from test.experiments.osdi2025.early_termination.run import run_experiment as run_early_termination
from test.experiments.osdi2025.multi_level.run import run_experiment as run_multi_level

EXPERIMENTS = {
    "kick_the_tires":   run_kick_the_tires,
    "numa_single_query":  run_numa_single,
    "numa_multi_query":   run_numa_multi,
    "aps_recall_targets": run_aps_recall,
    "early_termination":   run_early_termination,
    "multi_level":        run_multi_level,
}

def main():
    parser = argparse.ArgumentParser(description="Run one OSDI2025 experiment")
    parser.add_argument("-x","--experiment", choices=EXPERIMENTS.keys(), required=True)
    parser.add_argument("-c","--config", required=True,
                        help="Config name (without .yaml) under configs/")
    parser.add_argument("-o","--output-dir", default=None,
                        help="Where to write results (csv + plots)")
    args = parser.parse_args()

    base     = Path(__file__).resolve().parent
    cfg_path = base / args.experiment / "configs" / f"{args.config}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    out_dir = Path(args.output_dir) if args.output_dir \
        else base / args.experiment / "results" / args.config
    out_dir.mkdir(parents=True, exist_ok=True)

    # dispatch
    EXPERIMENTS[args.experiment](str(cfg_path), str(out_dir))

if __name__ == "__main__":
    main()