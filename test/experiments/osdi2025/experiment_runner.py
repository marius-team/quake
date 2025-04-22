#!/usr/bin/env python
"""
Master runner for OSDI2025 vector‐search experiments.

Usage:
  python -m test.experiments.osdi2025.experiment_runner \
      --experiment <name> \
      --config <cfg_name> \
      --output-dir <out_dir>
"""

import argparse
from pathlib import Path

# Make sure `test/`, `test/experiments/`, and subfolders all have an __init__.py
from test.experiments.osdi2025.numa_single_query.run import run_experiment as run_numa_single
from test.experiments.osdi2025.numa_multi_query.run  import run_experiment as run_numa_multi

EXPERIMENTS = {
    "numa_single_query": run_numa_single,
    "numa_multi_query":  run_numa_multi,
}

def main():
    parser = argparse.ArgumentParser(
        description="Run one OSDI2025 experiment"
    )
    parser.add_argument(
        "-x", "--experiment",
        choices=EXPERIMENTS.keys(),
        required=True,
        help="Which experiment to run"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Config name (without .yaml) under configs/"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Where to write results (csv + plots)"
    )
    args = parser.parse_args()

    exp      = args.experiment
    cfg_name = args.config
    base     = Path(__file__).resolve().parent
    cfg_path = base / exp / "configs" / f"{cfg_name}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    # compute output directory
    out_dir = Path(args.output_dir) if args.output_dir else base / exp / "results" / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # call the appropriate experiment
    runner = EXPERIMENTS[exp]
    runner(str(cfg_path), str(out_dir))

if __name__ == "__main__":
    main()