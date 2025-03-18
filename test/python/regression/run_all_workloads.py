#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path
import argparse

def run_workload(config: Path, results_dir: Path, run_name: str) -> None:
    """Run the regression test for a given config and store its results."""
    print(f"Running workload for config: {config.name} (run: {run_name})")
    # Create a dedicated subdirectory for this workload's results using the run name.
    output_dir = results_dir / config.stem / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute the regression test script with the given configuration and run name.
    # This assumes run_regression_test.py has been modified to accept a --name parameter.
    result = subprocess.run([
        "python", "run_regression_test.py",
        "--config", str(config),
        "--output", str(output_dir),
        "--name", run_name
    ])
    if result.returncode != 0:
        sys.exit(f"Error running workload: {config}")

def main():
    parser = argparse.ArgumentParser(description="Run all workload configurations with a given run name.")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of this run (e.g. 'baseline' or 'PR-123'). This will be appended to the results directory for each config.")
    args = parser.parse_args()

    # List your workload configuration files.
    configs = [
        Path("configs/sift1m_balanced.yaml"),
        Path("configs/sift1m_insert_heavy.yaml"),
        Path("configs/sift1m_read_only.yaml")
    ]

    # Base directory to store results for all workloads.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # Run each workload configuration.
    for config in configs:
        run_workload(config, results_dir, args.name)

    # Once all workloads are complete, call the comparison script.
    compare_cmd = [
        "python", "compare_results.py",
        "--results_dir", str(results_dir),
        "--plot_type", "aggregate",  # or 'both' if you want detailed as well
        "--output_aggregate", str(results_dir / f"aggregate_matrix_{args.name}.png")
    ]
    result = subprocess.run(compare_cmd)
    if result.returncode != 0:
        sys.exit("Error generating comparison plots.")
    else:
        print("All workloads completed and comparison plots generated.")

if __name__ == "__main__":
    main()