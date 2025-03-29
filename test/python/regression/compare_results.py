#!/usr/bin/env python
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_results(results_dir: Path) -> pd.DataFrame:
    """
    Recursively load all CSV files named "results.csv" from subdirectories of results_dir.
    Each CSV is assumed to come from one configuration (workload) run.
    """
    csv_files = list(results_dir.rglob("results.csv"))
    if not csv_files:
        raise ValueError(f"No CSV results found in {results_dir}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Assume the parent directory's name is the maintenance configuration name.
        df["maintenance_config"] = csv_file.parent.name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    # If the 'method' column is missing, use maintenance_config as the method.
    if "method" not in combined_df.columns:
        combined_df["method"] = combined_df["maintenance_config"]
    logging.info(f"Loaded {len(dfs)} CSV files with a total of {combined_df.shape[0]} rows.")
    return combined_df


def compare_metrics(
    df: pd.DataFrame, thresholds: Dict[str, float]
) -> Tuple[Dict[str, Dict[str, pd.Series]], List[Tuple[str, str, str, float, float, float]]]:
    """
    Compute average metrics per maintenance configuration and flag regression failures.
    For latency (lower is better) and recall (higher is better), compare averages against thresholds.
    """
    comparisons = {}
    regression_failures = []
    for config, subdf in df.groupby("maintenance_config"):
        metrics_to_compare = [metric for metric in thresholds.keys() if metric in subdf.columns]
        if not metrics_to_compare:
            continue
        avg_by_config = subdf.groupby("maintenance_config")[metrics_to_compare].mean(numeric_only=True)
        comparisons[config] = {}
        for metric, threshold in thresholds.items():
            if metric not in avg_by_config.columns:
                continue
            series = avg_by_config[metric]
            comparisons[config][metric] = series
            if metric == "latency_ms":
                best = series.min()
                for method, value in series.items():
                    diff = (value - best) / best
                    if diff > threshold:
                        regression_failures.append((metric, config, method, best, value, diff))
            elif metric == "recall":
                best = series.max()
                for method, value in series.items():
                    diff = (best - value) / best
                    if diff > threshold:
                        regression_failures.append((metric, config, method, best, value, diff))
    return comparisons, regression_failures


def plot_aggregate_matrix(df: pd.DataFrame, metrics: List[str], output_path: Path) -> None:
    """
    Create an aggregate matrix plot showing the average value for each metric (row) p
    er maintenance configuration (column).
    """
    # Group by maintenance_config and compute the mean of each metric.
    agg_data = {}
    for metric in metrics:
        agg_data[metric] = df.groupby("maintenance_config")[metric].mean()
    agg_df = pd.DataFrame(agg_data).T  # rows are metrics, columns are maintenance configurations

    fig, ax = plt.subplots(figsize=(8, 4))
    cax = ax.imshow(agg_df.values, cmap="viridis", aspect="auto")
    # Set x-axis labels (maintenance configurations)
    ax.set_xticks(np.arange(agg_df.shape[1]))
    ax.set_xticklabels(agg_df.columns, rotation=45, ha="right")
    # Set y-axis labels (metrics)
    ax.set_yticks(np.arange(agg_df.shape[0]))
    ax.set_yticklabels(agg_df.index)
    ax.set_title("Aggregate Metrics per Workload")
    # Annotate each cell with the value.
    for i in range(agg_df.shape[0]):
        for j in range(agg_df.shape[1]):
            ax.text(j, i, f"{agg_df.iloc[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Aggregate matrix plot saved to {output_path}")
    plt.show()


def plot_joint_detailed_per_operation_all(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a joint detailed per-operation plot that shows multiple maintenance configurations
    (methods) on the same axes. Four subplots are created for key metrics:
      - Query latency (ms)
      - Query recall
      - Resident set size (n_resident)
      - Number of partitions (n_list)
    For query-specific metrics (latency and recall), only operations of type 'query' are used.
    """
    # Filter query operations for latency and recall.
    query_df = df[df["operation_type"] == "query"]
    configs = df["maintenance_config"].unique()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Define the metrics to plot along with labels and their source dataframe.
    metrics = [
        ("latency_ms", "Query Latency (ms)", query_df),
        ("recall", "Query Recall", query_df),
        ("n_resident", "Resident Set Size", df),
        ("n_list", "Number of Partitions", df),
    ]

    for ax, (metric, ylabel, data_source) in zip(axs, metrics):
        for config in configs:
            data = data_source[data_source["maintenance_config"] == config]
            data = data.sort_values("operation_number")
            if not data.empty:
                ax.plot(data["operation_number"], data[metric], marker="o", label=config)
        ax.set_xlabel("Operation Number")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} per Operation")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Joint detailed per-operation plot saved to {output_path}")
    plt.show()


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Compare regression test results and produce aggregate and joint detailed plots."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result CSV files (searched recursively).",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        default="both",
        choices=["aggregate", "detailed", "both"],
        help="Type of plot to generate: 'aggregate', 'detailed' (joint), or 'both'.",
    )
    parser.add_argument(
        "--output_aggregate",
        type=Path,
        default=Path("aggregate_matrix.png"),
        help="Output path for the aggregate matrix plot.",
    )
    parser.add_argument(
        "--detailed_output",
        type=Path,
        default=Path("detailed_joint.png"),
        help="Output path for the joint detailed per-operation plot.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default='{"latency_ms": 0.05, "recall": 0.01}',
        help="JSON string of thresholds for regression checks.",
    )
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds)
    df = load_results(args.results_dir)

    comparisons, failures = compare_metrics(df, thresholds)
    if failures:
        logging.error("Regression failures detected:")
        for metric, config, method, best, current, diff in failures:
            logging.error(
                f"Workload '{config}' method '{method}' for metric '{metric}': best = {best:.2f}, "
                f"current = {current:.2f}, diff = {diff*100:.1f}%"
            )
    else:
        logging.info("No inter-method regressions detected within workloads.")

    if args.plot_type in ["aggregate", "both"]:
        # Here the aggregate matrix shows workloads (maintenance_config) on the x-axis
        # and each metric (e.g., latency and recall) as rows.
        plot_aggregate_matrix(df, metrics=["latency_ms", "recall"], output_path=args.output_aggregate)
    if args.plot_type in ["detailed", "both"]:
        plot_joint_detailed_per_operation_all(df, output_path=args.detailed_output)


if __name__ == "__main__":
    main()
