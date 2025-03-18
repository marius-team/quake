#!/usr/bin/env python
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Assume the parent directory name is the workload name.
        config_name = csv_file.parent.name
        df['config'] = config_name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    # Ensure a 'method' column exists. If not, default to a single method.
    if 'method' not in combined_df.columns:
        combined_df['method'] = 'Quake'

    logging.info(f"Loaded {len(dfs)} CSV files with a total of {combined_df.shape[0]} rows.")
    return combined_df

def compare_metrics(df: pd.DataFrame, thresholds: Dict[str, float]
                    ) -> Tuple[Dict[str, Dict[str, pd.Series]], List[Tuple[str, str, str, float, float, float]]]:
    """
    For each workload (config), compute average metrics per method and compare them.
    Regression failures are only flagged if more than one method exists for that workload.
    Returns:
      - A dictionary mapping workload -> (metric -> series indexed by method)
      - A list of regression failures in the form
         (metric, config, method, best, current, diff)
    """
    comparisons = {}
    regression_failures = []

    # Process each workload separately.
    for config, subdf in df.groupby('config'):
        # Select only the metrics we care about (those specified in thresholds)
        metrics_to_compare = [metric for metric in thresholds.keys() if metric in subdf.columns]
        if not metrics_to_compare:
            continue
        # Compute means only for the specified numeric columns.
        avg_by_method = subdf.groupby('method')[metrics_to_compare].mean(numeric_only=True)
        comparisons[config] = {}

        if len(avg_by_method) > 1:
            for metric, threshold in thresholds.items():
                if metric not in avg_by_method.columns:
                    continue
                series = avg_by_method[metric]
                comparisons[config][metric] = series
                if metric == 'latency_ms':
                    best = series.min()
                    for method, value in series.items():
                        diff = (value - best) / best
                        if diff > threshold:
                            regression_failures.append((metric, config, method, best, value, diff))
                elif metric == 'recall':
                    best = series.max()
                    for method, value in series.items():
                        diff = (best - value) / best
                        if diff > threshold:
                            regression_failures.append((metric, config, method, best, value, diff))
        else:
            # With only one method, store the series for the aggregate plot.
            for metric in metrics_to_compare:
                comparisons[config][metric] = avg_by_method[metric]
    return comparisons, regression_failures

def plot_aggregate_matrix(df: pd.DataFrame, metrics: List[str], output_path: Path) -> None:
    """
    For each metric, pivot the results to a matrix with rows = method and columns = workload (config),
    then display as a heatmap.
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot_table(values=metric, index='method', columns='config', aggfunc=np.mean)
        cax = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f'Average {metric}')
        # Annotate with values.
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="w")
        fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Aggregate matrix plot saved to {output_path}")

def plot_detailed_per_operation_all(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a detailed per-operation plot for each workload (config).
    For each workload, plot four panels:
      - Operation Latency (ms) with separate lines for query, insert, and delete operations.
      - Query Recall (for query operations)
      - Resident Set Size (if available)
      - Number of Partitions (if available)
    Saves one plot per workload in output_dir.
    """
    workloads = df['config'].unique()
    for workload in workloads:
        df_workload = df[df['config'] == workload]
        methods = df_workload['method'].unique()  # still available for legends in other panels

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Panel A: Operation Latency by Operation Type
        ax = axs[0, 0]
        op_types = df_workload['operation_type'].unique()
        for op_type in op_types:
            subset = df_workload[df_workload['operation_type'] == op_type]
            if 'operation_number' in subset.columns and 'latency_ms' in subset.columns:
                ax.plot(subset['operation_number'], subset['latency_ms'],
                        marker='o', linestyle='-', label=op_type)
        ax.set_title('Operation Latency (ms) by Operation Type')
        ax.set_xlabel('Operation Number')
        ax.set_ylabel('Latency (ms)')
        ax.legend()

        # Panel B: Query Recall (for query operations)
        ax = axs[0, 1]
        for m in methods:
            subset = df_workload[(df_workload['method'] == m) & (df_workload['operation_type'] == 'query')]
            if not subset.empty and 'operation_number' in subset.columns and 'recall' in subset.columns:
                ax.plot(subset['operation_number'], subset['recall'],
                        marker='o', linestyle='-', label=m)
        ax.set_title('Query Recall')
        ax.set_xlabel('Operation Number')
        ax.set_ylabel('Recall')
        ax.legend()

        # Panel C: Resident Set Size (if available)
        ax = axs[1, 0]
        for m in methods:
            subset = df_workload[df_workload['method'] == m]
            if 'operation_number' in subset.columns and 'n_resident' in subset.columns:
                ax.plot(subset['operation_number'], subset['n_resident'],
                        marker='o', linestyle='-', label=m)
        ax.set_title('Resident Set Size')
        ax.set_xlabel('Operation Number')
        ax.set_ylabel('Resident Vectors')
        ax.legend()

        # Panel D: Number of Partitions (if available)
        ax = axs[1, 1]
        for m in methods:
            subset = df_workload[df_workload['method'] == m]
            if 'operation_number' in subset.columns and 'n_list' in subset.columns:
                ax.plot(subset['operation_number'], subset['n_list'],
                        marker='o', linestyle='-', label=m)
        ax.set_title('Number of Partitions')
        ax.set_xlabel('Operation Number')
        ax.set_ylabel('Partitions')
        ax.legend()

        plt.tight_layout()
        plot_path = output_dir / f"detailed_{workload}.png"
        plt.savefig(plot_path)
        logging.info(f"Detailed per-operation plot for workload '{workload}' saved to {plot_path}")
        plt.close(fig)

def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Compare regression test results and produce aggregate and detailed plots."
    )
    parser.add_argument(
        '--results_dir', type=Path, default=Path('results'),
        help="Directory containing result CSV files (searched recursively)."
    )
    parser.add_argument(
        '--plot_type', type=str, default='both', choices=['aggregate', 'detailed', 'both'],
        help="Type of plot to generate: 'aggregate', 'detailed', or 'both'."
    )
    parser.add_argument(
        '--output_aggregate', type=Path, default=Path('results/aggregate_matrix.png'),
        help="Output path for the aggregate matrix plot."
    )
    parser.add_argument(
        '--detailed_output_dir', type=Path, default=Path('results/detailed'),
        help="Output directory for detailed per-operation plots."
    )
    parser.add_argument(
        '--thresholds', type=str, default='{"latency_ms": 0.05, "recall": 0.01}',
        help="JSON string of thresholds for regression checks."
    )
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds)
    df = load_results(args.results_dir)

    comparisons, failures = compare_metrics(df, thresholds)
    if failures:
        logging.error("Regression failures detected (comparing methods within the same workload):")
        for metric, config, method, best, current, diff in failures:
            logging.error(f"Workload '{config}' method '{method}' for metric '{metric}': best = {best:.2f}, "
                          f"current = {current:.2f}, diff = {diff*100:.1f}%")
    else:
        logging.info("No inter-method regressions detected within individual workloads.")

    if args.plot_type in ['aggregate', 'both']:
        plot_aggregate_matrix(df, metrics=["latency_ms", "recall"], output_path=args.output_aggregate)

    if args.plot_type in ['detailed', 'both']:
        args.detailed_output_dir.mkdir(parents=True, exist_ok=True)
        plot_detailed_per_operation_all(df, output_dir=args.detailed_output_dir)

if __name__ == '__main__':
    main()