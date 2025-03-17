#!/usr/bin/env python
import argparse
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(results_dir):
    """
    Recursively load all CSV files named "results.csv" from subdirectories of results_dir.
    Each CSV is assumed to come from one configuration run.
    """
    csv_files = glob.glob(os.path.join(results_dir, '**', 'results.csv'), recursive=True)
    if not csv_files:
        raise ValueError(f"No CSV results found in {results_dir}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract the configuration name from the parent directory name
        config_name = os.path.basename(os.path.dirname(csv_file))
        df['config'] = config_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compare_metrics(df, thresholds):
    """
    Compute average metrics per configuration and compare them against thresholds.

    thresholds is a dict where keys are metric names (e.g., 'latency_ms', 'recall')
    and values are the maximum acceptable relative differences.

    For 'latency_ms' lower values are better; for 'recall' higher is better.
    """
    comparisons = {}
    regression_failures = []
    for metric, threshold in thresholds.items():
        avg_metric = df.groupby('config')[metric].mean()
        comparisons[metric] = avg_metric

        # Define baseline: best value across configurations
        if metric == 'latency_ms':
            best = avg_metric.min()
            for config, value in avg_metric.items():
                diff = (value - best) / best
                if diff > threshold:
                    regression_failures.append((metric, config, best, value, diff))
        elif metric == 'recall':
            best = avg_metric.max()
            for config, value in avg_metric.items():
                diff = (best - value) / best
                if diff > threshold:
                    regression_failures.append((metric, config, best, value, diff))
    return comparisons, regression_failures

def plot_comparisons(comparisons, output_path):
    """
    Create a bar plot for each metric showing the average value for each configuration.
    """
    num_metrics = len(comparisons)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics))
    if num_metrics == 1:
        axes = [axes]
    for ax, (metric, series) in zip(axes, comparisons.items()):
        series.plot(kind='bar', ax=ax)
        ax.set_title(f'Average {metric} by Configuration')
        ax.set_xlabel('Configuration')
        ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple regression test results and produce a summary plot."
    )
    parser.add_argument(
        '--results_dir', type=str, default='results',
        help="Directory containing result CSV files (searched recursively)."
    )
    parser.add_argument(
        '--output_plot', type=str, default='results/comparison_plot.png',
        help="Output path for the comparison plot."
    )
    parser.add_argument(
        '--thresholds', type=str, default='{"latency_ms": 0.05, "recall": 0.01}',
        help=("JSON string of thresholds for regression checks "
              "(e.g., '{\"latency_ms\": 0.05, \"recall\": 0.01}').")
    )
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds)

    df = load_results(args.results_dir)
    comparisons, failures = compare_metrics(df, thresholds)

    if failures:
        print("Regression failures detected:")
        for metric, config, best, current, diff in failures:
            print(f"Metric '{metric}' in config '{config}' regressed: best = {best:.2f}, "
                  f"current = {current:.2f}, diff = {diff*100:.1f}%")
    else:
        print("All metrics are within acceptable thresholds.")

    plot_comparisons(comparisons, args.output_plot)

if __name__ == '__main__':
    main()