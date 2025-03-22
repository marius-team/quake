import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from quake import IndexBuildParams, QuakeIndex, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall, to_path

# Set up logging
log = logging.getLogger(__name__)

# Constants
MIN_FANOUT = 8


# Dataset Management
def get_dataset(cfg):
    dataset_path = get_original_cwd() / to_path(cfg.dataset.path)
    vectors, queries, gt = load_dataset(cfg.dataset.name, dataset_path)
    return vectors, queries[: cfg.experiment.nq], gt[: cfg.experiment.nq]


# Index Management
def build_or_load_index(cfg, num_workers):
    index_dir = get_original_cwd() / to_path(cfg.paths.index_dir)
    index_path = index_dir / f"{cfg.dataset.name}_dynamic_ivf{cfg.index.nc}.index"
    if not index_path.exists() or cfg.overwrite.index:
        # Build and save index
        vectors, _, _ = get_dataset(cfg)
        build_index = QuakeIndex()
        build_params = IndexBuildParams()
        build_params.nlist = cfg.index.nc
        build_params.metric = cfg.index.metric
        build_index.build(vectors, torch.arange(vectors.size(0)), build_params)
        build_index.save(str(index_path.absolute()))
        log.info(f"Index built and saved to {index_path}")

    # Load existing index
    index = QuakeIndex()
    index.load(str(index_path.absolute()), num_workers)

    log.info(f"Index loaded from {index_path} with {num_workers} workers")

    return index


# Experiment Execution
def run_single_experiment(args):
    method, recall_target, recompute_ratio, use_precompute, cfg, action, n_workers = args
    index = build_or_load_index(cfg, n_workers)
    _, queries, gt = get_dataset(cfg)
    k = cfg.experiment.k

    result_dir = get_original_cwd() / cfg.paths.results_dir / method
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running experiment for {method} at recall {recall_target} with action {action}")

    if method == "Oracle":
        result_path = result_dir / f"recall_{recall_target:.2f}.csv"

        data_df = run_experiment_for_configuration(
            index=index,
            queries=queries,
            gt=gt,
            k=k,
            recall_target=recall_target,
            oracle=True,
        )
        # Save per-query data
        data_df.to_csv(result_path, index=False)
        log.info(f"Results saved to {result_path}")
    elif method == "FixedNProbe":
        result_path = result_dir / f"recall_{recall_target:.2f}.csv"
        if result_path.exists() and not cfg.overwrite.results:
            log.info(f"Results for {method} at recall {recall_target} already exist. Skipping.")
            return

        data_df = run_experiment_for_configuration(
            index=index,
            queries=queries,
            gt=gt,
            k=k,
            recall_target=recall_target,
            fixed_nprobe=True,
        )
        # Save per-query data
        data_df.to_csv(result_path, index=False)
        log.info(f"Results saved to {result_path}")
    elif method.startswith("APS"):
        result_path = result_dir / f"recall_{recall_target:.2f}.csv"
        if result_path.exists() and not cfg.overwrite.results:
            log.info(f"Results for {method} at recall {recall_target} already exist. Skipping.")
            return

        data_df = run_experiment_for_configuration(
            index=index,
            queries=queries,
            gt=gt,
            k=k,
            recall_target=recall_target,
            recompute_ratio=recompute_ratio,
            use_precompute=use_precompute,
        )
        # Save per-query data
        data_df.to_csv(result_path, index=False)
        log.info(f"Results saved to {result_path}")
    else:
        raise ValueError(f"Unknown method: {method}")


# Results Management and Plotting
def collect_and_plot_results(cfg):
    methods = cfg.methods
    recall_targets = cfg.experiment.recall_targets
    all_data = []

    for method in methods:
        for recall_target in recall_targets:
            result_dir = cfg.paths.results_dir / method
            result_path = result_dir / f"recall_{recall_target:.2f}.csv"
            if not result_path.exists():
                log.warning(f"Result file {result_path} does not exist. Skipping.")
                continue
            data_df = pd.read_csv(result_path)
            data_df["Recall Target"] = recall_target

            complete_name = method
            data_df["Method"] = complete_name
            all_data.append(data_df)

    if not all_data:
        log.error("No data available for plotting.")
        return

    df_plot = pd.concat(all_data, ignore_index=True)

    # Clean data
    df_plot = df_plot.dropna(subset=["total_time_ms", "nprobe", "recall"])
    df_plot = df_plot[(df_plot["total_time_ms"] >= 0) & (df_plot["nprobe"] >= 0) & (df_plot["recall"] >= 0)]

    df_plot["Query Time (ms)"] = df_plot["total_time_ms"]
    df_plot["Recall"] = df_plot["recall"]

    # Compute stats
    grouped = df_plot.groupby(["Recall Target", "Method"])
    stats = grouped.agg(
        {
            "Query Time (ms)": ["min", "mean", "max"],
            "nprobe": ["min", "mean", "max"],
            "Recall": ["min", "mean", "max"],
            "buffer_init_time_ms": ["min", "mean", "max"],
            "job_enqueue_time_ms": ["min", "mean", "max"],
            "boundary_distance_time_ms": ["min", "mean", "max"],
            "job_wait_time_ms": ["min", "mean", "max"],
            "result_aggregate_time_ms": ["min", "mean", "max"],
        }
    ).reset_index()
    stats.columns = [" ".join(col).strip() for col in stats.columns.values]

    # Compute 'other_time_ms' and include in stats
    df_plot["other_time_ms"] = df_plot["total_time_ms"] - (
        df_plot["buffer_init_time_ms"]
        + df_plot["job_enqueue_time_ms"]
        + df_plot["boundary_distance_time_ms"]
        + df_plot["job_wait_time_ms"]
        + df_plot["result_aggregate_time_ms"]
    )
    df_plot["other_time_ms"] = df_plot["other_time_ms"].apply(lambda x: x if x >= 0 else 0)
    other_time_stats = grouped["other_time_ms"].agg(["min", "mean", "max"]).reset_index()
    other_time_stats.columns = [
        "Recall Target",
        "Method",
        "other_time_ms min",
        "other_time_ms mean",
        "other_time_ms max",
    ]
    stats = pd.merge(stats, other_time_stats, on=["Recall Target", "Method"])

    # Compute p99 for 'Query Time (ms)' and 'Recall'
    p95_latency = grouped["Query Time (ms)"].quantile(0.95).reset_index()
    p95_recall = grouped["Recall"].quantile(0.05).reset_index()
    p95_latency.rename(columns={"Query Time (ms)": "Query Time (ms) p95"}, inplace=True)
    p95_recall.rename(columns={"Recall": "Recall p95"}, inplace=True)
    stats = stats.merge(p95_latency, on=["Recall Target", "Method"])
    stats = stats.merge(p95_recall, on=["Recall Target", "Method"])

    # Save this as intermediate results
    stats_save_path = cfg.paths.plot_dir / "all_intermediate_stats.csv"
    stats.to_csv(stats_save_path, index=False)

    # Plotting
    plot_recall_only(df_plot, stats, cfg.paths.plot_dir)
    plot_mean_line_plots(df_plot, stats, cfg.paths.plot_dir)
    plot_query_overheads(stats, cfg.paths.plot_dir)


palette = {"Oracle": "C0", "APS": "C1"}


def plot_recall_only(df_plot, stats, plot_dir):
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.8)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig, ax = plt.subplots(figsize=(6, 4))
    stats["Method"] = stats["Method"].replace("Adaptive nprobe", "APS")

    # Shared x-axis range
    x_min = stats["Recall Target"].min()
    x_max = stats["Recall Target"].max()

    # Generate detailed x-axis ticks
    x_ticks = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    ax.set_title("Sift1M: Measured Recall vs. Recall Target", fontsize=14, fontweight="bold")

    for method in stats["Method"].unique():
        method_stats = stats[stats["Method"] == method].sort_values("Recall Target")
        ax.plot(
            method_stats["Recall Target"],
            method_stats["Recall mean"],
            label=method,
            color=palette[method],
            linewidth=4,
            marker="o",
            markersize=10,
        )

        # plot p95 recall
        # p95_recall = method_stats['Recall p95']
        # ax.plot(method_stats['Recall Target'], p95_recall, color=palette[method], linestyle='--', linewidth=8)

    x_vals = np.linspace(x_min, x_max, 100)
    ax.plot(x_vals, x_vals, color="black", linestyle="--", linewidth=4, label="Recall Target")

    ax.set_ylabel("Recall", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])  # Custom y-axis ticks
    ax.set_xticks(x_ticks)
    ax.legend(fontsize=8, prop={"weight": "bold"}, loc="lower right")  # Legend moved to top-left
    ax.grid(True)

    plt.tight_layout()
    plot_path = plot_dir / "recall_plot.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    log.info(f"Plot saved to {plot_path}")
    plt.show()


# Plotting Functions
def plot_mean_line_plots(df_plot, stats, plot_dir):
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.8)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Create a vertically stacked layout with shared x-axis
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)  # Stacked plots with shared x-axis

    stats["Method"] = stats["Method"].replace("Adaptive nprobe", "APS")

    # Shared x-axis range
    x_min = stats["Recall Target"].min()
    x_max = stats["Recall Target"].max()

    # Generate detailed x-axis ticks
    x_ticks = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    axes[0].set_title("Sift1M: Recall, Query Latency, and Nprobe vs. Recall Target", fontsize=20, fontweight="bold")

    # Plot 1: Recall
    for method in stats["Method"].unique():
        method_stats = stats[stats["Method"] == method].sort_values("Recall Target")
        axes[0].plot(
            method_stats["Recall Target"],
            method_stats["Recall mean"],
            label=method,
            color=palette[method],
            linewidth=8,
            marker="o",
            markersize=10,
        )
        # plot p95 recall
        # p95_recall = method_stats['Recall p95']
        # axes[0].plot(method_stats['Recall Target'], p95_recall, color=palette[method], linestyle='--', linewidth=8)

    # Diagonal reference line
    x_vals = np.linspace(x_min, x_max, 100)
    axes[0].plot(x_vals, x_vals, color="black", linestyle="--", linewidth=8, label="Recall Target")

    axes[0].set_ylabel("Recall", fontsize=16, fontweight="bold")
    axes[0].tick_params(axis="both", which="major", labelsize=16)
    axes[0].set_yticks([0.7, 0.8, 0.9, 1.0])  # Custom y-axis ticks
    axes[0].set_xticks(x_ticks)
    axes[0].legend(fontsize=8, prop={"weight": "bold"}, loc="lower right")  # Legend moved to top-left
    axes[0].grid(True)

    # Plot 2: Query Time
    for method in stats["Method"].unique():
        method_stats = stats[stats["Method"] == method].sort_values("Recall Target")
        axes[1].plot(
            method_stats["Recall Target"],
            method_stats["Query Time (ms) mean"],
            label=method,
            color=palette[method],
            linewidth=8,
            marker="o",
            markersize=10,
        )

        # plot p95 query time
        p95_query_time = method_stats["Query Time (ms) p95"]
        axes[1].plot(method_stats["Recall Target"], p95_query_time, color=palette[method], linestyle="--", linewidth=8)

    # axes[1].set_ylabel('Query Time (ms)', fontsize=20, fontweight='bold')
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="both", which="major", labelsize=16)
    axes[1].set_yticks([0.2, 0.4, 0.8, 1.6, 3.2])  # Custom y-axis ticks
    axes[1].set_yticklabels(["0.2ms", "0.4ms", "0.8ms", "1.6ms", "3.2ms"])  # Add units to tick labels
    axes[1].set_xticks(x_ticks)
    # axes[1].legend(fontsize=16, prop={'weight': 'bold'}, loc='upper left')  # Legend moved to top-left
    axes[1].grid(True)
    #
    # # Plot 3: nprobe
    for method in stats["Method"].unique():
        method_stats = stats[stats["Method"] == method].sort_values("Recall Target")
        axes[2].plot(
            method_stats["Recall Target"],
            method_stats["nprobe mean"],
            label=method,
            color=palette[method],
            linewidth=8,
            marker="o",
            markersize=10,
        )

    axes[2].set_xlabel("Recall Target", fontsize=16, fontweight="bold")  # Shared x-axis
    axes[2].set_ylabel("nprobe", fontsize=16, fontweight="bold")
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="both", which="major", labelsize=16)
    axes[2].set_yticks([4, 8, 16, 32, 64])  # Custom y-axis ticks
    axes[2].set_yticklabels(["4", "8", "16", "32", "64"])  # Add units to tick labels
    axes[2].set_xticks(x_ticks)
    # axes[2].legend(fontsize=16, prop={'weight': 'bold'}, loc='upper left')  # Legend moved to top-left
    axes[2].grid(True)

    plt.tight_layout()
    plot_path = plot_dir / "mean_line_plots_stacked.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    log.info(f"Plot saved to {plot_path}")
    plt.show()


def plot_query_overheads(stats, plot_dir):
    """
    Plot stacked bar charts showing the mean time taken by each component for each method and recall target,
    including an 'Other' category representing the remaining time.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    # Define the components and their labels
    components = [
        "buffer_init_time_ms",
        "job_enqueue_time_ms",
        "boundary_distance_time_ms",
        "job_wait_time_ms",
        "result_aggregate_time_ms",
        "other_time_ms",  # Added 'Other' category
    ]
    component_labels = [
        "Buffer Init",
        "Job Enqueue",
        "Boundary Distance",
        "Job Wait",
        "Result Aggregate",
        "Other",  # Label for 'Other'
    ]
    # Blue, Orange, Green, Red, Purple, Gray]
    component_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#7f7f7f"]

    # remap the method names: Adaptive nprobe -> Adaptive Partition Scanning
    # stats[methods == 'Adaptive nprobe'] = 'Adaptive Partition Scanning'
    stats["Method"] = stats["Method"].replace("Adaptive nprobe", "APS")

    # Define methods and their hatching patterns
    methods = stats["Method"].unique()

    method_hatches = {method: "" if method == "Oracle" else "//" for method in methods}

    # Prepare data for plotting
    recall_targets = sorted(stats["Recall Target"].unique())

    # Initialize plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    bar_width = 0.1
    opacity = 0.8

    n_groups = len(recall_targets)
    index = np.arange(n_groups)

    # Set positions for each method's bars
    # method_offsets = np.linspace(-bar_width/2, bar_width/2, len(methods))
    positions = {}
    positions["Oracle"] = index - 3 * bar_width / 2
    positions["APS"] = index - bar_width / 2
    positions["APS-R"] = index + bar_width / 2
    positions["APS-RP"] = index + 3 * bar_width / 2
    positions["FixedNProbe"] = index + 5 * bar_width / 2

    # Initialize cumulative bottoms for stacking
    cumulative_bottoms = {method: np.zeros(n_groups) for method in methods}

    # Plot each component
    for comp, comp_label, comp_color in zip(components, component_labels, component_colors):
        for method in methods:
            print(f"Plotting {comp} for {method}")

            comp_means = []
            for rt in recall_targets:
                row = stats[(stats["Recall Target"] == rt) & (stats["Method"] == method)]
                if not row.empty:
                    comp_time = row[f"{comp} mean"].values[0]
                else:
                    comp_time = 0
                comp_means.append(comp_time)
            print(f"Component means: {comp_means}")

            method_lookup = method.split("+")[0].strip()
            pos = positions[method_lookup]
            ax.bar(
                pos,
                comp_means,
                bar_width,
                bottom=cumulative_bottoms[method],
                color=comp_color,
                alpha=opacity,
                hatch=method_hatches[method],
                edgecolor="black",
                label=comp_label if (method == methods[0]) else "",
            )
            cumulative_bottoms[method] += comp_means

    # Set x-axis labels and ticks
    ax.set_xlabel("Recall Target", fontsize=14, weight="bold")
    ax.set_ylabel("Time (ms)", fontsize=14, weight="bold")
    ax.set_title("Query Overheads by Method and Recall Target", fontsize=16, weight="bold")
    ax.set_xticks(index)
    ax.set_xticklabels([f"{rt:.2f}" for rt in recall_targets], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Create custom legend for components
    component_handles = [Patch(facecolor=color, edgecolor="black") for color in component_colors]
    component_legend = ax.legend(
        component_handles, component_labels, title="Components", title_fontsize=14, fontsize=12, loc="upper left"
    )

    # Create custom legend for methods
    method_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=method_hatches[method], label=method) for method in methods
    ]
    ax.legend(method_handles, methods, title="Method", title_fontsize=14, fontsize=12, loc="upper right")

    # Add the component legend back to the plot
    ax.add_artist(component_legend)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.5)

    # set ylimit
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plot_path = plot_dir / "query_overheads.png"
    plt.savefig(plot_path)
    log.info(f"Overhead plot saved to {plot_path}")
    plt.show()


# Helper Functions
def run_experiment_for_configuration(
    index, queries, gt, k, recall_target, oracle=False, recompute_ratio=0.05, use_precompute=True, fixed_nprobe=False
):

    timing_infos = []

    if oracle:
        nprobes = []
        for i, query in enumerate(queries):
            max_nprobe = index.nlist()
            min_nprobe = 1
            best_nprobe = max_nprobe

            # find minimum nprobe that achieves recall_target
            while min_nprobe <= max_nprobe:
                curr_nprobe = (min_nprobe + max_nprobe) // 2

                search_params = SearchParams()
                search_params.nprobe = curr_nprobe
                search_params.k = k
                search_params.recall_target = -1

                search_result = index.search(query.unsqueeze(0), search_params)
                ids, dist, timing_info = search_result.ids, search_result.distances, search_result.timing_info
                recall = compute_recall(ids, gt[i].unsqueeze(0), k)

                if float(recall) >= recall_target:
                    max_nprobe = curr_nprobe - 1
                    best_nprobe = curr_nprobe
                else:
                    min_nprobe = curr_nprobe + 1

            nprobes.append(best_nprobe)

        # Second run to execute queries
        all_ids = []
        all_dists = []

        for query, nprobe in zip(queries, nprobes):

            search_params = SearchParams()
            search_params.nprobe = nprobe
            search_params.k = k
            search_params.recall_target = -1

            search_result = index.search(query.unsqueeze(0), search_params)

            ids, dist, timing_info = search_result.ids, search_result.distances, search_result.timing_info
            all_ids.append(ids)
            all_dists.append(dist)
            timing_infos.append(timing_info)
        recalls = compute_recall(torch.cat(all_ids, dim=0), gt, k)
    elif fixed_nprobe:
        print("FIXED NPROBE")
        # compute nprobe for recall target for all queries, rather than per-query

        # first use binary search to find the nprobe that achieves the recall target
        max_nprobe = index.nlist()
        min_nprobe = 1
        best_nprobe = max_nprobe

        while min_nprobe <= max_nprobe:
            curr_nprobe = (min_nprobe + max_nprobe) // 2
            all_ids = []
            all_dists = []

            search_params = SearchParams()
            search_params.nprobe = curr_nprobe
            search_params.k = k
            search_params.recall_target = -1

            for i, query in enumerate(queries):
                search_result = index.search(query.unsqueeze(0), search_params)
                ids, dist, timing_info = search_result.ids, search_result.distances, search_result.timing_info
                all_ids.append(ids)
                all_dists.append(dist)
            recalls = compute_recall(torch.cat(all_ids, dim=0), gt, k)
            avg_recall = recalls.mean().item()
            if avg_recall >= recall_target:
                max_nprobe = curr_nprobe - 1
                best_nprobe = curr_nprobe
            else:
                min_nprobe = curr_nprobe + 1

        # Second run to execute queries
        all_ids = []
        all_dists = []

        search_params = SearchParams()
        search_params.nprobe = best_nprobe
        search_params.k = k
        search_params.recall_target = -1

        for query in queries:
            search_result = index.search(query.unsqueeze(0), search_params)
            ids, dist, timing_info = search_result.ids, search_result.distances, search_result.timing_info
            all_ids.append(ids)
            all_dists.append(dist)
            timing_infos.append(timing_info)
        recalls = compute_recall(torch.cat(all_ids, dim=0), gt, k)
    else:
        all_ids = []
        all_dists = []

        search_params = SearchParams()
        search_params.nprobe = -1
        search_params.k = k
        search_params.initial_search_fraction = 0.1
        search_params.recall_target = recall_target
        search_params.recompute_threshold = 0.01
        search_params.use_precomputed = use_precompute
        search_params.num_threads = 1

        # debug print search params
        print(
            f"Search Params: {search_params.nprobe}, {search_params.k}, {search_params.recall_target}, "
            f"{search_params.recompute_threshold}, {search_params.use_precomputed}"
        )

        for query in queries:
            search_result = index.search(query.unsqueeze(0), search_params)
            ids, dist, timing_info = search_result.ids, search_result.distances, search_result.timing_info
            all_ids.append(ids)
            all_dists.append(dist)
            timing_infos.append(timing_info)

        ids = torch.cat(all_ids, dim=0)
        # dists = torch.cat(all_dists, dim=0)
        recalls = compute_recall(ids, gt, k)

    per_query_data = []
    for i, timing_info in enumerate(timing_infos):
        per_query_nprobe = timing_info.partitions_scanned
        buffer_init_time_ms = timing_info.buffer_init_time_ns / 1e6
        job_enqueue_time_ms = timing_info.job_enqueue_time_ns / 1e6
        boundary_distance_time_ms = timing_info.boundary_distance_time_ns / 1e6
        job_wait_time_ms = timing_info.job_wait_time_ns / 1e6
        result_aggregate_time_ms = timing_info.result_aggregate_time_ns / 1e6
        total_time_ms = timing_info.total_time_ns / 1e6

        per_query_recall = float(recalls[i])
        per_query_data.append(
            {
                "nprobe": per_query_nprobe,
                "recall": per_query_recall,
                "buffer_init_time_ms": buffer_init_time_ms,
                "job_enqueue_time_ms": job_enqueue_time_ms,
                "boundary_distance_time_ms": boundary_distance_time_ms,
                "job_wait_time_ms": job_wait_time_ms,
                "result_aggregate_time_ms": result_aggregate_time_ms,
                "total_time_ms": total_time_ms,
            }
        )

    per_query_data_df = pd.DataFrame(per_query_data)
    # per_query_data_df = per_query_data_df[1:]  # Skip the first query
    return per_query_data_df


def get_nprobe_for_recall_target(recall_target, nlist):
    nprobe = int(nlist * recall_target / 2)
    nprobe = max(1, min(nprobe, nlist))
    return nprobe


@hydra.main(config_path="configs", config_name="sift1m")
def main(cfg: DictConfig):
    # Set up directories
    base_dir = get_original_cwd() / Path(cfg.base_dir)

    cfg.paths.index_dir = Path(cfg.paths.index_dir)
    cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.results_dir = base_dir / cfg.paths.results_dir
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.plot_dir = base_dir / cfg.paths.plot_dir
    cfg.paths.plot_dir.mkdir(parents=True, exist_ok=True)

    if cfg.mode == "run":
        # Prepare experiment parameters
        methods = cfg.methods
        recall_targets = cfg.experiment.recall_targets
        n_workers = cfg.experiment.n_workers
        experiment_args = []
        for method in methods:
            for recall_target in recall_targets:
                if method == "Oracle":
                    experiment_args.append((method, recall_target, 0.05, True, cfg, "execute_queries", n_workers))
                elif method == "APS":
                    experiment_args.append((method, recall_target, 0.0, True, cfg, "execute_queries", n_workers))
                elif method == "APS-R":
                    experiment_args.append((method, recall_target, -1, True, cfg, "execute_queries", n_workers))
                elif method == "APS-RP":
                    experiment_args.append((method, recall_target, -1, False, cfg, "execute_queries", n_workers))
                elif method == "FixedNProbe":
                    experiment_args.append((method, recall_target, -1, True, cfg, "execute_queries", n_workers))

        # Run experiments
        for args in experiment_args:
            run_single_experiment(args)

    elif cfg.mode == "plot":
        collect_and_plot_results(cfg)


if __name__ == "__main__":
    main()
