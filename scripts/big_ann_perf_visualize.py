import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, LogLocator, FormatStrFormatter

SECONDS_TO_MS = 10.0 ** 3
MIN_TO_SECONDS = 60
MIN_TO_MS = MIN_TO_SECONDS * SECONDS_TO_MS

SRC_DIR = "big_ann_perf_numbers"
DST_DIR = "bigann_perf_vis"

NUM_OPERATIONS_TO_VISUALIZE = 1300
def visualize_perf(src_dataset, config_details):
    # Load CSV
    df = pd.read_csv(os.path.join(SRC_DIR, src_dataset))
    df = df.head(NUM_OPERATIONS_TO_VISUALIZE)
    df = df.fillna(0)

    # Filter rows for step_type == "search"
    search_df = df[df['step_type'] == 'search']

    # Extract data
    x = search_df['step_num'].values
    recall_mean = np.round(search_df['recall_mean'].values, 2)
    latency_search = search_df['latency_ms'].values

    insert_df = df[df['step_type'] == 'insert']
    latency_insert = insert_df['latency_ms'].values
    x_insert = insert_df['step_num'].values

    delete_df = df[df['step_type'] == 'delete']
    latency_delete = delete_df['latency_ms'].values
    x_delete = delete_df['step_num'].values

    x_maintenance = df['step_num'].values
    latency_maintenance = df['mainteance_ms'].values

    # Convert sums to minutes
    total_search_min = latency_search.sum() / MIN_TO_MS
    total_delete_min = latency_delete.sum() / MIN_TO_MS
    total_insert_min = latency_insert.sum() / MIN_TO_MS
    total_maintenance_min = latency_maintenance.sum() / MIN_TO_MS
    total_all_min = total_search_min + total_delete_min + total_insert_min + total_maintenance_min

    # Create 3 rows x 2 columns subplots
    fig, axs = plt.subplots(4, 2, figsize=(12, 12), sharex=True)

    # Row 0
    axs[0, 0].plot(x, recall_mean, color='green')
    axs[0, 0].set_ylabel('Recall@10')
    axs[0, 0].set_title('Recall Mean')
    print("Average Recall for run", src_dataset, "is", np.mean(recall_mean))

    axs[0, 1].plot(x, latency_search, color='purple')
    axs[0, 1].set_ylabel('Latency (ms)')
    axs[0, 1].set_title(f'Search Latency (Total: {total_search_min:.2f} min)')

    # Row 1
    axs[1, 0].plot(x_delete, latency_delete, color='red')
    axs[1, 0].set_ylabel('Latency (ms)')
    axs[1, 0].set_title(f'Delete Latency (Total: {total_delete_min:.2f} min)')

    axs[1, 1].plot(x_insert, latency_insert, color='orange')
    axs[1, 1].set_ylabel('Latency (ms)')
    axs[1, 1].set_title(f'Insert Latency (Total: {total_insert_min:.2f} min)')

    # Row 2
    axs[2, 0].plot(x_maintenance, latency_maintenance, color='blue')
    axs[2, 0].set_xlabel('Step Num')
    axs[2, 0].set_ylabel('Latency (ms)')
    axs[2, 0].set_title(f'Maintenance Latency (Total: {total_maintenance_min:.2f} min)')

    axs[2, 0].plot(x_maintenance, latency_maintenance, color='blue')

    # Bottom right subplot: Vectors and Partitions over time
    non_search_df = df[df['step_type'] != 'search']
    ax1 = axs[2, 1]
    ax1.plot(non_search_df['step_num'].values, non_search_df['num_vectors'].values, color='olive', label='Num Vectors')
    ax1.set_xlabel('Step Num')
    ax1.set_ylabel('Num Vectors', color='olive')
    ax1.tick_params(axis='y', labelcolor='olive')

    # Create second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot num_partitions on right y-axis
    ax2.plot(non_search_df['step_num'].values, non_search_df['num_partitions'].values, color='brown', label='Num Partitions')
    ax2.set_ylabel('Num Partitions', color='brown')
    ax2.tick_params(axis='y', labelcolor='brown')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

    axs[2, 1].set_title('Vector and Partition Count')

    # Plot the scan percentage needed to cover the ground truth vectors
    axs[3, 0].plot(search_df['step_num'].values, search_df['gt_scan_mean'].values, color='green')
    axs[3, 0].fill_between(search_df['step_num'].values, search_df['gt_scan_mean'].values - search_df['gt_scan_dev'].values, search_df['gt_scan_mean'].values + search_df['gt_scan_dev'].values, 
        color='green', alpha=0.3)
    axs[3, 0].set_xlabel('Step Num')
    axs[3, 0].set_ylabel('Partition Scan Percentage')
    axs[3, 0].set_title('Rank of GT partitions as Search Candidates')

    # Plot Scan Throughput
    axs[3, 1].plot(search_df['step_num'].values, search_df['worker_scan_throughput'].values, color='gray')
    axs[3, 1].set_xlabel('Step Num')
    axs[3, 1].set_ylabel('Scan Throughput (GB/s)')
    axs[3, 1].set_title('Worker Scan Throughput')

    # Overall title including total latency sum in minutes
    fig.suptitle(f'{config_details}\nTotal Latency: {total_all_min:.2f} minutes', fontsize=16)
    
    fig.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

def visualize_percentage_variation(src_dataset):
    # Load CSV
    df = pd.read_csv(os.path.join(SRC_DIR, src_dataset))
    search_df = df[df['step_type'] == 'search']

    # Get unique nprobe_percentage values
    nprobe_values = sorted(search_df['nprobe_percentage'].unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top subplot: latency_ms vs step_num
    for i, nprobe in enumerate(nprobe_values):
        sub_df = search_df[search_df['nprobe_percentage'] == nprobe]
        axes[0].plot(sub_df['step_num'].values, sub_df['latency_ms'].values, label=f'Factor Scanned: {nprobe}', color=colors[i])
        total_time_minutes = sub_df['latency_ms'].values.sum() / MIN_TO_MS
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_yscale('log')
    axes[0].set_title('Search Latency over Step Number')
    axes[0].legend()

    # Bottom left subplot: recall_mean vs step_num
    for i, nprobe in enumerate(nprobe_values):
        sub_df = search_df[search_df['nprobe_percentage'] == nprobe]
        axes[1].plot(sub_df['step_num'].values, sub_df['recall_mean'].values, label=f'Factor Scanned: {nprobe}', color=colors[i])
    axes[1].set_xlabel('Step Number')
    axes[1].set_ylabel('Recall Mean')
    axes[1].set_title('Recall Mean over Step Number')

    plt.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

def visualize_vary_worker(src_dataset):
    # Read CSV
    df = pd.read_csv(os.path.join(SRC_DIR, src_dataset))
    search_df = df[df['step_type'] == 'search']

    # List of metrics to plot
    metrics = [
        ("search_latency_ms", "Batch Search Latency (seconds)"),
        ("worker_partition_size", "Avg Scan Partition Size"), 
        ("worker_scan_throughput", "Partition Scan Throughput (GB/s)"), 
        ("recall_mean", "Query Recall")
    ]

    # Get unique worker counts for consistent colors
    worker_counts = sorted(search_df['num_search_workers'].unique())
    colors = plt.get_cmap('tab10').colors  # or use any other color map

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 10))
    axs = axs.flatten()

    for idx, (metric, metric_label) in enumerate(metrics):
        scale_factor = 1.0
        if metric == "search_latency_ms":
            scale_factor = 1000.0

        ax = axs[idx]
        for i, worker in enumerate(worker_counts):
            data = search_df[search_df['num_search_workers'] == worker].sort_values('step_num')
            ax.plot(data['step_num'].values, data[metric].values/scale_factor, label=f'Num Search Workers: {worker}', color=colors[i % len(colors)])
        
        ax.set_xlabel('Step Num')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()

        if metric == "search_latency_ms":
            ax.set_yscale('log')
            minor_locator = mticker.LogLocator(subs=np.arange(2, 10))
            ax.yaxis.set_minor_locator(minor_locator)
            minor_formatter = mticker.FormatStrFormatter("%.1f")
            ax.yaxis.set_minor_formatter(minor_formatter)

            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

    fig.suptitle(f'Vary Search Workers Experiemnt (Scan Percentage = 15%, Batch Size = 500)', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

def visualize_vary_batch_size(src_dataset):
    # Read CSV
    df = pd.read_csv(os.path.join(SRC_DIR, src_dataset))
    search_df = df[df['step_type'] == 'search']
    search_df = search_df[search_df["step_num"] > 5]

    metric_names = [
        ('search_latency_ms', 'Search Latency (ms)'),
        ('worker_partition_size', 'Partition Size'),
        ('worker_scan_throughput', 'Worker Scan Throughput'),
        ('worker_scan_time_ms', 'Worker Scan Time (ms)'),
        ('worker_result_time_ms', 'Worker Result Write Time (ms)')
    ]
    colors = plt.get_cmap('tab10').colors  # or use any other color map

    batch_sizes = list(search_df['batch_size'].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for i, (col, title) in enumerate(metric_names):

        ax = axes[i]
        for j, batch in enumerate(batch_sizes):
            batch_data = search_df[search_df['batch_size'] == batch]
            ax.plot(batch_data['step_num'].values, batch_data[col].values, label=f'Batch Size {batch}', color=colors[j % len(colors)])
        ax.set_title(title)

        ax.set_xlabel('Step Num')
        ax.legend()

    fig.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

def visualize_hardware_metrics(src_dataset):
    # Read CSV
    df = pd.read_csv(os.path.join(SRC_DIR, src_dataset))
    search_df = df[df['step_type'] == 'search']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # 1) worker_partition_size versus step_num
    axs[0, 0].plot(search_df["step_num"].values, search_df["worker_partition_size"].values)
    axs[0, 0].set_title("Partition Size vs Step")
    axs[0, 0].set_ylabel("Worker Partition Size")

    # 2) worker_scan_throughput versus step_num
    axs[0, 1].plot(search_df["step_num"].values, search_df["worker_scan_throughput"].values, color="orange")
    axs[0, 1].set_title("Scan Throughput vs Step")
    axs[0, 1].set_ylabel("Scan Throughput")

    # 3) measured_ipc versus step_num
    axs[1, 0].plot(search_df["step_num"].values, search_df["measured_ipc"].values, color="green")
    axs[1, 0].set_title("Measured IPC vs Step")
    axs[1, 0].set_xlabel("Step Num")
    axs[1, 0].set_ylabel("Measured IPC")

    # 4) cache_miss_rate versus step_num
    axs[1, 1].plot(search_df["step_num"].values, search_df["cache_miss_rate"].values, color="red")
    axs[1, 1].set_title("Cache Miss Rate vs Step")
    axs[1, 1].set_xlabel("Step Num")
    axs[1, 1].set_ylabel("Cache Miss Rate")

    fig.suptitle(f'Batch Query Search with Hardware Counters (# of Workers = 8, Batch Size = 256)', fontsize=16)

    fig.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

if __name__ == "__main__":
    '''
    # visualize_vary_worker("perf_debug_scan_0.15_vary_num_search_workers.csv")
    # visualize_vary_batch_size("perf_debug_scan_0.15_vary_batch_size.csv")
    visualize_hardware_metrics("perf_debug_scan_0.15_counters.csv")
    '''

    configs_to_visualize = [
        ("test_perf_debug_scan_0.14_worker_batch_tuning.csv", "Scan Percentage = 14%, Query Batch Size = 256, Partition Chunk Size = 256"),
        ("perf_debug_scan_0.14_worker_batch_tuning_lower.csv", "Scan Percentage = 14%, Query Batch Size = 256, Partition Chunk Size = 128")
    ]

    for dataset, config_details in configs_to_visualize:
        visualize_perf(dataset, config_details)
