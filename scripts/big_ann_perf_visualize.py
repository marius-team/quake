import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm

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
    print("Average Recall across the run is", np.mean(recall_mean), "for", src_dataset)

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

    # Hide unused subplot (3,1)
    axs[3, 1].axis('off')
    for i in range (4):
        for j in range(2):
            axs[i, j].tick_params(labelbottom=True)

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

if __name__ == "__main__":
    configs_to_visualize = [
        ("scan_0.1_no_aps_refinment_wma_delete.csv", "Scan Percentage = 10%, Mainteance with No Refinment and Search with No APS"),
        ("scan_0.12_no_aps_refinment_wma_delete.csv", "Scan Percentage = 12%, Mainteance with No Refinment and Search with No APS")
    ]

    for dataset, config_details in configs_to_visualize:
        visualize_perf(dataset, config_details)