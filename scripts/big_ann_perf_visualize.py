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
def visualize_perf(src_dataset):
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
    fig, axs = plt.subplots(3, 2, figsize=(15, 10), sharex=True)

    # Row 0
    axs[0, 0].plot(x, recall_mean, color='green')
    axs[0, 0].set_ylabel('Recall@10')
    axs[0, 0].set_title('Recall Mean')

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

    # Hide unused subplot (2,1)
    axs[2, 1].axis('off')

    # Overall title including total latency sum in minutes
    fig.suptitle(f'BigANN streaming task query performance (k=10, Target Recall = 0.9, Num Operations = {NUM_OPERATIONS_TO_VISUALIZE})\nTotal Latency: {total_all_min:.2f} minutes', fontsize=16)
    
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

    # Bottom subplot: recall_mean vs step_num
    for i, nprobe in enumerate(nprobe_values):
        sub_df = search_df[search_df['nprobe_percentage'] == nprobe]
        axes[1].plot(sub_df['step_num'].values, sub_df['recall_mean'].values, label=f'Factor Scanned: {nprobe}', color=colors[i])
    axes[1].set_xlabel('Step Number')
    axes[1].set_ylabel('Recall Mean')
    axes[1].set_title('Recall Mean over Step Number')

    plt.tight_layout()
    plt.savefig(os.path.join(DST_DIR, src_dataset.replace("csv", "png")), dpi=300)

if __name__ == "__main__":
    # visualize_perf("scan_all_no_batch_no_aps.csv")
    # visualize_perf("scan_all_using_batching_250_no_aps.csv")
    # visualize_perf("scan_all_using_batching_2500_no_aps.csv")
    visualize_perf("scan_using_batching_250_aps_recall_0.9_search_0.3.csv")
    visualize_perf("scan_using_batching_250_aps_recall_0.9_search_0.15.csv")
    visualize_perf("scan_using_batching_250_aps_recall_0.9_search_0.15_0.05.csv")
    visualize_percentage_variation("scan_vary_using_batching_250_no_aps.csv")