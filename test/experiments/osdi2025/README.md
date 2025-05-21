
# Quake OSDI 2025 Artifact Evaluation

This artifact provides the experimental setup for the Quake vector search system, as described in the OSDI 2025 submission.

## Structure

The experiments are organized as follows:



```plaintext
./test/experiments/osdi2025/
├── common/                  \# Shared utilities for experiments
│   └── experiment\_utils.py
├── experiment\_runner.py     \# Main script to run any experiment
├── README.md                \# This file
├── \<experiment\_name\>/       \# Directory for each specific experiment
│   ├── configs/             \# YAML configuration files for the experiment
│   │   └── \<dataset\_config\>.yaml
│   └── run.py               \# Python script to execute the experiment logic
└── ...                      \# Other experiment directories
```

Each experiment directory (e.g., `aps_recall_targets`, `early_termination`) contains its specific logic in `run.py` and configurations in its `configs/` subdirectory. The `common/experiment_utils.py` module provides shared functionality for loading data, configurations, preparing indexes, and plotting, aiming to simplify individual experiment scripts.

## Installation

**Requirements:**
- Python 3.9 or later
- Linux (Tested on Ubuntu 22.04)
- Conda package manager

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/marius-team/quake.git](https://github.com/marius-team/quake.git)
    cd quake
    git submodule update --init --recursive
    ```

2.  **Set Up the Conda Environment:**
    Create and activate the environment using the provided YAML file:
    ```bash
    conda env create -f environments/ubuntu-latest/conda.yaml
    conda activate quake-env
    ```

3.  **Install Quake:**
    Ensure you are in the `quake` repository root.
    ```bash
    pip install --no-use-pep517 .
    ```

## Running Experiments

All experiments are launched using the central `experiment_runner.py` script located in the `test/experiments/osdi2025/` directory.

**General Command Structure:**
Make sure to run all commands from the **repository root directory (`quake/`)**.

```bash
OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
    --experiment <experiment_name> \
    --config <config_name_without_yaml> \
    --output-dir <path_to_custom_output_directory> # Optional
````

**Explanation:**

* `OMP_NUM_THREADS=1`: Recommended for consistent performance measurement as OMP interferes with our custom parallelism. Adjust as needed based on specific experiment guidelines if any.
* `--experiment <experiment_name>`: Specifies which experiment to run (e.g., `kick_the_tires`, `aps_recall_targets`).
* `--config <config_name_without_yaml>`: Specifies the configuration file from `test/experiments/osdi2025/<experiment_name>/configs/`. For `sift1m.yaml`, use `sift1m`.
* `--output-dir <path_to_custom_output_directory>` (Optional): If provided, results (CSVs, plots) will be saved here. Default is `test/experiments/osdi2025/<experiment_name>/results/<config_name>/`.

### Kick the Tires (Quick Verification)

1.  **Run the Quickstart Example:**
    Tests Quake's core operations.

    ```bash
    python3 examples/quickstart.py
    ```

2.  **Run the `kick_the_tires` Benchmark:**
    Tests basic system functionality and performance with a dynamic workload.

    ```bash
    OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
        --experiment kick_the_tires \
        --config sift1m
    ```

    Outputs will be in `test/experiments/osdi2025/kick_the_tires/results/sift1m/`.

## Experiment Summaries and Outputs

The following table summarizes each experiment and its primary outputs. Output files are typically generated within the experiment's specific output directory (refer to the `--output-dir` option or its default value).

| Experiment Name        | Description                                                                                                  | Key Outputs (Illustrative Filenames)                                                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `aps_recall_targets`   | Evaluates Quake's Adaptive Partition Scanning (APS) against baselines for various recall targets.            | `aps_recall_targets_summary.csv`, `aps_recall_vs_metrics.png`                                                                                                       |
| `early_termination`    | Compares APS with early termination strategies like LAET and Auncel.                                         | `aps_et_results_<dataset>.csv`, `latency_plot_<dataset>.png`, Markdown table (printed to console)                                                                     |
| `kick_the_tires`       | Basic dynamic workload stress test focusing on latency, recall, and system state over operations.            | Per-index `results.csv` (e.g., `Quake10000/results.csv`), main `unified_plot.png`                                                                                       |
| `maintenance_ablation` | Evaluates the impact of Quake's maintenance operations under a dynamic workload, extending `kick_the_tires`. | Per-index `results.csv`, main `summary_table.csv`, `summary_table.md`, `unified_plot.png`, `time_breakdown.png`                                                        |
| `measure_skew_and_perf` | Measures data skew characteristics and their performance impact on a dataset.                                | Script-specific CSVs/plots detailing skew metrics and performance correlations.                                                                                       |
| `multi_level`          | Benchmarks Quake with multiple index levels against single-level configurations, varying `nprobe`.           | Per-index `results.csv` (e.g., `Quake5000/results.csv`), main `recall_vs_qps.png`                                                                                    |
| `numa_multi_query`     | Tests Quake's multi-query batch performance under different NUMA and worker configurations.                  | Per-index `results.csv`, main `<config_csv_name>.csv` (aggregate), `<config_csv_name_stem>_batch_latency.png`, `<config_csv_name_stem>_batch_recall.png`        |
| `numa_single_query`    | Tests Quake's single-query performance under different NUMA and worker configurations.                       | Per-index `results.csv`, main `<config_csv_name>.csv` (aggregate), `<config_csv_name_stem>_child_latency.png`, `<config_csv_name_stem>_recall.png`            |
| `vary_levels`          | Evaluates Quake with varying numbers of centroids in parent index levels.                                    | Per-level `results.csv` (e.g., `L<nc>/results.csv`), main `recall_vs_qps.png`                                                                                       |
| `workload_comparison`  | Compares Quake's performance across different types of synthetic or real-world workloads.                    | Script-specific CSVs/plots comparing workloads.                                                                                                                       |

**To run any of these, replace `<experiment_name>` and `<config_name_without_yaml>` in the general command structure.** For example, to run the `aps_recall_targets` experiment with its `sift1m` configuration:

```bash
OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
    --experiment aps_recall_targets \
    --config sift1m
```

Results, including CSV files and plots as detailed above, will be saved to the specified (or default) output directory. Check the terminal output for the exact location.

## Notes

* **Dataset Downloads:** The `quake.datasets.ann_datasets.load_dataset` utility will attempt to download datasets if they are not found locally (typically in a `data/` directory in the repository root). An internet connection is required for initial downloads.
* **Disk Space:** Some datasets (e.g., SIFT1B, MSTuring100M) are large and require significant disk space.
* **Execution Time:** Full experimental runs can be time-consuming, ranging from minutes for small configurations to many hours for large-scale evaluations. The `kick_the_tires` experiment with `sift1m` is designed for relatively quick verification.

<!-- end list -->

```
```