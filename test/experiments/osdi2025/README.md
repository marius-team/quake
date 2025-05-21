
# Quake OSDI 2025 Artifact Evaluation

This artifact provides the experimental setup for the Quake vector search system, as described in the OSDI 2025 submission.

## Structure

The experiments are organized as follows:



```plaintext
quake/test/experiments/osdi2025/
├── experiment_runner.py     \# Main script to run any experiment
├── experiment_utils.py      \# Shared functionality across experiments
├── README.md                \# This file
├── paper/                   \# PDF of the paper
│   ├── Quake.pdf            \# Original Submission
│   ├── Quake-Revision.pdf   \# Revised Version
├── <experiment_name>/       \# Directory for each specific experiment
│   ├── configs/             \# YAML configuration files for the experiment
│   │   └── <dataset_config>.yaml
│   └── run.py               \# Python script to execute the experiment logic
└── ...                      \# Other experiment directories
```

Each experiment directory (e.g., `aps_recall_targets`, `early_termination`) contains its specific logic in `run.py` and configurations in its `configs/` subdirectory. The `experiment_utils.py` module provides shared functionality for loading data, configurations, preparing indexes, and plotting, aiming to simplify individual experiment scripts.

## Installation

**Requirements:**
- Python 3.9 or later
- Linux (Tested on Ubuntu 22.04)
- Conda package manager

### Install from Scratch

This will install Quake, the baselines and all necessary dependencies.

Run `sudo install.sh` 

This will:
- Install required system packages (GCC/G++ 11, CMake, Miniconda, Intel oneAPI, etc.)
- Set up a Conda environment (quake-env) with all Python dependencies
- Build and install the Quake library and baselines (Faiss, SCANN, DiskANN, SVS)
- Automatically clone, build, and install all required repositories

### Quake Installation Only (No Baselines)

To just evaluate Quake use this.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/marius-team/quake.git
    cd quake
    git checkout osdi2025
    git submodule update --init --recursive
    
    ```

2.  **Set Up the Conda Environment:**
    Create and activate the environment using the provided YAML file:
    ```bash
    conda env create -f environments/ubuntu-cpu/conda.yaml
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

* `OMP_NUM_THREADS=1`: Recommended for consistent performance measurement as OMP interferes with our custom parallelism.
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

### Quick Reproduction

These experiments should take less than 10 minutes to run in total if the datasets/indexes have been prepared ahead of time. Otherwise it will take 30-60 minutes on the first run.

1. Reproduction of APS (Figure 6 in `paper/Quake.pdf`)

   This compares an Oracle against APS at various recall targets. 

    ```bash
    OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
        --experiment aps_recall_targets \
        --config sift1m
    ```

2. Reproduction of NUMA-aware searching (Figure 5 in `paper/Quake.pdf`)

    This measures query latency and varies the number of workers used in searching and whether partitions are pinned to NUMA regions.

    ```bash
    OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
        --experiment numa_single_query \
        --config msturing10m
    ```

3. Maintenance Ablation Study (Table 6 in `paper/Quake-Revision.pdf`)

    This removes critical pieces of the Quake maintenance policy and measure its effect on workload performance.

    ```bash
    OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
        --experiment maintenance_ablation \
        --config sift1m
    ```
    

### Full Reproduction (In progress)

## Experiment Summaries and Outputs

The following table summarizes each experiment and their figure/table in paper. Output files are generated within the experiment's specific output directory (refer to the `--output-dir` option or its default value).

Some experiments are added as revisions for the camera ready. See `paper/Quake.pdf` for the initial submission and `paper/Quake-Revision.pdf` for the revised version

| Experiment Name        | Description                                                                                        | Reproduces in Paper       |
|------------------------|----------------------------------------------------------------------------------------------------|---------------------------|
| `aps_recall_targets`   | Evaluates Quake's Adaptive Partition Scanning (APS) against baselines for various recall targets.  | Figure 6 in Quake         |
| `early_termination`    | Compares APS with early termination strategies like LAET and Auncel.                               | Table 5 in Quake-Revision |
| `kick_the_tires`       | Basic dynamic workload stress test focusing on latency, recall, and system state over operations.  |                           |
| `maintenance_ablation` | Evaluates the impact of Quake's maintenance operations under a dynamic workload.                   | Table 6 in Quake-Revision |
| `numa_multi_query`     | Tests Quake's multi-query batch performance under different NUMA and worker configurations.        | Revision Experiment       |
| `numa_single_query`    | Tests Quake's single-query performance under different NUMA and worker configurations.             | Figure 5 in Quake         |

**To run any of these, replace `<experiment_name>` and `<config_name_without_yaml>` in the general command structure.** For example, to run the `aps_recall_targets` experiment with its `sift1m` configuration:

```bash
OMP_NUM_THREADS=1 python3 -m test.experiments.osdi2025.experiment_runner \
    --experiment aps_recall_targets \
    --config sift1m
```

Results, including CSV files and plots as detailed above, will be saved to the specified (or default) output directory. Check the terminal output for the exact location.

## Notes

* **Dataset Downloads:** The `quake.datasets.ann_datasets.load_dataset` utility will attempt to download datasets if they are not found locally (typically in a `data/` directory in the repository root). An internet connection is required for initial downloads.
* **Execution Time:** Full experimental runs can be time-consuming, ranging from minutes for small configurations to many hours for large-scale evaluations. The `kick_the_tires` experiment with `sift1m` is designed for relatively quick verification.

<!-- end list -->
