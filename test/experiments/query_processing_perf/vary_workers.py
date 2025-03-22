import subprocess
from pathlib import Path

import pandas as pd
import yaml

from quake import MaintenancePolicyParams
from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator


def create_maintenance_params(m_config):
    """
    Create a MaintenancePolicyParams object from a maintenance configuration.
    Here we assume that the YAML thresholds are given in milliseconds and convert them to nanoseconds.
    """
    m_params = MaintenancePolicyParams()
    if "delete_threshold" in m_config:
        m_params.delete_threshold_ns = m_config["delete_threshold"]
    if "split_threshold" in m_config:
        m_params.split_threshold_ns = m_config["split_threshold"]
    if "refinement_radius" in m_config:
        m_params.refinement_radius = m_config["refinement_radius"]
    if "refinement_iterations" in m_config:
        m_params.refinement_iterations = m_config["refinement_iterations"]
    if "enable_delete_rejection" in m_config:
        m_params.enable_delete_rejection = m_config["enable_delete_rejection"]

    return m_params


def run_experiment_for_config(curr_config, config):
    print(f"\n=== Running maintenance config: {curr_config['name']} ===")

    # Load dataset using the configuration.
    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"].get("path", "data")
    vectors, queries, gt = load_dataset(dataset_name, dataset_path)

    # Set up workload parameters.
    workload_cfg = config["workload"]
    base_experiment_name = config["name"]
    workload_dir = Path(config.get("workload_dir", "workloads")) / base_experiment_name
    results_dir = Path(config.get("results_dir", "results")) / base_experiment_name / curr_config["name"]
    workload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the dynamic workload generator.
    workload_gen = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=vectors,
        metric=config["index"]["metric"],
        insert_ratio=workload_cfg["insert_ratio"],
        delete_ratio=workload_cfg["delete_ratio"],
        query_ratio=workload_cfg["query_ratio"],
        update_batch_size=workload_cfg["update_batch_size"],
        query_batch_size=workload_cfg["query_batch_size"],
        number_of_operations=workload_cfg["number_of_operations"],
        initial_size=workload_cfg["initial_size"],
        cluster_size=workload_cfg["cluster_size"],
        cluster_sample_distribution=workload_cfg["cluster_sample_distribution"],
        queries=queries,
        seed=config.get("seed", 1738),
    )

    # Generate the workload if it doesn't already exist.
    if not workload_gen.workload_exists():
        print("Generating workload...")
        workload_gen.generate_workload()
    else:
        print("Workload already exists; reusing generated workload.")

    # Build a fresh index.
    index = QuakeWrapper()
    build_params = {
        "nc": config["index"].get("nc", 1024),
        "metric": config["index"]["metric"],
        "num_workers": curr_config["build"]["n_workers"],
    }

    # Initialize maintenance policy if enabled.
    do_maint = config.get("do_maintenance", False)

    # Set up and run the workload evaluator.
    evaluator = WorkloadEvaluator(workload_dir=workload_dir, output_dir=results_dir)
    search_params = config["index"]["search"]
    search_params["n_threads"] = curr_config["search"]["n_threads"]
    results = evaluator.evaluate_workload(
        name="quake_test",
        index=index,
        build_params=build_params,
        search_params=search_params,
        do_maintenance=do_maint,
        batch=True,
    )

    return results


def run_experiments_and_compare():
    # Load the overall configuration.
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / Path("configs/sift1m_read_only.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_results_dir = Path(config.get("results_dir", "results")) / config["name"]
    base_results_dir.mkdir(parents=True, exist_ok=True)

    overwrite = config.get("overwrite", False)

    experiments_results = {}
    # Loop over each maintenance configuration.
    for curr_config in config.get("configs", []):
        result_file = base_results_dir / curr_config["name"] / "results.csv"

        if result_file.exists() and not overwrite:
            print(f"Results already exist for maintenance config '{curr_config['name']}'. Loading results...")
            df = pd.read_csv(result_file)
            experiments_results[curr_config["name"]] = df
        else:
            results = run_experiment_for_config(curr_config, config)
            df = pd.DataFrame(results)
            result_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(result_file, index=False)
            experiments_results[curr_config["name"]] = df

    # After running experiments (or loading results), call the existing compare_results.py script
    # to generate detailed per-operation plots and the aggregate matrix.
    results_root = Path(config.get("results_dir", "results"))
    output_aggregate = results_root / "aggregate_matrix.png"
    print("Generating detailed comparison plots using compare_results.py ...")
    subprocess.run(
        [
            "python",
            "test/python/regression/compare_results.py",
            "--results_dir",
            str(results_root),
            "--plot_type",
            "both",
            "--output_aggregate",
            str(output_aggregate),
        ],
        check=True,
    )


if __name__ == "__main__":
    run_experiments_and_compare()
