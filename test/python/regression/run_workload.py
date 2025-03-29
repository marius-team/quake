#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from quake.datasets.ann_datasets import load_dataset

# Import your project-specific modules.
from quake.index_wrappers.quake import QuakeWrapper
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")


def load_configuration(config_path: Path) -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file {config_path}: {e}")
        raise


def run_performance_test(
    index: QuakeWrapper,
    build_params: dict,
    search_params: dict,
    workload_generator: DynamicWorkloadGenerator,
    evaluator: WorkloadEvaluator,
    csv_output_path: Path,
) -> None:
    """
    Runs the workload on the index, collects per-operation metrics, and writes a CSV file.
    """
    logging.info("Generating workload operations...")
    if not workload_generator.workload_exists():
        workload_generator.generate_workload()

    logging.info("Evaluating workload...")
    results = evaluator.evaluate_workload(
        name="Quake", index=index, build_params=build_params, search_params=search_params
    )

    # Create output directory if it does not exist.
    csv_output_path.mkdir(parents=True, exist_ok=True)
    csv_output_file = csv_output_path / "results.csv"

    # Save results as CSV.
    df = pd.DataFrame(results)
    df.to_csv(csv_output_file, index=False)
    logging.info(f"CSV results saved to {csv_output_file}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Run Regression Test for the Vector Search Library")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default="results/current.csv", help="Output CSV file path")
    parser.add_argument("--name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results if set")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_configuration(config_path)

    # Set seeds for reproducibility.
    seed = config.get("seed", 1738)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Use the stem of the config file as a default experiment name.
    workload_name = config_path.stem
    name = args.name if args.name is not None else workload_name

    # Setup workload and output directories.
    workload_dir = Path(config.get("workload_dir", "workloads/experiment")) / workload_name
    workload_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.get("results_dir", "results")) / workload_name / name

    # Check if results already exist.
    results_csv = output_dir / "results.csv"
    if results_csv.exists() and not args.overwrite:
        logging.info(
            f"Results already exist in {output_dir}. Skipping run_regression_test. Use --overwrite to force rerun."
        )
        return

    # Load dataset.
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name")
    dataset_path = dataset_config.get("path", None)
    logging.info(f"Loading dataset {dataset_name} from {dataset_path} ...")
    vectors, queries, gt = load_dataset(dataset_name, dataset_path)

    # Extract index parameters from the configuration.
    build_params = {"nc": config["index"].get("nc", 1000), "metric": config["index"].get("metric", "l2")}
    search_params = {
        "k": config["index"]["search"].get("k", 10),
        "nprobe": config["index"]["search"].get("nprobe", 10),
        "recall_target": config["index"]["search"].get("recall_target", -1),
        "use_precomputed": config["index"]["search"].get("use_precomputed", False),
        "batched_scan": config["index"]["search"].get("batched_scan", False),
    }

    # Instantiate the workload generator.
    workload_generator = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=vectors,
        metric=build_params["metric"],
        insert_ratio=config["workload"].get("insert_ratio", 0.3),
        delete_ratio=config["workload"].get("delete_ratio", 0.2),
        query_ratio=config["workload"].get("query_ratio", 0.5),
        update_batch_size=config["workload"].get("update_batch_size", 100),
        query_batch_size=config["workload"].get("query_batch_size", 100),
        number_of_operations=config["workload"].get("number_of_operations", 1000),
        initial_size=config["workload"].get("initial_size", 10000),
        cluster_size=config["workload"].get("cluster_size", 100),
        cluster_sample_distribution=config["workload"].get("cluster_sample_distribution", "uniform"),
        queries=queries,
        seed=seed,
    )

    # Instantiate the evaluator.
    evaluator = WorkloadEvaluator(workload_dir=workload_dir, output_dir=output_dir)

    # Instantiate your index.
    index = QuakeWrapper()

    # Run the regression test.
    run_performance_test(index, build_params, search_params, workload_generator, evaluator, output_dir)


if __name__ == "__main__":
    main()
