#!/usr/bin/env python
import argparse
import yaml
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time

# Import your vector search index and workload classes.
# (These should be defined in your project; adjust the import paths as needed.)
from quake.index_wrappers.quake import QuakeWrapper
from quake.datasets.ann_datasets import load_dataset
from quake import IndexBuildParams, SearchParams
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

def run_performance_test(index: QuakeWrapper,
                         build_params: IndexBuildParams,
                         search_params: SearchParams,
                         workload_generator: DynamicWorkloadGenerator,
                         evaluator: WorkloadEvaluator,
                         csv_output_path: Path):
    """
    Runs the workload on the index, collects per-operation metrics, and writes a CSV file.
    """
    print("Generating workload operations...")
    if not workload_generator.workload_exists():
        workload_generator.generate_workload()

    print("Evaluating workload...")
    results = evaluator.evaluate_workload(name="Quake",
                                          index=index,
                                          build_params=build_params,
                                          search_params=search_params)

    # create output directory if it does not exist
    csv_output_path = Path(csv_output_path)
    csv_output_path.mkdir(parents=True, exist_ok=True)

    csv_output_file = csv_output_path / "results.csv"

    # Save the results as a CSV file.
    df = pd.DataFrame(results)
    df.to_csv(csv_output_file, index=False)
    print(f"CSV results saved to {csv_output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Run Regression Test for the Vector Search Library"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/current.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name of the experiment')

    args = parser.parse_args()

    # Load configuration from YAML.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility.
    seed = config.get('seed', 1738)

    workload_name = args.config.split('/')[-1].split('.')[0]

    if args.name is not None:
        name = args.name
    else:
        name = workload_name

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset.
    dataset_name = config['dataset']['name']
    dataset_path = config['dataset'].get('path', None)
    print(f"Loading dataset {dataset_name} from {dataset_path} ...")
    vectors, queries, gt = load_dataset(dataset_name, dataset_path)

    build_params = {
        "nc": config['index'].get('nc', 1000),
        "metric": config['index'].get('metric', 'l2')
    }

    search_params = {
        "k": config['index']['search'].get('k', 10),
        "nprobe": config['index']['search'].get('nprobe', 10),
        "recall_target": config['index']['search'].get('recall_target', -1),
        "use_precomputed": config['index']['search'].get('use_precomputed', False),
        "batched_scan": config['index']['search'].get('batched_scan', False)
    }

    # Set up workload generator.
    workload_dir = config.get('workload_dir', 'workloads/experiment')
    workload_dir = Path(workload_dir) / workload_name
    workload_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(config.get('results_dir', 'results')) / workload_name / name

    workload_generator = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=vectors,
        metric=build_params['metric'],
        insert_ratio=config['workload'].get('insert_ratio', 0.3),
        delete_ratio=config['workload'].get('delete_ratio', 0.2),
        query_ratio=config['workload'].get('query_ratio', 0.5),
        update_batch_size=config['workload'].get('update_batch_size', 100),
        query_batch_size=config['workload'].get('query_batch_size', 100),
        number_of_operations=config['workload'].get('number_of_operations', 1000),
        initial_size=config['workload'].get('initial_size', 10000),
        cluster_size=config['workload'].get('cluster_size', 100),
        cluster_sample_distribution=config['workload'].get('cluster_sample_distribution', 'uniform'),
        queries=queries,
        seed=seed
    )

    # Set up evaluator.
    evaluator = WorkloadEvaluator(
        workload_dir=workload_dir,
        output_dir=output_dir
    )

    index = QuakeWrapper()

    # Run the performance test and output CSV.
    run_performance_test(index, build_params, search_params, workload_generator, evaluator, output_dir)

if __name__ == '__main__':
    main()