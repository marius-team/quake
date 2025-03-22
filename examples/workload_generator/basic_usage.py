#!/usr/bin/env python
"""
Basic usage example for the vector search workload generator and evaluator.
===================================

This script demonstrates how to generate a dynamic workload and then evaluate it
using the workload generator components. It creates a random base dataset and
query set, generates a workload of insert, delete, and query operations, and then
evaluates the performance of an example index (DynamicIVF).

Users can modify the parameters below to generate workloads suited to their needs.
"""
import math
from pathlib import Path

from quake.datasets.ann_datasets import load_dataset
from quake.index_wrappers.quake import QuakeWrapper
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator


def main():
    # Directories for workload and evaluation output
    workload_dir = Path("./workload_example")
    output_dir = Path("./evaluation_output")
    workload_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    base_vectors, queries, _ = load_dataset("sift1m")

    # Workload generation parameters
    insert_ratio = 0.9
    delete_ratio = 0.0
    query_ratio = 0.1
    update_batch_size = 10000
    query_batch_size = 10
    number_of_operations = 1000
    initial_size = 100000
    cluster_size = int(math.sqrt(base_vectors.shape[0]))
    cluster_sample_distribution = "uniform"  # or "skewed"
    query_cluster_sample_distribution = "uniform"  # or "skewed" / "skewed_fixed"
    metric = "l2"
    seed = 9299

    # Search parameters
    search_k = 10
    recall_target = 0.9

    # Create a DynamicWorkloadGenerator instance
    generator = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=base_vectors,
        metric=metric,
        insert_ratio=insert_ratio,
        delete_ratio=delete_ratio,
        query_ratio=query_ratio,
        update_batch_size=update_batch_size,
        query_batch_size=query_batch_size,
        number_of_operations=number_of_operations,
        initial_size=initial_size,
        cluster_size=cluster_size,
        cluster_sample_distribution=cluster_sample_distribution,
        queries=queries,
        query_cluster_sample_distribution=query_cluster_sample_distribution,
        seed=seed,
    )

    # Generate the workload (operations are saved to disk along with a runbook)
    print("Generating workload...")
    generator.generate_workload()

    # Create a WorkloadEvaluator instance and evaluate the workload
    evaluator = WorkloadEvaluator(workload_dir=workload_dir, output_dir=output_dir)

    print("Evaluating workload...")

    nc = 1000
    build_params = {"nc": nc, "metric": "l2"}
    search_params = {"k": search_k, "recall_target": recall_target}

    index = QuakeWrapper()

    results = evaluator.evaluate_workload(
        name="quake_test",
        index=index,
        build_params=build_params,
        search_params=search_params,
        do_maintenance=True,
    )

    print("Evaluation results:")
    print(results)


if __name__ == "__main__":
    main()
