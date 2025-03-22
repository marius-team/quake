import json
import tempfile
from pathlib import Path

import pytest
import torch

from quake.index_wrappers.quake import QuakeWrapper
from quake.workload_generator import DynamicWorkloadGenerator, UniformSampler, WorkloadEvaluator


# Create a small synthetic dataset for testing
@pytest.fixture
def synthetic_dataset():
    # For reproducibility
    torch.manual_seed(9299)

    base_vectors = torch.randn(10000, 128)
    queries = torch.randn(100, 128)

    workload_dir = Path(tempfile.gettempdir()) / "workload_test"

    return base_vectors, queries, workload_dir


def test_workload_generation(synthetic_dataset):
    base_vectors, queries, workload_dir = synthetic_dataset
    workload_dir.mkdir(exist_ok=True)

    # Set up a simple workload configuration.
    # These ratios must sum to 1.
    insert_ratio = 0.3
    delete_ratio = 0.2
    query_ratio = 0.5
    update_batch_size = 50
    query_batch_size = 10
    number_of_operations = 20
    initial_size = 200
    cluster_size = 50
    cluster_sample_distribution = "uniform"
    query_cluster_sample_distribution = "uniform"
    seed = 1234

    # Initialize the workload generator with the synthetic data
    workload_gen = DynamicWorkloadGenerator(
        workload_dir=workload_dir,
        base_vectors=base_vectors,
        metric="l2",
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

    # For testing, use a simple sampler (could also test StratifiedClusterSampler)
    workload_gen.sampler = UniformSampler()

    # Generate the workload
    workload_gen.generate_workload()

    # Check that the workload directory contains expected files
    runbook_path = workload_dir / "runbook.json"
    operations_dir = workload_dir / "operations"
    assert runbook_path.exists(), "Runbook file was not created."
    assert operations_dir.exists(), "Operations directory was not created."

    # Load and inspect the runbook
    with open(runbook_path, "r") as f:
        runbook = json.load(f)

    # Check that runbook has keys for 'initialize' and 'operations'
    assert "initialize" in runbook, "Runbook missing 'initialize' key."
    assert "operations" in runbook, "Runbook missing 'operations' key."

    # Basic checks: number of operations should be <= number_of_operations
    op_count = len(runbook["operations"])
    assert op_count <= number_of_operations, "Too many operations generated."


def test_workload_evaluation(synthetic_dataset):
    base_vectors, queries, workload_dir = synthetic_dataset
    nc = 100
    build_params = {"nc": nc, "metric": "l2"}

    index = QuakeWrapper()

    evaluator = WorkloadEvaluator(
        workload_dir=workload_dir,
        output_dir=workload_dir,
    )

    # Evaluate the workload
    results = evaluator.evaluate_workload(
        name="quake_test",
        index=index,
        build_params=build_params,
        search_params={"k": 5, "nprobe": 1},
        do_maintenance=True,
    )

    # Basic checks on the returned results
    assert isinstance(results, list), "Results should be a list."
    for result in results:
        # For query operations, recall should be a float between 0 and 1
        if result["operation_type"] == "query":
            assert 0.0 <= result["recall"] <= 1.0, "Recall out of bounds."
