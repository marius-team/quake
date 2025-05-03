#!/usr/bin/env python3
"""
"""

import time
import yaml
import logging
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from quake.index_wrappers.quake import QuakeWrapper
from quake.index_wrappers.faiss_ivf import FaissIVF
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall
from quake.workload_generator import DynamicWorkloadGenerator, WorkloadEvaluator

logger = logging.getLogger(__name__)

def run_experiment(cfg_path: str, output_dir: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    data_dir = cfg["dataset"].get("path", "")
    vectors, queries, gt = load_dataset(cfg["dataset"]["name"], data_dir)

    metric = cfg["dataset"]["metric"]
    print(f"Dataset loaded: vectors {vectors.shape}, queries {queries.shape}")

    insert_ratio = cfg["workload_generator"]["insert_ratio"]
    delete_ratio = cfg["workload_generator"]["delete_ratio"]
    query_ratio = cfg["workload_generator"]["query_ratio"]
    update_batch_size = cfg["workload_generator"]["update_batch_size"]
    query_batch_size = cfg["workload_generator"]["query_batch_size"]
    number_of_operations = cfg["workload_generator"]["number_of_operations"]
    initial_size = cfg["workload_generator"]["initial_size"]
    cluster_size = cfg["workload_generator"]["cluster_size"]
    cluster_sample_distribution = cfg["workload_generator"]["cluster_sample_distribution"]
    query_cluster_sample_distribution = cfg["workload_generator"]["query_cluster_sample_distribution"]
    seed = cfg["workload_generator"]["seed"]

    overwrite_workload = cfg["overwrite"]["workload"]

    print(out)

    generator = DynamicWorkloadGenerator(
        workload_dir=out,
        base_vectors=vectors,
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

    if generator.workload_exists() and not overwrite_workload:
        print("Workload already exists, skipping generation.")
    else:
        # Generate the workload (operations are saved to disk along with a runbook)
        print("Generating workload...")
        generator.generate_workload()

    print("Evaluating workload...")

    for index_cfg in cfg["indexes"]:
        name = index_cfg["name"]
        print(f"\nEvaluating index: {name}")
        curr_results_dir = out / name
        curr_results_dir.mkdir(parents=True, exist_ok=True)
        evaluator = WorkloadEvaluator(workload_dir=out, output_dir=curr_results_dir)
        # Define search parameters for evaluation (e.g., number of neighbors and workers)
        build_params = index_cfg["build_params"]
        search_params = index_cfg["search_params"]

        if name == "Quake":
            # Initialize the Quake index
            index = QuakeWrapper()
        elif name == "IVF":
            index = FaissIVF()
        else:
            raise ValueError(f"Unknown index type: {name}")

        evaluator.evaluate_workload(name, index, build_params, search_params, do_maintenance=True)
        print(f"Evaluation completed for index {name}")