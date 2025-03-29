#!/usr/bin/env python
"""
Quake Basic Example
================

This example demonstrates the basic functionality of Quake:
- Building an index from a sample dataset.
- Executing a search query on the index.
- Removing and adding vectors to the index.
- Performing maintenance on the index..

Ensure you have set up the conda environment (quake-env) and installed Quake prior to running this example.

Usage:
    python examples/quickstart.py
"""

import time

import torch

from quake import IndexBuildParams, QuakeIndex, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall


def main():
    print("=== Quake Basic Example ===")

    # Load a sample dataset (sift1m dataset as an example)
    dataset_name = "sift1m"
    print("Loading %s dataset..." % dataset_name)
    vectors, queries, gt = load_dataset(dataset_name)

    # Use a subset of the queries for this example
    ids = torch.arange(vectors.size(0))
    nq = 100
    queries = queries[:nq]
    gt = gt[:nq]

    ######### Build the index #########
    build_params = IndexBuildParams()
    build_params.nlist = 1024
    build_params.metric = "l2"
    print(
        "Building index with num_clusters=%d over %d vectors of dimension %d..."
        % (build_params.nlist, vectors.size(0), vectors.size(1))
    )
    start_time = time.time()
    index = QuakeIndex()
    index.build(vectors, ids, build_params)
    end_time = time.time()
    print(f"Build time: {end_time - start_time:.4f} seconds\n")

    ######### Search the index #########
    # Set up search parameters
    search_params = SearchParams()
    search_params.k = 10
    search_params.nprobe = 10
    # or set a recall target
    # search_params.recall_target = 0.9

    print(
        "Performing search of %d queries with k=%d and nprobe=%d..."
        % (queries.size(0), search_params.k, search_params.nprobe)
    )
    start_time = time.time()
    search_result = index.search(queries, search_params)
    end_time = time.time()
    recall = compute_recall(search_result.ids, gt, search_params.k)

    print(f"Mean recall: {recall.mean().item():.4f}")
    print(f"Search time: {end_time - start_time:.4f} seconds\n")

    ######### Remove vectors from index #########
    n_remove = 100
    print("Removing %d vectors from the index..." % n_remove)
    remove_ids = torch.arange(0, n_remove)
    start_time = time.time()
    index.remove(remove_ids)
    end_time = time.time()
    print(f"Remove time: {end_time - start_time:.4f} seconds\n")

    ######### Add vectors to index #########
    n_add = 100
    print("Adding %d vectors to the index..." % n_add)
    add_ids = torch.arange(vectors.size(0), vectors.size(0) + n_add)
    add_vectors = torch.randn(n_add, vectors.size(1))

    start_time = time.time()
    index.add(add_vectors, add_ids)
    end_time = time.time()
    print(f"Add time: {end_time - start_time:.4f} seconds\n")

    ######### Perform maintenance on the index #########
    print("Perform maintenance on the index...")
    start_time = time.time()
    maintenance_info = index.maintenance()
    end_time = time.time()

    print(f"Num partitions split: {maintenance_info.n_splits}")
    print(f"Num partitions merged: {maintenance_info.n_deletes}")
    print(f"Maintenance time: {end_time - start_time:.4f} seconds\n")

    ######### Save and load the index #########
    # Optionally save the index
    # index.save("quake_index")

    # Index can be loaded with:
    # index = QuakeIndex()
    # index.load("quake_index")


if __name__ == "__main__":
    main()
