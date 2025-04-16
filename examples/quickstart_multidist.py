#!/usr/bin/env python
"""
Quake Multi-Server Distributed Example
====================================

This example demonstrates the distributed functionality of Quake with multiple servers:
- Building an index on multiple servers
- Distributing queries across servers
- Comparing performance with single-server setup

Usage:
    python examples/quickstart_multidist.py
"""

import time
from typing import List

import torch

from quake import IndexBuildParams, QuakeIndex, SearchParams
from quake.datasets.ann_datasets import load_dataset
from quake.utils import compute_recall
from quake.distributedwrapper.distributedindex import DistributedIndex
from quake.distributedwrapper import distributed

# Server addresses
SERVERS = [
    "quake2:50052",
    "quake3:50053",
    "quake4:50054"
]

def run_single_server_test(dist_index: DistributedIndex, server_address: str, queries: torch.Tensor, gt: torch.Tensor, k: int, nprobe: int):
    """Run the test on a single server."""
    print(f"\n=== Single Server Test ({server_address}) ===")
    
    index, _, search_params = dist_index.get_index_and_params(server_address)
    
    print(f"Searching {queries.size(0)} queries with k={k}, nprobe={nprobe}...")
    start_time = time.time()
    search_result = index.search(queries, search_params)
    search_time = time.time() - start_time
    recall = compute_recall(search_result.ids, gt, k)
    
    print(f"Mean recall: {recall.mean().item():.4f}")
    print(f"Search time: {search_time:.4f} seconds")
    
    return search_time, recall.mean().item()

def run_distributed_test(dist_index: DistributedIndex, queries: torch.Tensor, gt: torch.Tensor, k: int, nprobe: int):
    """Run the test using the distributed index."""
    print("\n=== Distributed Test ===")
    print(f"Using {len(dist_index.server_addresses)} servers: {', '.join(dist_index.server_addresses)}")
    
    # Search
    print(f"Searching {queries.size(0)} queries with k={k}, nprobe={nprobe}...")
    start_time = time.time()
    result_ids = dist_index.search(queries)
    search_time = time.time() - start_time
    recall = compute_recall(result_ids, gt, k)
    
    print(f"Mean recall: {recall.mean().item():.4f}")
    print(f"Search time: {search_time:.4f} seconds")
    
    return search_time, recall.mean().item()

def main():
    # Load dataset
    print("Loading sift1m dataset...")
    vectors, queries, gt = load_dataset("sift1m")
    
    # Use a subset for testing
    ids = torch.arange(vectors.size(0))
    nq = 100  # More queries to better demonstrate distribution
    queries = queries[:nq]
    gt = gt[:nq]
    
    # Test parameters
    k = 10
    nprobe = 10

    # Build the distributed index first
    build_params_kw_args = {
        "nlist": 32,
        "metric": "l2"
    }
    search_params_kw_args = {
        "k": k,
        "nprobe": nprobe
    }
    dist_index = DistributedIndex(SERVERS, build_params_kw_args, search_params_kw_args)
    print("Building index on all servers...")
    start_time = time.time()
    dist_index.build(vectors, ids)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f} seconds")

    # Run single server tests
    single_server_results = []
    for server in SERVERS:
        results = run_single_server_test(dist_index, server, queries, gt, k, nprobe)
        single_server_results.append(results)
    
    # Run distributed test
    dist_results = run_distributed_test(dist_index, queries, gt, k, nprobe)
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print("Single Server Results:")
    for i, (server, (search_time, recall)) in enumerate(zip(SERVERS, single_server_results)):
        print(f"Server {i+1} ({server}):")
        print(f"  Search time: {search_time:.4f}s")
        print(f"  Recall: {recall:.4f}")
    
    print("\nDistributed Results:")
    print(f"Search time: {dist_results[0]:.4f}s")
    print(f"Recall: {dist_results[1]:.4f}")
    
    # Calculate speedup
    avg_single_search_time = sum(r[0] for r in single_server_results) / len(single_server_results)
    speedup = avg_single_search_time / dist_results[0]
    print(f"\nSearch speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main() 
