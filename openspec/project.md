# Project Context

## Purpose
Quake is a high-performance library for **dynamic approximate nearest neighbor (ANN) search**. It uses an IVF (inverted file) partitioned index structure with support for real-time updates, adaptive search, and automatic cost-based maintenance. The project targets production vector search workloads where the index must be updated continuously without full rebuilds.

Key capabilities:
- **Dynamic indexing** — build, add, remove, and auto-maintain the index in real time
- **Adaptive search** — specify a recall target and Quake auto-tunes the number of partitions to scan
- **NUMA-aware execution** — partitions are allocated on specific NUMA nodes; worker threads are pinned to local cores
- **Multi-threaded scan** — parallel partition scanning with per-core worker threads and job queues
- **PyTorch integration** — all data flows through `torch::Tensor` for seamless ML interop

## Tech Stack
- **C++17** — core index implementation
- **Python 3** — user-facing API, wrappers, evaluation scripts
- **PyTorch / libtorch** — tensor types, Python bindings infrastructure
- **Faiss** — distance computation, heap operations, IVF infrastructure (third-party)
- **OpenBLAS** — BLAS operations for batch distance computation
- **pybind11** — C++/Python bindings (via `wrap.cpp`)
- **CMake 3.24+** — build system
- **Conda** — environment management
- **Google Test** — C++ unit and integration tests
- **pytest** — Python tests
- **Optional: libnuma** — NUMA-aware memory allocation (enabled via `QUAKE_USE_NUMA`)
- **Optional: CuVS/CUDA** — GPU-accelerated clustering

## Project Conventions

### Code Style
- C++ follows the project's existing formatting (no explicit formatter config found — match surrounding code)
- C++ headers in `src/cpp/include/`, implementations in `src/cpp/src/`
- Header guards use `#pragma once`
- Class names: `PascalCase` (e.g., `QuakeIndex`, `PartitionManager`)
- Method names: `snake_case` (e.g., `split_partitions`, `add_vectors`)
- Member variables: trailing underscore (e.g., `codes_`, `ids_`, `partition_manager_`)
- Python code follows standard PEP 8 conventions

### Architecture Patterns
- **Central orchestrator**: `QuakeIndex` owns and delegates to `PartitionManager`, `QueryCoordinator`, and `MaintenancePolicy`
- **Partitioned storage**: `DynamicInvertedLists` holds a map of `IndexPartition` objects (one per partition)
- **NUMA locality**: partition data (`codes_`, `ids_`) allocated via `quake_alloc()` on specific NUMA nodes; scan workers pinned to local cores
- **Cost-based maintenance**: `MaintenanceCostEstimator` + `HitCountTracker` drive split/merge/refine decisions
- **Multi-level indexing**: optional hierarchical parent index over centroids for large-scale datasets
- **Swap-last deletion**: removing a row swaps the last element into the gap to avoid shifting

### Testing Strategy
- C++ tests use Google Test, located in `test/cpp/`
- Python tests use pytest, located in `test/python/`
- Build tests with CMake (`-DBUILD_TESTS=ON`)
- Run C++ tests via CTest (`ctest` or individual binaries in `cmake-build-*/bin/`)
- Run Python tests via `pytest test/python/`

### Git Workflow
- Main branch for stable code
- Feature branches for changes
- Submodules for third-party dependencies (`git submodule update --init --recursive` after clone)

## Domain Context

### IVF Index Structure
Vectors are clustered into partitions using k-means. Each partition has a centroid. At search time, the query is compared to centroids to find the `nprobe` nearest partitions, which are then scanned exhaustively for the top-k nearest neighbors.

### Key Classes
| Class | Header | Role |
|-------|--------|------|
| `QuakeIndex` | `quake_index.h` | Central orchestrator: build, search, add, remove, maintenance, save/load |
| `PartitionManager` | `partition_manager.h` | Manages IVF structure: partition init, add/remove, split/merge/refine |
| `QueryCoordinator` | `query_coordinator.h` | Search execution: serial, batched, and worker scan modes |
| `MaintenancePolicy` | `maintenance_policies.h` | Cost-based maintenance: hit tracking, split/merge detection |
| `DynamicInvertedLists` | `dynamic_inverted_list.h` | NUMA-aware partition storage, ID-to-location mapping |
| `IndexPartition` | `index_partition.h` | Single partition: contiguous `codes_`/`ids_` arrays, NUMA allocation |
| `Clustering` | `clustering.h` | k-means implementations (CPU, optional GPU) |

### Configuration Structs (common.h)
- `IndexBuildParams` — nlist, niter, metric, num_workers, use_numa, use_gpu
- `SearchParams` — k, nprobe, recall_target, batched_scan
- `MaintenancePolicyParams` — window_size, min_partition_size, alpha, refinement_iterations

### Current Limitations
1. Metrics: L2 (Euclidean) and IP (inner product) only
2. Data types: float32 vectors, int64 IDs only
3. CPU-only search (GPU used only for clustering)
4. No per-vector attribute storage or filtering
5. Single-node only

## Important Constraints
- **Zero-overhead principle**: features that are not used must not impose any cost on the default path (e.g., unfiltered search must not be slower after adding attribute support)
- **NUMA locality**: scan-time hot paths must read only from NUMA-local memory — no cross-node accesses in the inner loop
- **Backward compatibility**: changes to save/load must handle indexes written by older versions
- **Single writer**: mutations are mutex-guarded; only one write operation at a time
- **PyTorch tensor interface**: all data crosses the C++/Python boundary as `torch::Tensor`

## External Dependencies
- **Faiss** (third-party, vendored via submodule) — distance computation, heaps, IVF foundation
- **PyTorch** — tensor types and Python integration
- **OpenBLAS** — BLAS routines for batch distance computation
- **pybind11** — C++/Python bindings
- **libnuma** (optional) — NUMA-aware allocation on Linux
- **CuVS/CUDA** (optional) — GPU-accelerated k-means

## File Organization
```
src/
├── cpp/
│   ├── include/              # C++ headers
│   ├── src/                  # C++ implementations
│   ├── bindings/wrap.cpp     # pybind11 bindings
│   └── third_party/          # Faiss, etc.
└── python/
    ├── __init__.py
    ├── index_wrappers/       # QuakeWrapper, FaissIVF, HNSW, DiskANN wrappers
    └── ...
test/
├── cpp/                      # Google Test C++ tests
└── python/                   # pytest Python tests
docs/                         # Sphinx documentation, ARCHITECTURE.md
openspec/                     # Spec-driven design (proposals, specs, tasks)
examples/                     # Usage examples (quickstart.py)
environments/                 # Conda environment YAML files
```
