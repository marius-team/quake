# Quake Architecture Overview

Quake is a high-performance library for **dynamic approximate nearest neighbor (ANN) search**. It uses a partitioned index structure (IVF-style) with support for real-time updates, adaptive search, and automatic maintenance.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Python API Layer                               │
│   (quake module, QuakeWrapper, IndexWrapper)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PyBind11 Bindings                                 │
│   (wrap.cpp - exposes C++ classes to Python via PyTorch tensors)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QuakeIndex                                     │
│   (Central orchestrator: build, search, add, remove, maintenance)           │
├──────────────────┬───────────────────┬──────────────────────────────────────┤
│ PartitionManager │ QueryCoordinator  │ MaintenancePolicy                    │
└──────────────────┴───────────────────┴──────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Storage & Computation Layer                          │
│   DynamicInvertedLists, IndexPartition, Clustering, List Scanning           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. QuakeIndex (`quake_index.h`)

The **central class** that orchestrates all index operations. It provides the main API for:

| Operation | Description |
|-----------|-------------|
| `build()` | Create an index from vectors using k-means clustering |
| `search()` | Execute k-NN queries with configurable nprobe or recall target |
| `add()` | Insert new vectors into appropriate partitions |
| `remove()` | Delete vectors by ID |
| `maintenance()` | Trigger partition splits/merges for optimal performance |
| `save()`/`load()` | Persist and restore the index |

**Key Members:**
- `parent_`: Optional hierarchical parent index over centroids (multi-level indexing)
- `partition_manager_`: Manages the partitioned data structure
- `query_coordinator_`: Handles search query execution
- `maintenance_policy_`: Decides when and how to restructure partitions

---

### 2. PartitionManager (`partition_manager.h`)

Manages the **partitioned (IVF) structure** of the index. Responsibilities include:

- **Partition initialization**: Create partitions from k-means clustering
- **Vector assignment**: Assign vectors to their nearest centroid/partition
- **Dynamic operations**: Add/remove vectors, split/merge/refine partitions
- **NUMA distribution**: Distribute partitions across NUMA nodes for locality

**Key Methods:**
```cpp
void init_partitions(parent, partitions);      // Initialize from clustering
ModifyTimingInfo add(vectors, ids, assignments); // Add vectors to partitions
ModifyTimingInfo remove(ids);                   // Remove vectors by ID
Clustering split_partitions(partition_ids);     // Split overloaded partitions
void delete_partitions(partition_ids);          // Merge/delete underutilized partitions
void refine_partitions(partition_ids);          // Re-cluster for better quality
```

---

### 3. QueryCoordinator (`query_coordinator.h`)

The **query execution engine** that orchestrates search operations. Features:

- **Multi-threaded scanning**: Parallel partition scanning with worker threads
- **NUMA-aware execution**: Pin workers to cores and use local memory
- **Batched scanning**: Efficient processing of multiple queries together
- **Adaptive search (APS)**: Dynamically determine partitions to scan based on recall target

**Scan Modes:**
| Mode | Description |
|------|-------------|
| `serial_scan` | Sequential single-threaded scanning |
| `batched_serial_scan` | Batch queries by partition for efficiency |
| `worker_scan` | Parallel scanning with dedicated worker threads |

**Key Structures:**
- `ScanJob`: Encapsulates a scan task (partition, query, k)
- `CoreResources`: Per-worker buffers and state
- `NUMAResources`: NUMA-local memory and job queues

---

### 4. MaintenancePolicy (`maintenance_policies.h`)

Implements **cost-based maintenance** to keep the index optimized over time:

- **Hit tracking**: Monitor which partitions are accessed during queries
- **Split detection**: Identify overloaded partitions that slow down search
- **Merge detection**: Find underutilized partitions that waste resources
- **Local refinement**: Re-cluster nearby partitions for better boundaries

**Key Components:**
- `HitCountTracker`: Sliding window tracker for partition access patterns
- `MaintenanceCostEstimator`: Estimates scan latency and maintenance costs

---

### 5. DynamicInvertedLists (`dynamic_inverted_list.h`)

A **dynamic, NUMA-aware inverted list** implementation that extends Faiss's InvertedLists:

- Stores vectors and IDs in a map of `IndexPartition` objects
- Supports efficient insertion, update, and deletion
- Provides NUMA-aware memory allocation
- Maintains ID-to-location mapping for fast lookups

---

### 6. IndexPartition (`index_partition.h`)

Represents a **single partition** of the index:

- Manages contiguous memory blocks for vectors (codes) and IDs
- Supports NUMA-local allocation
- Tracks delta changes for maintenance decisions
- Auto-resizes based on configurable thresholds

---

### 7. Clustering (`clustering.h`)

Provides **k-means clustering** implementations:

- `kmeans_cpu()`: CPU-based k-means using Faiss
- `kmeans_cuvs_sample_and_predict()`: GPU-accelerated k-means (optional)
- `kmeans_refine_partitions()`: Iterative refinement of existing partitions

---

### 8. List Scanning (`list_scanning.h`)

Low-level **distance computation and top-k selection**:

- SIMD-optimized distance calculations (AVX512 support)
- `TopkBuffer`: Efficient top-k result aggregation
- BLAS-accelerated batch distance computation
- Support for L2 (Euclidean) and IP (inner product) metrics

---

## Python API Layer

### Module Structure (`src/python/`)

```
src/python/
├── __init__.py              # Imports C++ bindings
├── utils.py                 # Utility functions
├── workload_generator.py    # Query workload generation
├── datasets/                # Dataset loaders
└── index_wrappers/
    ├── wrapper.py           # Abstract IndexWrapper base class
    ├── quake.py             # QuakeWrapper implementation
    ├── faiss_ivf.py         # Faiss IVF wrapper (comparison)
    ├── faiss_hnsw.py        # Faiss HNSW wrapper (comparison)
    └── diskann.py           # DiskANN wrapper (comparison)
```

### Key Python Classes

**`QuakeWrapper`** (quake.py): High-level Python interface
```python
wrapper = QuakeWrapper()
wrapper.build(vectors, nc=1024, metric="l2")
ids, dists = wrapper.search(queries, k=10, nprobe=32)
wrapper.add(new_vectors)
wrapper.remove(ids_to_remove)
wrapper.maintenance()
```

---

## Configuration Parameters

### IndexBuildParams
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nlist` | 0 | Number of partitions (0 = flat index) |
| `niter` | 5 | K-means iterations |
| `metric` | "l2" | Distance metric ("l2" or "ip") |
| `num_workers` | 0 | Worker threads for search |
| `use_numa` | false | Enable NUMA-aware allocation |
| `use_gpu` | false | Use GPU for clustering |

### SearchParams
| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 1 | Number of neighbors to return |
| `nprobe` | 1 | Partitions to probe |
| `recall_target` | -1 | Adaptive search recall target (0-1) |
| `batched_scan` | false | Enable batched scanning |

### MaintenancePolicyParams
| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 1000 | Query window for hit tracking |
| `min_partition_size` | 32 | Minimum partition size |
| `alpha` | 0.9 | Maintenance policy weight |
| `refinement_iterations` | 3 | Local refinement iterations |

---

## Data Flow

### Build Flow
```
Vectors + IDs
     │
     ▼
┌─────────────┐
│  K-Means    │ ── Cluster vectors into nlist partitions
└─────────────┘
     │
     ▼
┌─────────────────────┐
│  PartitionManager   │ ── Initialize DynamicInvertedLists
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  QueryCoordinator   │ ── Initialize workers (if parallel)
└─────────────────────┘
```

### Search Flow
```
Query Vectors
     │
     ▼
┌─────────────────────┐
│  Parent Index       │ ── Find top-nprobe nearest centroids
│  (if hierarchical)  │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  QueryCoordinator   │ ── Dispatch scan jobs to workers
└─────────────────────┘
     │
     ├──► Worker 1 ──► Scan Partition A
     ├──► Worker 2 ──► Scan Partition B
     └──► Worker N ──► Scan Partition C
     │
     ▼
┌─────────────────────┐
│  Result Aggregation │ ── Merge top-k from all scanned partitions
└─────────────────────┘
     │
     ▼
SearchResult (ids, distances)
```

---

## Key Design Decisions

1. **PyTorch Integration**: Uses torch::Tensor for seamless ML workflow integration
2. **Faiss Foundation**: Builds on Faiss infrastructure (distance functions, heap operations)
3. **Dynamic Updates**: Full support for real-time add/remove without rebuild
4. **Adaptive Search**: Automatically adjusts nprobe to meet recall targets
5. **Cost-Based Maintenance**: Data-driven decisions for partition restructuring
6. **NUMA Awareness**: Optimized for multi-socket systems with local memory access
7. **Multi-Level Indexing**: Optional hierarchical index for very large datasets

---

## File Organization

```
src/
├── cpp/
│   ├── include/              # Header files
│   │   ├── quake_index.h     # Main index class
│   │   ├── partition_manager.h
│   │   ├── query_coordinator.h
│   │   ├── maintenance_policies.h
│   │   ├── dynamic_inverted_list.h
│   │   ├── index_partition.h
│   │   ├── clustering.h
│   │   ├── list_scanning.h
│   │   ├── common.h          # Types, constants, utilities
│   │   └── ...
│   ├── src/                  # Implementation files
│   │   ├── quake_index.cpp
│   │   ├── partition_manager.cpp
│   │   ├── query_coordinator.cpp
│   │   └── ...
│   ├── bindings/
│   │   └── wrap.cpp          # PyBind11 bindings
│   └── third_party/          # External dependencies
└── python/
    ├── __init__.py
    ├── index_wrappers/
    └── ...
```

---

## Dependencies

- **PyTorch**: Tensor operations and Python bindings
- **Faiss**: Distance computation, heap operations, IVF infrastructure
- **OpenBLAS**: BLAS operations for batch distance computation
- **pybind11**: C++/Python bindings
- **Optional: CuVS/CUDA**: GPU-accelerated clustering
- **Optional: libnuma**: NUMA-aware memory allocation
