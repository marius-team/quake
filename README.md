# Quake: Query-Adaptive KNN Index

Quake is a C++ library (with Python bindings) for dynamic, high‑performance approximate nearest neighbor (ANN) search. Its core operations—building a dynamic index, adaptive search, real‑time updates, and automatic maintenance—ensure high-throughput updates and low-latency queries without manual tuning.

---
## Key Advantages

- **Dynamic Indexing:**  
  Build, update and automatically maintain the index in real time without full rebuilds.

- **Adaptive Search:**  
  Specify a recall target (e.g. 90% recall) and let Quake automatically choose the number of partitions to scan.

- **High Performance:**  
  Leveraging multi‑threading, SIMD, and NUMA, Quake delivers both low latency and high throughput.

- **PyTorch Integration:**  
  Directly work with PyTorch tensors for easy integration into machine learning workflows.

---

## Installation

### Prerequisites

- **C++ Compiler:** Must support C++17 (GCC 8+, Clang 10+, or MSVC 2019+)
- **CMake:** Version 3.16 or later
- **PyTorch 2.0+:** (Download from [PyTorch.org](https://pytorch.org/))
- **Python 3:** Required for Python bindings

### Build Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/marius-team/quake.git
   cd quake
   ```

2. **Install**

   ```bash
   pip install .
   ```"

3**(C++ Βuild) Configure the build**
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
      make bindings -j$(nproc)
   ```

   To enable GPU, NUMA, or AVX512 support, add:
   ```
   cmake -DCMAKE_BUILD_TYPE=Release -DQUAKE_ENABLE_GPU=ON -DQUAKE_USE_NUMA=ON -DQUAKE_USE_AVX512=ON ..
   ```

### Using Quake

Quake’s core operations are simple: build an index, search it, update it, and maintain it.

#### C++ Usage

1. **Build the Index**
``` cpp
#include "quake_index.h"

auto index = std::make_shared<QuakeIndex>();

// Configure build parameters
auto build_params = std::make_shared<IndexBuildParams>();
build_params->nlist = 1000;      // Number of partitions (clusters)
build_params->metric = "l2";     // Use Euclidean (L2) distance
build_params->niter = 5;         // K‑means iterations for clustering

// Prepare your data
Tensor vectors = torch::randn({100000, 128});
Tensor ids = torch::arange(100000, torch::kInt64);

// Build the index
index->build(vectors, ids, build_params);
```

2. **Search the Index**

Query the index. Specify either a fixed number of partitions (nprobe) or a recall target to let Quake adapt:
``` cpp
// Configure search parameters
auto search_params = std::make_shared<SearchParams>();
search_params->k = 10;            // Top‑10 neighbors
search_params->recall_target = 0.9;    // Aim for 90% recall
// Alternatively, set search_params->nprobe = 50;

// Query vector(s)
Tensor queries = torch::randn({100, 128});

// Perform search
auto result = index->search(queries, search_params);

// result->ids: [100, 10] tensor of neighbor IDs
// result->distances: [100, 10] tensor of corresponding distances
```

3. **Updating the Index**

Add vectors
``` cpp
Tensor new_vectors = torch::randn({1000, 128});
Tensor new_ids = torch::arange(100000, 101000, torch::kInt64);
index->add(new_vectors, new_ids);
```

Remove vectors
``` cpp
Tensor remove_ids = new_ids.slice(/*dim=*/0, 0, 500);
index->remove(remove_ids);
```

4. **Maintenance**
   
Call maintenance to trigger dynamic partition adjustments (e.g. splitting or merging partitions):
``` cpp
auto timing_info = index->maintenance();
// Use timing_info to see split/delete statistics if desired.
```

5. **Saving and Loading**
``` cpp
// Save the index
index->save("path/to/index_dir");

// Later or in another program, load the index:
auto new_index = std::make_shared<QuakeIndex>();
new_index->load("path/to/index_dir");
```

#### Python Usage

The Python API mirrors the C++ usage:

``` python
import quake
import torch

# Build the index
index = quake.QuakeIndex()
build_params = quake.IndexBuildParams()
build_params.nlist = 1000
build_params.metric = "l2"
vectors = torch.randn(100000, 128)
ids = torch.arange(100000, dtype=torch.int64)
index.build(vectors, ids, build_params)

# Search the index
search_params = quake.SearchParams()
search_params.k = 10
search_params.nprobe = 50  # or set recall_target instead
queries = torch.randn(100, 128)
results = index.search(queries, search_params)
print("Neighbor IDs:", results.ids)
print("Distances:", results.distances)
```

####  Key Limitations (for now)

Quake currently has the following limitations which will be addressed in future development.

1. **Metrics:** Only supports “l2” (Euclidean) and “ip” (inner product).
2. **Data Types:** Vectors must be float32; IDs must be int64.
3. **CPU-only Search:** Even if built with GPU, search is performed on the CPU.
4. **Vectors Only:** Currently we don't support storing or filtering on per-vector attributes.
5. **Single Node:** Currently supports only a single node.

#### Contact

For questions or contributions, please open an issue or reach out to jasonmohoney@gmail.com
   
