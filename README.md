# Quake: Query-Adaptive KNN Index

Quake is a C++ library (with Python bindings) for dynamic approximate nearest neighbor (ANN) search. At it's core is a partitioned index that supports automatic incremental maintenance while providing low-latency search. It is designed to provide both **high-throughput updates and low-latency search**.

---
Quake provides the following functionality:

- **Dynamic Indexing**  
  Build an index and update it in real‑time without full rebuilds.

- **Fast Search**  
  Use intra- and inter-query parallelism, SIMD, and NUMA for low-latency and high throughput queries.

- **Automatic Maintenance**  
  Built‑in policies dynamically split or merge partitions based on measured latency and query hit statistics.

- **Hit Your Recall Targets**  
  Automatically scans the appropriate number of parititons (nprobe) in order to meet a specfied recall target without tuning.

- **Pytorch Integration**  
  Operates over PyTorch tensors, use from C++ or through Python bindings. 

---

## Installation

### Prerequisites

- **C++ Compiler:** Must support C++17 (GCC 8+, Clang 10+, or MSVC 2019+)
- **CMake:** Version 3.16 or later
- **LibTorch:** (Download from [PyTorch.org](https://pytorch.org/))
- **Python 3:** Required for Python bindings
- **Optional:** Faiss (for GPU support), OpenMP, NUMA libraries

### Build Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/youruser/quake.git
   cd quake
   ```

2. **Configure the build**
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

   To enable GPU, NUMA, or AVX512 support, add:
   ```
   cmake -DCMAKE_BUILD_TYPE=Release -DQUAKE_ENABLE_GPU=ON -DQUAKE_USE_NUMA=ON -DQUAKE_USE_AVX512=ON ..
   ```
3. **Compile**
   ```
   make bindings -j$(nproc)
   ```
   
The build produces a Python module named _bindings that you can import.

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
``` cpp
// Configure search parameters
auto search_params = std::make_shared<SearchParams>();
search_params->k = 10;            // Top‑10 neighbors
search_params->recall_target = .9 // Automatically determine the number of partitions to scan to reach the recall target
// search_params->nprobe = 50;    // Alternatively, manually set the number of partitions to scan

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

The Python API is identical to the C++ API.

``` python
from _bindings QuakeIndex, IndexBuildParams, SearchParams
import torch

# Create and build the index
index = QuakeIndex()
build_params = IndexBuildParams()
build_params.nlist = 1000
build_params.metric = "l2"
vectors = torch.randn(100000, 128)
ids = torch.arange(100000, dtype=torch.int64)
index.build(vectors, ids, build_params)

# Search the index
search_params = SearchParams()
search_params.k = 10
search_params.nprobe = 50
queries = torch.randn(100, 128)
results = index.search(queries, search_params)

print("Neighbor IDs:", results.ids)
print("Distances:", results.distances)

# Add new vectors
new_vectors = torch.randn(1000, 128)
new_ids = torch.arange(100000, 101000, dtype=torch.int64)
index.add(new_vectors, new_ids)

# Remove vectors
remove_ids = new_ids[:500]
index.remove(remove_ids)

# Run maintenance
timing_info = index.maintenance()

# Saving and loading
index.save("path/to/index_dir")
new_index = QuakeIndex()
new_index.load("path/to/index_dir")
```

#### Current Limitations

Quake currently has the following limitations which will be addressed in future development.

1. **Metrics:** Supports the Dot Product ("ip") and Euclidean ("l2) metrics.
2. **Data Types:** For vectors we only support float32s and for the vector ids we support int64s.
3. **Vectors Only:** Currently we don't support storing or filtering on per-vector attributes.
4. **CPU-only Search:** While the index can be built using GPUs, we only support searching on the CPU.
5. **Single Node:** The implementation currently supports only a single node.

#### Contact

For questions or contributions, please open an issue or reach out to jasonmohoney@gmail.com
   
