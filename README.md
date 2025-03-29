
![image](https://github.com/user-attachments/assets/559fe8da-84a6-4e44-a06a-cd35c5012e9a)

# Query-Adaptive Vector Search

Quake is a library for dynamic, high‑performance approximate nearest neighbor (ANN) search.

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

##  Key Limitations (for now)

Quake has the following limitations which will be addressed in future development.

1. **Metrics:** Only supports “l2” (Euclidean) and “ip” (inner product).
2. **Data Types:** Vectors must be float32; IDs must be int64.
3. **CPU-only Search:** Even if built with GPU, search is performed on the CPU.
4. **Vectors Only:** Currently we don't support storing or filtering on per-vector attributes.
5. **Single Node:** Currently supports only a single node.

---


### Quick Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/marius-team/quake.git
   cd quake
   git submodule update --init --recursive
   ```

2. **Set Up the Conda Environment:**

   Create and activate the environment using the provided YAML file:

   ```bash
   conda env create -f environments/ubuntu-latest/conda.yaml
   conda activate quake-env
   ```

3. **Install Quake:**

   ```bash
   pip install --no-use-pep517 .
   ```

---

## Using the Python API

Quake’s Python API provides a set of core operations. The following sections describe each operation step by step. For a complete working demonstration, please refer to the full example in [`examples/quickstart.py`](examples/quickstart.py).

### 1. Building the Index

- **Operation:** Create an index from your dataset.
- **Steps:**
    - Instantiate a `QuakeIndex` object.
    - Configure `IndexBuildParams` (e.g., set the number of clusters and metric type).
    - Call the `build()` method with your vectors and corresponding IDs.
- **Example:**
  ```python
  import quake
  import torch
  
  vectors = torch.randn(10000, 128)
  ids = torch.arange(10000)
  
  index = quake.QuakeIndex()
  build_params = quake.IndexBuildParams()
  build_params.nlist = 1024  # Number of clusters
  build_params.metric = "l2" # Use Euclidean distance
  index.build(vectors, ids, build_params)
  ```

### 2. Searching the Index

- **Operation:** Execute search queries.
- **Steps:**
    - Create a `SearchParams` instance (set the number of neighbors `k`, number of partitions to probe `nprobe`, or a recall target).
    - Use the `search()` method with your query tensor.
    - Retrieve neighbor IDs and distances.
- **Example:**
  ```python
  search_params = quake.SearchParams()
  search_params.k = 10
  search_params.nprobe = 10
  result = index.search(queries, search_params)
  ```

### 3. Updating the Index

- **Operation:** Modify the index by removing and adding vectors.
- **Steps:**
    - **Removal:** Call `remove()` with the tensor of IDs to remove.
    - **Addition:** Call `add()` with new vectors and their corresponding IDs.
- **Example:**
  ```python
  remove_ids = torch.arange(100)
  add_vectors = torch.randn(100, 128)
  add_ids = torch.arange(10000, 10100) # IDs must be unique
  index.remove(remove_ids)
  index.add(add_vectors, add_ids)
  ```

### 4. Performing Maintenance

- **Operation:** Trigger dynamic maintenance (e.g., partition splits or merges).
- **Steps:**
    - Simply call the `maintenance()` method.
    - Inspect the returned timing information for details on splits/merges.
- **Example:**
  ```python
  maintenance_info = index.maintenance()
  ```

---

## Further Documentation

For detailed instructions and advanced usage, please see the documentation: http://marius-project.org/quake/
- [Installation Guide](http://marius-project.org/quake/install)
- [Developer Guide](http://marius-project.org/quake/development_guide)
- [API Documentation](http://marius-project.org/quake/api)

---
### Contact

For questions or contributions, please open an issue or reach out to jasonmohoney@gmail.com
   
