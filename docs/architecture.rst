Quake Architecture
==================

Overview
--------
Quake is a NUMA‑aware, workload-adaptive vector search index designed for large-scale approximate nearest neighbor (ANN) retrieval. Its architecture is built upon five key components:

1. **QuakeIndex**
   The main entry point for users. It is responsible for building the index, handling query execution, and managing the overall vector data.

2. **PartitionManager**
   Divides the dataset into partitions and assigns these partitions to processing cores in a NUMA‑aware manner, ensuring data is allocated on the appropriate memory node.

3. **QueryCoordinator**
   Distributes queries across worker threads by managing per‑core job queues and local Top‑K buffers, and then aggregates the results efficiently.

4. **MaintenancePolicy**
   Implements dynamic maintenance strategies—including partition splitting, merging, and re-clustering—to adapt the index to changing workloads.

5. **Adaptive Partition Scanning (APS)**
   Monitors recall progress during query execution and enables early termination of scanning once the recall target is met, reducing unnecessary computation.

Detailed Documentation
----------------------
The following pages provide more details on each of these components:

.. toctree::
   :maxdepth: 1

   coordinator