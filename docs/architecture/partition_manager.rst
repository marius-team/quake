PartitionManager
========================================

Overview
--------
The PartitionManager class is a higher-level coordinator for managing the partitions of a dynamic IVF index.
It provides the logic for initializing, modifying, and maintaining partitions. Internally, it relies on a DynamicInvertedLists object—which handles the low‐level
storage of vector codes and IDs using a map of IndexPartition objects—to actually store the data, and the MaintenancePolicy object to determine when to perform partition reconfiguration.

Key Responsibilities
--------------------
- **Initialization:**
  Use a clustering result (a Clustering object) to initialize partitions in the underlying DynamicInvertedLists.
  This includes setting partition IDs, centroids, and assigning vectors to partitions.

- **Vector Assignment and Modification:**
  Add new vectors into appropriate partitions based on explicit assignments or by performing a search on the parent index.
  It also handles the removal of vectors by their IDs.

- **Partition Reconfiguration:**
  Manage partition splits, merges, and refinement. PartitionManager provides methods for splitting a partition into
  smaller clusters and for reassigning vectors when partitions are merged or refined.

- **Distribution:**
  Offer methods to distribute a flat index into multiple partitions (``distribute_flat``) or partitioned index (``distribute_partitions``) and assign partitions
  across cores for parallel query processing.

- **Serialization:**
  Save and load the complete state of the partition manager (including all partitions) from disk.

Differences from DynamicInvertedLists
---------------------------------------
DynamicInvertedLists is the low-level container that holds partitions (each as an IndexPartition) and provides basic
operations for adding, updating, and retrieving vector codes and IDs. In contrast, PartitionManager adds an additional
layer of high-level logic, such as:

- Converting clustering results into a partitioned index.
- Managing vector assignment across partitions.
- Performing partition reconfiguration (splitting, refinement, deletion) in the context of the overall index.
- Coordinating distribution across workers and serialization.

Thus, while DynamicInvertedLists is responsible for the raw storage and basic CRUD operations, PartitionManager
orchestrates these operations in the context of an evolving, partitioned index.

Usage
-----
PartitionManager is intended to be used by the high-level index (e.g. QuakeIndex) to manage partitions during index
construction and maintenance. It encapsulates the logic for partition initialization, vector assignment, and periodic
reconfiguration of partitions.
