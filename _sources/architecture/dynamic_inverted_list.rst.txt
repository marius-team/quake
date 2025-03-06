DynamicInvertedLists Class Design
===================================

Overview
--------

The ``DynamicInvertedLists`` class provides a dynamic, NUMA-aware inverted list implementation
by extending the Faiss ``InvertedLists`` interface. Internally, it manages a collection of partitions,
each represented by an instance of the ``IndexPartition`` class, which holds a contiguous block
of vector codes and corresponding IDs.

Key Responsibilities
--------------------

- **Data Management:**
  Each partition stores a set of encoded vectors and their IDs. Partitions are stored in an
  unordered map, enabling dynamic creation and deletion.

- **Dynamic Operations:**
  Supports adding, updating, and removing entries both within a single partition and across
  partitions (batch updates).

- **NUMA Awareness:**
  Offers methods to set and get NUMA node details for each partition. This facilitates better data
  locality in NUMA systems, allowing memory allocations to be placed on specific nodes.

- **Faiss Compatibility:**
  Implements all the required virtual methods from the Faiss ``InvertedLists`` interface so that
  it can be seamlessly used in Faiss-based search and clustering workflows.

- **Serialization:**
  Provides functionality to serialize (save) and deserialize (load) the entire inverted list structure,
  including partition data and metadata.

Limitations
-----------

- The ``resize`` method is currently a no-op because the partitions are stored in a map.
- The lookup within each partition (using a linear search via ``find_id``) is simple but may become
  inefficient for very large partitions.