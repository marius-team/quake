IndexPartition
====================================

Overview
--------
The ``IndexPartition`` class is a core component in partitioned index.
It is responsible for managing a contiguous block of memory that stores encoded vector data
(codes) alongside their corresponding vector IDs. The class provides dynamic memory management
for vector storage—including adding, updating, and removing vectors, and support for optional
NUMA-aware memory allocation.

Key Responsibilities
--------------------
- **Memory Management:**
  Allocates, reallocates, and frees memory for vector codes and IDs. The class dynamically resizes
  its internal buffer to accommodate new vectors.

- **Dynamic Updates:**
  Provides methods to append new vectors, update existing ones, and remove entries (using a swap‐with‐last
  strategy, so order is not preserved).

- **Lookup:**
  Implements a linear search via ``find_id()`` to locate vectors by their ID.

- **NUMA Support (Optional):**
  If compiled with NUMA support (``QUAKE_USE_NUMA`` defined), it can reassign its memory to a specified NUMA node.

Public Interface
----------------
Developers using this class should pay attention to the following methods:

- **set_code_size(int64_t code_size):**
  Sets the size (in bytes) of each vector code. This must be done before any vectors are added.

- **append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes):**
  Appends new vector entries to the partition. The method automatically ensures that there is enough
  capacity by resizing if necessary.

- **update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes):**
  Updates existing entries in place starting from a given offset. It will throw an exception if the
  specified range is out of bounds.

- **remove(int64_t index):**
  Removes a vector at the given index by swapping in the last vector. This operation does not preserve
  the original order of vectors.

- **resize(int64_t new_capacity):**
  Resizes the partition’s internal buffer. If the new capacity is lower than the current number of vectors,
  the partition will be truncated.

- **clear():**
  Frees all allocated memory and resets the partition’s state.

- **find_id(idx_t id) const:**
  Searches for a vector by its ID using a linear scan and returns its index or -1 if not found.

- **reallocate_memory(int64_t new_capacity):**
  Forces a reallocation of the internal memory buffer, copying over existing data.

Usage Considerations
--------------------
- **Removal Semantics:**
  The removal method uses a swap-with-last strategy. This is efficient but does not preserve the order of vectors.

- **Exceptions:**
  Methods such as ``set_code_size()`` and ``update()`` enforce that certain conditions are met
  (e.g., no vectors exist yet or the update range is valid), throwing exceptions otherwise.

- **NUMA Considerations:**
  When NUMA support is enabled, the developer may assign the partition to a specific NUMA node to improve performance on multi-socket systems.

- **Memory Efficiency:**
  The partition grows by doubling its capacity when necessary, ensuring amortized constant time for appends.

- **Thread Safety:**
  This class is not internally synchronized. In multi-threaded environments, external synchronization
  must be applied when concurrently modifying the same partition.

IndexPartition Layout
---------------------

.. mermaid::

    %%{
      init: {
        "theme": "default",
        "themeVariables": {
          "fontSize": "14px",
          "fontFamily": "Courier New",
            "fontWeight": "bold"
        }
      }
    }%%
    flowchart LR

        subgraph IP["IndexPartition"]
            direction TB
             MD["buffer_size_
                num_vectors_
                code_size_
                numa_node_
                core_id_"]
            CDPTR["codes_ (uint8_t*)"]
            IDPTR["ids_ (int64_t*)"]
        end

        subgraph ML["Memory Layout"]
            direction TB
            CA["Vector Buffer<br/>size = buffer_size_ × code_size_"]
            IA["ID Buffer<br/>size = buffer_size_ x sizeof(int64_t)"]
        end

        CDPTR --> CA
        IDPTR --> IA

Conclusion
----------
The ``IndexPartition`` class is a low-level container for managing partitions of an index.
It encapsulates memory management, dynamic updates, and (optionally) NUMA-aware operations.
By understanding its public interface and internal design, developers can effectively integrate
and extend the partitioned index system.