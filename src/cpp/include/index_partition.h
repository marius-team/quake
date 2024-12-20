//
// Created by Jason on 12/18/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef INDEX_PARTITION_H
#define INDEX_PARTITION_H

#include <common.h>

class IndexPartition {
public:
    int numa_node_ = -1;
    int thread_id_ = -1;

    int64_t buffer_size_ = 0;   // Allocated capacity (number of vectors)
    int64_t num_vectors_ = 0;   // Current number of vectors
    int64_t code_size_ = 0;     // Size of each code in bytes (must be set before operations)

    uint8_t* codes_ = nullptr;  // Encoded vectors
    idx_t* ids_ = nullptr;      // Vector IDs

    IndexPartition() = default;

    IndexPartition(int64_t num_vectors,
                   uint8_t* codes,
                   idx_t* ids,
                   int64_t code_size);

    IndexPartition(IndexPartition&& other) noexcept;

    IndexPartition& operator=(IndexPartition&& other) noexcept;

    ~IndexPartition();

    // Set the code size (if not known at construction time)
    void set_code_size(int64_t code_size);

    // Append new entries to the end of this partition
    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes);

    // Update existing entries in place
    void update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes);

    // Remove an entry at a given index (swap last element into this position)
    void remove(int64_t index);

    // Reserve capacity for at least new_capacity entries
    void resize(int64_t new_capacity);

    // Clear all memory and reset
    void clear();

    // Optional: find the index of a given ID (linear search)
    int64_t find_id(idx_t id) const;

#ifdef QUAKE_USE_NUMA
    // Set the NUMA node and move data there if necessary
    void set_numa_node(int new_numa_node);
#endif

private:
    void move_from(IndexPartition&& other);
    void free_memory();
    void reallocate_memory(int64_t new_capacity);
    void ensure_capacity(int64_t required);

    template <typename T>
    T* allocate_memory(size_t num_elements, int numa_node);
};

#endif //INDEX_PARTITION_H
