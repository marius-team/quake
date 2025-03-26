#ifndef DISK_INDEX_PARTITION_H
#define DISK_INDEX_PARTITION_H

#include <index_partition.h>


class DiskIndexPartition : public IndexPartition {
public:
    /// Default constructor.
    DiskIndexPartition() = default;

    /**
     * @brief Parameterized constructor.
     *
     * Initializes the partition with a given number of vectors and copies in the provided codes and IDs.
     *
     * @param num_vectors The initial number of vectors.
     * @param codes Pointer to the buffer holding the encoded vectors.
     * @param ids Pointer to the vector IDs.
     * @param code_size Size of each code in bytes.
     */
    DiskIndexPartition(int64_t num_vectors,
                    uint8_t* codes,
                    idx_t* ids,
                    int64_t code_size);

    /**
     * @brief Move constructor.
     *
     * Transfers the contents from another partition into this one.
     *
     * @param other The partition to move from.
     */
    DiskIndexPartition(DiskIndexPartition&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * Transfers the contents from another partition into this one, clearing existing data.
     *
     * @param other The partition to move from.
     * @return Reference to this partition.
     */
    DiskIndexPartition& operator=(DiskIndexPartition&& other) noexcept;

    /// Destructor. Frees all allocated memory.
    ~DiskIndexPartition();

    // overriden methods
    void set_code_size(int64_t code_size) override;
    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) override;
    void update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) override;
    void remove(int64_t index) override;
    void resize(int64_t new_capacity) override;
    void clear() override;
    int64_t find_id(idx_t id) const override;
    void reallocate_memory(int64_t new_capacity) override;
    
    
    
    // disk specific method
    void rellocate_disk(); //
    void read_from_disk(); // load the partition to memory from disk
    void write_to_disk(); // store the vectors on disk
    


    
#ifdef QUAKE_USE_NUMA
    /**
     * @brief Set the NUMA node for the partition.
     *
     * Moves the memory to the specified NUMA node if necessary.
     *
     * @param new_numa_node The target NUMA node.
     */
    void set_numa_node(int new_numa_node);
#endif

private:
    string file_handler // TODO: should this be a pointer to file handler class
    bool is_in_memory // indicate whether the partitionis in memory

    
}

#endif // DISK_INDEX_PARTITION_H