#ifndef FILE_INDEX_PARTITION_H
#define FILE_INDEX_PARTITION_H

#include <index_partition.h>
#include <string>


class FileIndexPartition : public IndexPartition {
public:
    /// Default constructor.
    FileIndexPartition() = default;

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
    FileIndexPartition(int64_t num_vectors,
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
    FileIndexPartition(FileIndexPartition&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * Transfers the contents from another partition into this one, clearing existing data.
     *
     * @param other The partition to move from.
     * @return Reference to this partition.
     */
    FileIndexPartition& operator=(FileIndexPartition&& other) noexcept;

    /// Destructor. Frees all allocated memory.
    ~FileIndexPartition();

    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes);
    void update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes);
    void remove(int64_t index);
    void resize(int64_t new_capacity);
    void clear();
    int64_t find_id(idx_t id) const;
    void reallocate(int64_t new_capacity);
    
    
    
    // disk specific method
    void load(); // load the partition to memory from disk
    void save(); // store the vectors on disk
    


    
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
    std::string file_path;
    bool is_in_memory = false; // indicate whether the partition is in memory
    bool is_dirty = false; // indicate whether the partition is dirty (changes haven't been synced to disk)
    
};

#endif // FILE_INDEX_PARTITION_H