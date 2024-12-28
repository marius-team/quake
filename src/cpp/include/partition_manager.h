//
// Created by Jason on 12/22/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARTITION_MANAGER_H
#define PARTITION_MANAGER_H

#include <common.h>
#include <dynamic_inverted_list.h>

class QuakeIndex;

/**
 * @brief Class that manages partitions for a dynamic IVF index.
 *
 * Responsibilities:
 *  - Initialize partition structures (e.g., create nlist partitions).
 *  - Add vectors into appropriate partitions (assign & add).
 *  - Remove or reassign vectors from partitions.
 *  - Handle merges/splits if needed.
 */
class PartitionManager {
public:
    shared_ptr<QuakeIndex> parent_ = nullptr; ///< Pointer to a higher-level parent index.
    std::shared_ptr<faiss::DynamicInvertedLists> partitions_ = nullptr; ///< Pointer to the inverted lists.

    /**
     * @brief Constructor for PartitionManager.
     */
    PartitionManager();

    /**
     * @brief Destructor.
     */
    ~PartitionManager();

    /**
     * @brief Initialize partitions with a clustering
     * @param parent Pointer to the parent index over the centroids.
     * @param partitions Clustering object containing the partitions to initialize.
     */
    void init_partitions(shared_ptr<QuakeIndex> parent, shared_ptr<Clustering> partitions);

    /**
    * @brief Add vectors to the appropriate partition(s).
    * @param vectors Tensor of shape [num_vectors, dimension], or codes if already encoded.
    * @param vector_ids Tensor of shape [num_vectors].
    * @param assignments Tensor of shape [num_vectors] containing partition IDs. If not provided, vectors are assigned using the parent index.
    */
    void add(const Tensor &vectors, const Tensor &vector_ids, const Tensor &assignments = Tensor());

    /**
     * @brief Remove vectors by ID from the index.
     * @param ids Tensor of shape [num_to_remove].
     */
    void remove(const Tensor &ids);

    /**
     * @brief Get vectors by ID.
     */
    Tensor get(const Tensor &ids);

    /**
     * @brief Split a given partition into multiple smaller ones.
     * @param partition_id The ID/index of the partition to split.
     */
    shared_ptr<Clustering> split_partitions(const Tensor &partition_ids);

    /**
     * @brief Delete multiple partitions and reassign vectors
     * @param partition_ids Vector of partition IDs to merge.
     * @param reassign If true, reassign vectors to other partitions.
     */
    void delete_partitions(const Tensor &partition_ids, bool reassign = false);

    /**
     * @brief Add partitions to the level
     * @param partitions Clustering object containing the partitions to add.
     */
    void add_partitions(shared_ptr<Clustering> partitions);

    /**
     * @brief Select partitions and their centroids.
     * @param partition_ids Tensor of shape [num_partitions] containing partition IDs.
     * @param copy If true, copies the data; otherwise, uses references.
     */
    shared_ptr<Clustering> select_partitions(const Tensor &partition_ids, bool copy = false);

    /**
     * @brief Return total number of vectors across all partitions.
     */
    int64_t ntotal() const;

    /**
     * @brief Return the number of partitions currently in the manager.
     */
    int64_t nlist() const;

    int d() const;

    Tensor get_partition_sizes(Tensor partition_ids);

    Tensor get_partition_ids();

    void save(const string &path);

    void load(const string &path);
};


#endif //PARTITION_MANAGER_H
