//
// Created by Jason on 12/22/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARTITION_MANAGER_H
#define PARTITION_MANAGER_H

#include <common.h>
#include <dynamic_inverted_list.h>
#include <arrow/api.h>

class QuakeIndex;

/**
 * @brief Class that manages partitions for a dynamic IVF index.
 *
 * Responsibilities:
 *  - Initialize partition structures (e.g., create nlist partitions).
 *  - Add vectors into appropriate partitions (assign & add).
 *  - Remove or reassign vectors from partitions.
 *  - Handle merges/splits.
 */
class PartitionManager {
public:
    shared_ptr<QuakeIndex> parent_ = nullptr; ///< Pointer to a higher-level parent index.
    std::shared_ptr<faiss::DynamicInvertedLists> partition_store_ = nullptr; ///< Pointer to the inverted lists.
    int64_t curr_partition_id_ = 0; ///< Current partition ID.

    bool debug_ = false; ///< If true, print debug information.
    bool check_uniques_ = false; ///< If true, check that vector IDs are unique and don't already exist in the index.

    std::set<int64_t> resident_ids_; ///< Set of partition IDs.

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
    void init_partitions(shared_ptr<QuakeIndex> parent, shared_ptr<Clustering> partitions, bool check_uniques = true);

    /**
    * @brief Add vectors to the appropriate partition(s).
    * @param vectors Tensor of shape [num_vectors, dimension], or codes if already encoded.
    * @param vector_ids Tensor of shape [num_vectors].
    * @param assignments Tensor of shape [num_vectors] containing partition IDs. If not provided, vectors are assigned using the parent index.
    * @return Timing information for the operation.
    */
    shared_ptr<ModifyTimingInfo> add(const Tensor &vectors, const Tensor &vector_ids, const Tensor &assignments = Tensor(), bool check_uniques = true,std::shared_ptr<arrow::Table> attributes_table = {});

    /**
    * @brief Filter the appropriate row from the attribute table
    * @param table Arrow table for the attributes.
    * @param vector_id Vector_id by which we are filtering.
    * @return Table containing only the row pertaining to the vector_id
    */
    std::shared_ptr<arrow::Table> filterRowById(std::shared_ptr<arrow::Table> table, int64_t vector_id);

    /**
     * @brief Remove vectors by ID from the index.
     * @param ids Tensor of shape [num_to_remove].
     * @return Timing information for the operation.
     */
    shared_ptr<ModifyTimingInfo> remove(const Tensor &ids);

    /**
     * @brief Get vectors by ID.
     */
    Tensor get(const Tensor &ids);

    /**
     * @brief No copy version of get
     * @param ids Vector of IDs.
     */
     vector<float *> get_vectors(vector<int64_t> ids);

    /**
     * @brief Split a given partition into multiple smaller ones.
     * @param partition_ids The partition IDs to split.
     */
    shared_ptr<Clustering> split_partitions(const Tensor &partition_ids);

    /**
    * @brief Refine selected partitions using k-means
    * @param partition_ids Tensor of shape [num_partitions] containing partition IDs. If empty, refines all partitions.
    * @param refinement_iterations Number of refinement iterations. If 0, then only reassigns vectors.
    */
    void refine_partitions(Tensor partition_ids = Tensor(), int refinement_iterations = 0);

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
     * @brief Distribute the partitions across multiple workers.
     * @param num_workers The number of workers to distribute the partitions across.
     */
    void distribute_partitions(int num_workers);

    /**
     * @brief Set the core ID for a given partition.
     * @param partition_id The ID of the partition.
     */
    void set_partition_core_id(int64_t partition_id, int core_id);

    /**
     * @brief Return the core ID for a given partition.
     * @param partition_id The ID of the partition.
     */
    int get_partition_core_id(int64_t partition_id);

    /**
     * @brief Return total number of vectors across all partitions.
     */
    int64_t ntotal() const;

    /**
     * @brief Return the number of partitions currently in the manager.
     */
    int64_t nlist() const;

    /**
     * @brief Return the dimensionality of the vectors in the partitions.
     */
    int d() const;

    /**
     * @brief Get the sizes of the partitions.
     * @param partition_ids Tensor of shape [num_partitions] containing partition IDs.
     */
    Tensor get_partition_sizes(Tensor partition_ids = Tensor());

    /**
     * @brief Get the partition size.
     * @param partition_ids Vector of partition IDs.
     */
     vector<int64_t> get_partition_sizes(vector<int64_t> partition_ids);

    /**
     * @brief Get the partition size.
     * @param partition_id The ID of the partition.
     */
    int64_t get_partition_size(int64_t partition_id);

    /**
     * @brief Get the partition IDs.
     */
    Tensor get_partition_ids();

    /**
    * @brief Get ids of vectors.
    */
    Tensor get_ids();

    /**
     * @brief Validate the state of the index partitions.
     */
    bool validate();

    /**
     * @brief Save the partition manager to a file.
     * @param path Path to save the partition manager.
     */
    void save(const string &path);

    /**
     * @brief Load the partition manager from a file.
     * @param path Path to load the partition manager.
     */
    void load(const string &path);
};


#endif //PARTITION_MANAGER_H
