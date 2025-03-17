//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUAKE_INDEX_H
#define QUAKE_INDEX_H

#include <maintenance_policies.h>
#include <dynamic_inverted_list.h>
#include <partition_manager.h>
#include <query_coordinator.h>

/**
 * @brief Class that manages a Quake partitioned index. Provides methods for building, modifying, searching, and maintaining the index..
 */
class QuakeIndex {
public:
    shared_ptr<QuakeIndex> parent_; ///< Pointer to a higher-level parent index over the centroids
    shared_ptr<PartitionManager> partition_manager_; ///< Pointer to the partition manager.
    shared_ptr<QueryCoordinator> query_coordinator_; ///< Pointer to the query coordinator.
    shared_ptr<MaintenancePolicy> maintenance_policy_; ///< Pointer to the maintenance policy.

    MetricType metric_; ///< Metric type for the index.
    shared_ptr<IndexBuildParams> build_params_; ///< Parameters for building the index.
    shared_ptr<MaintenancePolicyParams> maintenance_policy_params_; ///< Parameters for the maintenance policy.
    int current_level_ = 0; ///< Current level of the index.

    bool debug_ = false; ///< If true, print debug information.

    /**
     * @brief Constructor for QuakeIndex.
     * @param current_level The current level of the index.
     */
    QuakeIndex(int current_level = 0);

    /**
     * @brief Destructor.
     */
    ~QuakeIndex();

    /**
     * @brief Build the index.
     * @param x Tensor of shape [num_vectors, dimension].
     * @param ids Tensor of shape [num_vectors].
     * @param build_params Parameters for building the index.
     * @return Timing information for the build.
     */
    shared_ptr<BuildTimingInfo> build(Tensor x, Tensor ids, shared_ptr<IndexBuildParams> build_params, std::vector<std::shared_ptr<arrow::Table>> data_frame = {});

    /**
     * @brief Search for vectors in the index.
     * @param x Tensor of shape [num_queries, dimension].
     * @param search_params Parameters for the search operation.
     * @return Search results.
     */
    shared_ptr<SearchResult> search(Tensor x, shared_ptr<SearchParams> search_params);

    /**
     * @brief Get vectors by ID.
     * @param ids Tensor of shape [num_ids].
     * @return Tensor of shape [num_ids, dimension].
     */
    Tensor get(Tensor ids);

    /**
     * @brief Get IDs in the index.
     */
    Tensor get_ids();

    /**
     * @brief Add vectors to the index.
     * @param x Tensor of shape [num_vectors, dimension].
     * @param ids Tensor of shape [num_vectors].
     * @return Timing information for the add operation.
     */
    shared_ptr<ModifyTimingInfo> add(Tensor x, Tensor ids);

    /**
     * @brief Remove vectors from the index.
     * @param ids Tensor of shape [num_ids].
     * @return Timing information for the remove operation.
     */
    shared_ptr<ModifyTimingInfo> remove(Tensor ids);

    /**
     * @brief In place modification of the index.
     * @param ids Tensor of shape [num_ids].
     * @param x Tensor of shape [num_ids, dimension].
     */
    shared_ptr<ModifyTimingInfo> modify(Tensor ids, Tensor x);

    /**
     * @brief Initialize the maintenance policy.
     * @param maintenance_policy_params Parameters for the maintenance policy.
     */
    void initialize_maintenance_policy(shared_ptr<MaintenancePolicyParams> maintenance_policy_params);

    /**
     * @brief Perform maintenance operations.
     * @return Timing information for the maintenance.
     */
    shared_ptr<MaintenanceTimingInfo> maintenance();

    /**
     * @brief Validate the state of the index.
     * @return True if the index is valid, false otherwise.
     */
    bool validate();

    /**
     * @brief Save the index to a file.
     * @param path Path to save the index.
     */
    void save(const std::string &path);

    /**
     * @brief Load the index from a file.
     * @param path Path to load the index.
     * @param n_workers Number of workers to use for query processing.
     */
    void load(const std::string &path, int n_workers = 0);

    /**
     * @brief Get the total number of vectors in the index.
     * @return The total number of vectors.
     */
    int64_t ntotal();

    /**
     * @brief Get the number of partitions in the index.
     * @return The number of partitions.
     */
    int64_t nlist();

    /**
     * @brief Get the dimensionality of the vectors in the index.
     * @return The dimensionality of the vectors.
     */
    int d();
};

#endif //QUAKE_INDEX_H
