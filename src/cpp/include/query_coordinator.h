//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUERY_COORDINATOR_H
#define QUERY_COORDINATOR_H

#include <common.h>
#include <list_scanning.h>
#include <maintenance_policies.h>
#include <blockingconcurrentqueue.h>

class QuakeIndex;
class PartitionManager;

/**
 * @brief Structure representing a scan job.
 *
 * A ScanJob encapsulates all parameters required to perform a scan on a given index partition.
 */
struct ScanJob {
 int64_t partition_id;         ///< The identifier of the partition to be scanned.
 int k;                        ///< The number of neighbors (Top-K) to return.
 const float* query_vector;    ///< Pointer to the query vector.
 vector<int64_t> query_ids;    ///< Global query IDs; used in batched mode.
 bool is_batched = false;      ///< Indicates whether this is a batched query job.
 int64_t num_queries = 0;      ///< The number of queries in batched mode.
 int rank = 0;                 ///< Rank of the partition
};

/**
 * @brief The QueryCoordinator class.
 *
 * Distributes query scanning work across worker threads, aggregates results,
 * and supports both parallel and serial scan modes.
 */
class QueryCoordinator {
public:
    // Public member variables (for internal use)
    shared_ptr<PartitionManager> partition_manager_; ///< Manager for partition assignments.
    shared_ptr<MaintenancePolicy> maintenance_policy_; ///< Policy for index maintenance.
    shared_ptr<QuakeIndex> parent_;                    ///< Pointer to the parent index.
    MetricType metric_;                                ///< Distance metric for search queries.

    /**
     * @brief Structure representing per-core resources.
     *
     * Each core maintains its own pool of Top‑K buffers, a local query buffer, and a dedicated job queue.
     */
    struct CoreResources {
     int core_id; ///< Logical identifier of the core.
     vector<shared_ptr<TopkBuffer>> topk_buffer_pool; ///< Preallocated Top‑K buffers.
     vector<std::byte> local_query_buffer;            ///< Local aggregator for query results.
     moodycamel::BlockingConcurrentQueue<ScanJob> job_queue; ///< Job queue for scan jobs.
    };

    vector<CoreResources> core_resources_;             ///< Per‑core resources for worker threads.
    bool workers_initialized_ = false;                 ///< Flag indicating if worker threads are initialized.
    int num_workers_;                                  ///< Total number of worker threads.
    vector<std::thread> worker_threads_;               ///< Container for worker threads.
    vector<int64_t> worker_job_counter_;               ///< Job counters for each worker.
    vector<shared_ptr<TopkBuffer>> global_topk_buffer_pool_; ///< Global aggregator buffers.
    std::mutex global_mutex_;                          ///< Mutex for global synchronization.
    std::condition_variable global_cv_;                ///< Condition variable for thread coordination.
    std::atomic<int> stop_workers_;                    ///< Flag to signal workers to terminate.
    bool debug_ = false;                               ///< Debug mode flag.

    vector<vector<std::atomic<bool>>> job_flags_; ///< Flags to track job completion
    std::atomic<int64_t> job_pull_time_ns = 0; ///< Time spent pulling jobs from the queue.
    std::atomic<int64_t> job_process_time_ns = 0; ///< Time spent processing jobs.


    /**
    * @brief Constructs a QueryCoordinator.
    *
    * @param parent Shared pointer to the parent QuakeIndex.
    * @param partition_manager Shared pointer to the PartitionManager.
    * @param maintenance_policy Shared pointer to the MaintenancePolicy.
    * @param metric Distance metric used in search operations.
    * @param num_workers Number of worker threads to initialize (default is 0, where 0 means no parallelism).
    */
    QueryCoordinator(shared_ptr<QuakeIndex> parent,
        shared_ptr<PartitionManager> partition_manager,
        shared_ptr<MaintenancePolicy> maintenance_policy,
        MetricType metric,
        int num_workers=0);

    /**
    * @brief Destructor for QueryCoordinator.
    *
    * Cleans up resources and shuts down worker threads.
    */
    ~QueryCoordinator();

    /**
    * @brief Initiates a search operation.
    *
    * Searches the parent first to determine the partitions to scan. Then calls scan_partitions to perform the scan.
    *
    * @param x Tensor containing the query vector(s).
    * @param search_params Shared pointer to search parameters.
    * @return Shared pointer to the final SearchResult.
    */
    shared_ptr<SearchResult> search(Tensor x, shared_ptr<SearchParams> search_params);

    /**
     * @brief Performs a scan on the specified partitions.
     *
     * Selects the appropriate scan method based on the search parameters and coordinator configuration.
     *
     * @param x Tensor containing the query vector(s).
     * @param partition_ids Tensor with the list of partition IDs to scan.
     * @param search_params Shared pointer to search parameters.
     * @return Shared pointer to the aggregated SearchResult.
     */
    shared_ptr<SearchResult> scan_partitions(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    /**
     * @brief Executes a serial scan over the provided partitions.
     *
     * Performs a non-parallel scan, processing partitions sequentially.
     *
     * @param x Tensor containing the query vector(s).
     * @param partition_ids Tensor with the list of partition IDs to scan.
     * @param search_params Shared pointer to search parameters.
     * @return Shared pointer to the SearchResult.
     */
    shared_ptr<SearchResult> serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    /**
     * @brief Executes a batched serial scan for multiple queries.
     *
     * Groups queries by the partitions they need to scan and processes them in batches.
     *
     * @param x Tensor containing the query vector(s).
     * @param partition_ids Tensor with the list of partition IDs to scan.
     * @param search_params Shared pointer to search parameters.
     * @return Shared pointer to the SearchResult.
     */
    shared_ptr<SearchResult> batched_serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    /**
     * @brief Initializes worker threads for parallel scanning.
     *
     * Spawns worker threads and allocates per-core resources for processing scan jobs.
     *
     * @param num_workers Number of worker threads to initialize.
     */
    void initialize_workers(int num_workers);

    /**
     * @brief Shuts down all worker threads.
     *
     * Signals each worker to terminate and waits for their completion.
     */
    void shutdown_workers();

    /**
     * @brief Function executed by each worker thread.
     *
     * Processes scan jobs from the worker's job queue
     *
     * @param worker_id Identifier for the worker thread.
     */
    void partition_scan_worker_fn(int worker_id);

    /**
     * @brief Worker thread function to perform partition scanning.
     *
     * Processes scan jobs and returns the aggregated search result.
     *
     * @param x Tensor containing the query vector(s).
     * @param partition_ids Tensor with the list of partition IDs to scan.
     * @param search_params Shared pointer to search parameters.
     * @return Shared pointer to the SearchResult.
     */
    shared_ptr<SearchResult> worker_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

private:
    /**
     * @brief Allocates per-core resources.
     *
     * Sets up necessary buffers and job queues for a specific core.
     *
     * @param core_idx The index of the core.
     * @param num_queries Number of queries to support.
     * @param k Number of nearest neighbors (Top-K) to retrieve.
     * @param d Dimensionality of the query vectors.
     */
    void allocate_core_resources(int core_idx, int num_queries, int k, int d);
    };

#endif //QUERY_COORDINATOR_H
