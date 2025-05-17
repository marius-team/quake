// query_coordinator.h

#ifndef QUERY_COORDINATOR_H
#define QUERY_COORDINATOR_H

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable> // Not strictly used in this iteration, but good for future wait schemes
#include "common.h" // For Tensor, MetricType, SearchResult, SearchParams, TopkBuffer etc.
#include "blockingconcurrentqueue.h"
#include "list_scanning.h"

// Forward declarations
class QuakeIndex;
class PartitionManager;
class MaintenancePolicy;

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

// Structure to hold individual partition scan results from workers
struct PartitionScanResult {
    int64_t query_id_global; // Global index of the query this result belongs to
    int64_t original_partition_id; // The ID of the partition that was scanned
    std::vector<float> distances;
    std::vector<int64_t> indices;
    bool is_valid; // Indicates if the scan was successful and produced data

    PartitionScanResult(int64_t q_id = -1, int64_t p_id = -1, bool valid = false)
            : query_id_global(q_id), original_partition_id(p_id), is_valid(valid) {}

    PartitionScanResult(int64_t q_id, int64_t p_id, std::vector<float>&& dists, std::vector<int64_t>&& idxs)
            : query_id_global(q_id), original_partition_id(p_id), distances(std::move(dists)), indices(std::move(idxs)), is_valid(true) {}
};


// Structure to hold resources per core/worker
struct CoreResources {
    int core_id = -1;
    std::vector<uint8_t> local_query_buffer;
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffer_pool; // Local buffers for workers
    moodycamel::BlockingConcurrentQueue<ScanJob> job_queue;
};

class QueryCoordinator {
public:
    QueryCoordinator(std::shared_ptr<QuakeIndex> parent,
                     std::shared_ptr<PartitionManager> partition_manager,
                     std::shared_ptr<MaintenancePolicy> maintenance_policy,
                     MetricType metric,
                     int num_workers = 0,
                     bool use_numa = false);

    ~QueryCoordinator();

    std::shared_ptr<SearchResult> search(Tensor x, std::shared_ptr<SearchParams> search_params);
    void initialize_workers(int num_cores, bool use_numa = false);
    void shutdown_workers();

    std::shared_ptr<PartitionManager> partition_manager_;
    std::shared_ptr<MaintenancePolicy> maintenance_policy_;

    std::shared_ptr<QuakeIndex> parent_;
    MetricType metric_;
    int num_workers_;
    bool workers_initialized_;
    std::atomic<bool> stop_workers_{false};

    std::vector<std::thread> worker_threads_;
    std::vector<CoreResources> core_resources_;

    std::vector<std::shared_ptr<TopkBuffer>> global_topk_buffer_pool_;
    std::mutex global_pool_mutex_; // Protects resizing of global_topk_buffer_pool_

    // Non-blocking queue for workers to push results to the coordinator thread.
    // One such queue is needed if worker_scan handles multiple queries in one call,
    // or a single queue if worker_scan is only ever for one query's results.
    // For a single query (num_queries_total = 1 in worker_scan), one queue is enough.
    // If worker_scan can handle batches of queries, then a vector of these might be needed.
    // Let's assume for now worker_scan might process a batch of queries, so a vector of queues.
    std::vector<moodycamel::ConcurrentQueue<PartitionScanResult>> query_result_queues_;
    std::mutex result_queues_mutex_; // Protects resizing of query_result_queues_


    void partition_scan_worker_fn(int core_index);

    std::shared_ptr<SearchResult> worker_scan(Tensor x, Tensor partition_ids_to_scan_all_queries, std::shared_ptr<SearchParams> search_params);
    std::shared_ptr<SearchResult> serial_scan(Tensor x, Tensor partition_ids, std::shared_ptr<SearchParams> search_params);
    std::shared_ptr<SearchResult> batched_serial_scan(Tensor x, Tensor partition_ids, std::shared_ptr<SearchParams> search_params);

    std::shared_ptr<SearchResult> scan_partitions(Tensor x, Tensor partition_ids_to_scan, std::shared_ptr<SearchParams> search_params);
    void allocate_core_resources(int core_idx, int k_default, int d_default);


    bool debug_ = false;
    std::atomic<long long> job_pull_time_ns{0};
    std::atomic<long long> job_process_time_ns{0};
};

#endif // QUERY_COORDINATOR_H