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
    const float* query_vector;    ///< Pointer to the query vector (s)
    int64_t single_query_global_id = -1;
    bool is_batched = false;      ///< Indicates whether this is a batched query job.
    int64_t num_queries = 0;      ///< The number of queries in batched mode.
    int rank = 0;                 ///< Rank of the partition

    std::vector<int64_t> query_ids_for_batch_job;

    ScanJob() :
            is_batched(false),
            partition_id(-1),
            k(0),
            query_vector(nullptr),
            num_queries(0),
            single_query_global_id(-1) {}
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

struct AggregatedResultItem {
    int64_t query_global_id; // The global ID of the query this result pertains to
    PartitionScanResult result;    // The actual result data from the worker
};

//// Structure to hold resources per core/worker
//struct CoreResources {
//    int core_id = -1;
//    std::vector<uint8_t> local_query_buffer;
//    std::vector<std::shared_ptr<TopkBuffer>> topk_buffer_pool; // Local buffers for workers
//    moodycamel::BlockingConcurrentQueue<ScanJob> job_queue;
//};

// Forward declare QueryCoordinator if CoreResources is in a separate header used by it.
// class QueryCoordinator; // If needed

struct CoreResources {
    int core_id = -1;
    moodycamel::BlockingConcurrentQueue<int64_t> job_queue;
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffer_pool; // Pool of TopkBuffers for this core

    // NUMA-aware local buffer for query data
    uint8_t* local_query_data_ = nullptr;
    size_t local_query_buffer_capacity_bytes_ = 0;
    int allocated_on_numa_node_ = -1; // NUMA node this buffer is allocated on, -1 for default/unknown

    CoreResources() = default; // Keep default constructor

    // Proper cleanup in destructor
    ~CoreResources() {
        if (local_query_data_) {
            // Use a static helper from QueryCoordinator or a common utility to free
            // This avoids making QueryCoordinator a friend or complex dependencies
            // For now, conceptual: QueryCoordinator::free_buffer_for_core(this);
            // Or, more directly if the static free function is accessible:
#ifdef QUAKE_USE_NUMA
            if (allocated_on_numa_node_ != -1 && numa_available() != -1) {
                numa_free(local_query_data_, local_query_buffer_capacity_bytes_);
            } else {
                std::free(local_query_data_);
            }
#else
            std::free(local_query_data_);
#endif
            local_query_data_ = nullptr;
        }
    }

    // Disable copy constructor and copy assignment operator
    CoreResources(const CoreResources&) = delete;
    CoreResources& operator=(const CoreResources&) = delete;

    // Enable move constructor and move assignment operator (optional but good practice)
    CoreResources(CoreResources&& other) noexcept
            : core_id(other.core_id),
              job_queue(std::move(other.job_queue)), // Requires moodycamel queue to be movable or handle manually
              topk_buffer_pool(std::move(other.topk_buffer_pool)),
              local_query_data_(other.local_query_data_),
              local_query_buffer_capacity_bytes_(other.local_query_buffer_capacity_bytes_),
              allocated_on_numa_node_(other.allocated_on_numa_node_) {
        other.local_query_data_ = nullptr; // Nullify moved-from raw pointer
        other.local_query_buffer_capacity_bytes_ = 0;
        other.allocated_on_numa_node_ = -1;
    }

    CoreResources& operator=(CoreResources&& other) noexcept {
        if (this != &other) {
            // Cleanup existing resources
            if (local_query_data_) {
#ifdef QUAKE_USE_NUMA
                if (allocated_on_numa_node_ != -1 && numa_available() != -1) {
                    numa_free(local_query_data_, local_query_buffer_capacity_bytes_);
                } else { std::free(local_query_data_); }
#else
                std::free(local_query_data_);
#endif
            }

            core_id = other.core_id;
            job_queue = std::move(other.job_queue); // Requires moodycamel queue to be movable
            topk_buffer_pool = std::move(other.topk_buffer_pool);
            local_query_data_ = other.local_query_data_;
            local_query_buffer_capacity_bytes_ = other.local_query_buffer_capacity_bytes_;
            allocated_on_numa_node_ = other.allocated_on_numa_node_;

            other.local_query_data_ = nullptr;
            other.local_query_buffer_capacity_bytes_ = 0;
            other.allocated_on_numa_node_ = -1;
        }
        return *this;
    }
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

    moodycamel::ConcurrentQueue<AggregatedResultItem> aggregated_results_queue_;
    std::mutex result_queues_mutex_; // Protects resizing of query_result_queues_
    bool use_numa_ = false;
    std::vector<ScanJob> job_details_store_;


    void partition_scan_worker_fn(int core_index);

    std::shared_ptr<SearchResult> worker_scan(Tensor x, Tensor partition_ids_to_scan_all_queries, std::shared_ptr<SearchParams> search_params);
    std::shared_ptr<SearchResult> serial_scan(Tensor x, Tensor partition_ids, std::shared_ptr<SearchParams> search_params);
    std::shared_ptr<SearchResult> batched_serial_scan(Tensor x, Tensor partition_ids, std::shared_ptr<SearchParams> search_params);

    std::shared_ptr<SearchResult> scan_partitions(Tensor x, Tensor partition_ids_to_scan, std::shared_ptr<SearchParams> search_params);
    void allocate_core_resources(int core_idx, int k_default, int d_default, bool attempt_numa_for_buffer);


    bool debug_ = false;
    std::atomic<long long> job_pull_time_ns{0};
    std::atomic<long long> job_process_time_ns{0};
};

#endif // QUERY_COORDINATOR_H