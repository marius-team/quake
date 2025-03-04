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

struct ScanJob {
    int64_t partition_id;
    int k;
    const float* query_vector; // Pointer to the query vector
    vector<int64_t> query_ids;
    bool is_batched = false;   // false = single-query job, true = multi-query job
    int64_t num_queries = 0;   // number of queries in batched mode
};

class QueryCoordinator {
public:
    shared_ptr<PartitionManager> partition_manager_;
    shared_ptr<MaintenancePolicy> maintenance_policy_;
    shared_ptr<QuakeIndex> parent_;
    MetricType metric_;

    // Structure representing per-core resources.
    struct CoreResources {
        int core_id;  // Logical core identifier.
        // Pool of TopK buffers for queries assigned to this core.
        vector<shared_ptr<TopkBuffer>> topk_buffer_pool;
        // A per-core local aggregator for query results.
        vector<std::byte> local_query_buffer;
        // Job queue for this core.
        moodycamel::BlockingConcurrentQueue<ScanJob> job_queue;
    };

    // One CoreResources per worker core.
    vector<CoreResources> core_resources_;

    // Worker threads. (A typical design might spawn one thread per core.)
    bool workers_initialized_ = false;
    int num_workers_;
    vector<std::thread> worker_threads_;
    vector<int64_t> worker_job_counter_;

    // Global aggregator that merges local results.
    vector<shared_ptr<TopkBuffer>> global_topk_buffer_pool_;

    // Global synchronization for merging and job tracking.
    std::mutex global_mutex_;
    std::condition_variable global_cv_;
    std::atomic<int> stop_workers_;

    bool debug_ = false;

    QueryCoordinator(shared_ptr<QuakeIndex> parent,
        shared_ptr<PartitionManager> partition_manager,
        shared_ptr<MaintenancePolicy> maintenance_policy,
        MetricType metric,
        int num_workers=0);

    ~QueryCoordinator();

    shared_ptr<SearchResult> search(Tensor x, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> scan_partitions(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> batched_serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    void initialize_workers(int num_workers);

    void shutdown_workers();

    void partition_scan_worker_fn(int worker_id);

    shared_ptr<SearchResult> worker_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

private:
    void allocate_core_resources(int core_idx, int num_queries, int k, int d);
    };

#endif //QUERY_COORDINATOR_H
