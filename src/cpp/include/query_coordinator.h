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

    vector<std::thread> worker_threads_;
    int num_workers_;
    bool workers_initialized_ = false;
    vector<moodycamel::BlockingConcurrentQueue<int>> jobs_queue_;
    std::unordered_map<int, ScanJob> jobs_;

    // Top-K Buffers
    vector<shared_ptr<TopkBuffer>> query_topk_buffers_;

    // Synchronization
    std::mutex result_mutex_;
    std::atomic<bool> stop_workers_;

    bool debug_ = true;

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

    };

#endif //QUERY_COORDINATOR_H
