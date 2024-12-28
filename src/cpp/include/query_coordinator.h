//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUERY_COORDINATOR_H
#define QUERY_COORDINATOR_H

#include <common.h>

class QuakeIndex;
class PartitionManager;

class QueryCoordinator {
public:
    shared_ptr<PartitionManager> partition_manager_;
    shared_ptr<QuakeIndex> parent_;
    MetricType metric_;
    bool workers_initialized_ = false;

    QueryCoordinator(shared_ptr<QuakeIndex> parent, shared_ptr<PartitionManager> partition_manager, MetricType metric);

    ~QueryCoordinator();

    shared_ptr<SearchResult> search(Tensor x, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> scan_partitions(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    void partition_scan_worker_fn();

    shared_ptr<SearchResult> serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> worker_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    shared_ptr<SearchResult> batched_serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params);

    };

#endif //QUERY_COORDINATOR_H
