//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <query_coordinator.h>
#include <sys/fcntl.h>

#include "partition_manager.h"
#include "quake_index.h"
#include "list_scanning.h"

// query_coordinator.cpp
QueryCoordinator::QueryCoordinator(shared_ptr<QuakeIndex> parent,
                                   shared_ptr<PartitionManager> partition_manager,
                                   MetricType metric)
  : parent_(parent), partition_manager_(partition_manager), metric_(metric)
{
}

QueryCoordinator::~QueryCoordinator() = default;

shared_ptr<SearchResult> QueryCoordinator::search(Tensor x, shared_ptr<SearchParams> search_params) {

    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::search] partition_manager_ is null.");
    }

    x = x.contiguous();

    auto parent_timing_info = std::make_shared<SearchTimingInfo>();
    auto start = std::chrono::high_resolution_clock::now();

    // if there is no parent, then the coordinator is operating on a flat index and we need to scan all partitions
    Tensor partition_ids_to_scan;
    if (parent_ == nullptr) {
        // scan all partitions for each query
        partition_ids_to_scan = torch::arange(partition_manager_->nlist(), torch::kInt64);
        search_params->batched_scan = true;
    } else {
        auto parent_search_params = make_shared<SearchParams>();
        parent_search_params->k = search_params->nprobe;
        parent_search_params->recall_target = search_params->recall_target;
        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        parent_timing_info = parent_search_result->timing_info;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


    auto search_result = scan_partitions(x, partition_ids_to_scan, search_params);
    search_result->timing_info->parent_info = parent_timing_info;
    search_result->timing_info->quantizer_search_time_us = duration.count();

    return search_result;
}

shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params) {
    if (workers_initialized_) {
        return worker_scan(x, partition_ids, search_params);
    } else {
        if (search_params->batched_scan) {
            return batched_serial_scan(x, partition_ids, search_params);
        } else {
            return serial_scan(x, partition_ids, search_params);
        }
    }
}

shared_ptr<SearchResult> QueryCoordinator::batched_serial_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::scan_partitions] partition_manager_ is null.");
    }

    if (!x.defined() || x.size(0) == 0) {
        // Return empty
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->ids = torch::empty({0}, torch::kInt64);
        empty_res->distances = torch::empty({0}, torch::kFloat32);
        empty_res->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_res;
    }

    auto timing_info = std::make_shared<SearchTimingInfo>();

    auto start = std::chrono::high_resolution_clock::now();

    // if parent is null then we are scanning all partitions
    if (parent_ == nullptr) {
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
        vector<shared_ptr<TopkBuffer>> buffers = create_buffers(x.size(0), search_params->k, metric_ == faiss::METRIC_INNER_PRODUCT);

        // assume the same partitions are scanned for all queries
        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t partition_id = partition_ids_accessor[i];
            batched_scan_list(x.data_ptr<float>(),
                (float *) partition_manager_->partitions_->get_codes(partition_id),
                partition_manager_->partitions_->get_ids(partition_id),
                x.size(0),
                partition_manager_->partitions_->list_size(partition_id),
                x.size(1),
                buffers,
                metric_);
        }

        auto [topk_ids, topk_distances] = buffers_to_tensor(buffers);

        auto search_result = std::make_shared<SearchResult>();
        search_result->ids = topk_ids;
        search_result->distances = topk_distances;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        timing_info->partition_scan_time_us = duration.count();
        search_result->timing_info = timing_info;

        return search_result;
    } else {
        return serial_scan(x, partition_ids, search_params);
    }
}

shared_ptr<SearchResult> QueryCoordinator::worker_scan(Tensor x, Tensor partition_ids, shared_ptr<SearchParams> search_params) {
    return nullptr;
}

shared_ptr<SearchResult> QueryCoordinator::serial_scan(
    Tensor x,
    Tensor partition_ids_to_scan,
    shared_ptr<SearchParams> search_params
) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::scan_partitions] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        // Return empty
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->ids = torch::empty({0}, torch::kInt64);
        empty_res->distances = torch::empty({0}, torch::kFloat32);
        empty_res->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_res;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Basic info
    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;

    // Prepare output [num_queries, k]
    auto ret_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto ret_dists = torch::full({num_queries, k}, std::numeric_limits<float>::infinity(), torch::kFloat32);

    // For timing
    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_vectors = partition_manager_->ntotal();
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->d = dimension;
    timing_info->k = k;

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);

    if (partition_ids_to_scan.dim() == 1) {
        // all queries need to scan the same partitions
        partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({num_queries, partition_ids_to_scan.size(0)});
    }

    float* x_ptr = x.data_ptr<float>();
    float* out_dists_ptr = ret_dists.data_ptr<float>();
    int64_t* out_ids_ptr = ret_ids.data_ptr<int64_t>();
    auto partition_ids_to_scan_accessor = partition_ids_to_scan.accessor<int64_t, 2>();

    // For each query
    for (int64_t q = 0; q < num_queries; q++) {
        auto topk_buf = make_shared<TopkBuffer>(k, is_descending);
        const float* query_vec = x_ptr + q * dimension;

        // For each partition in clustering
        for (size_t j = 0; j < partition_ids_to_scan.size(1); j++) {
            int64_t pi = partition_ids_to_scan_accessor[q][j];

            scan_list(query_vec,
                (float *) partition_manager_->partitions_->get_codes(pi),
                partition_manager_->partitions_->get_ids(pi),
                partition_manager_->partitions_->list_size(pi),
                dimension,
                topk_buf,
                metric_);
        }
        std::vector<float> best_dists = topk_buf->get_topk();
        std::vector<int64_t> best_ids = topk_buf->get_topk_indices();
        int n_results = std::min<int>((int)best_dists.size(), k);

        // print out best results
        for (int i = 0; i < n_results; i++) {
            out_dists_ptr[q * k + i] = best_dists[i];
            out_ids_ptr[q * k + i] = best_ids[i];
        }
    }

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = ret_ids;
    search_result->distances = ret_dists;
    search_result->timing_info = timing_info;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    timing_info->partition_scan_time_us = duration.count();

    return search_result;
}