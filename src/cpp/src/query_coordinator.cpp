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
                                   MetricType metric,
                                   int num_workers)
  : parent_(parent), partition_manager_(partition_manager), metric_(metric)
{
    if (num_workers > 0) {
        initialize_workers(num_workers);
    }
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

void QueryCoordinator::initialize_workers(int num_workers) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    num_workers_ = num_workers;
    worker_threads_.reserve(num_workers_);
    job_queues_.resize(num_workers_);

    // Spawn worker threads
    for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
        worker_threads_.emplace_back(&QueryCoordinator::partition_scan_worker_fn, this, worker_id);
    }

    workers_initialized_ = true;
}

void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    // Signal all workers to stop
    stop_workers_.store(true);

    // Notify all workers
    for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
        {
            std::lock_guard<std::mutex> lock(job_queue_mutexes_[worker_id]);
            // Enqueue a special job to unblock the worker
            job_queues_[worker_id].emplace(ScanJob{ -1, -1 });
        }
        job_queue_condition_vars_[worker_id].notify_one();
    }

    // Join all worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Clear worker-related data
    worker_threads_.clear();
    job_queues_.clear();
    job_queue_mutexes_.clear();
    job_queue_condition_vars_.clear();
    stop_workers_.store(false);
    workers_initialized_ = false;
}



void QueryCoordinator::partition_scan_worker_fn(int worker_id) {

    int d = partition_manager_->d();
    shared_ptr<TopkBuffer> local_topk_buffer = nullptr;
    while (true) {
        ScanJob job;
        {
            std::unique_lock<std::mutex> lock(job_queue_mutexes_[worker_id]);
            job_queue_condition_vars_[worker_id].wait(lock, [&]() {
                return !job_queues_[worker_id].empty();
            });

            job = job_queues_[worker_id].front();
            job_queues_[worker_id].pop();
        }

        // Check for shutdown signal
        if (job.query_id == -1 && job.partition_id == -1) {
            break;
        }

        if (local_topk_buffer == nullptr) {
            local_topk_buffer = make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
        } else {
            local_topk_buffer->set_k(job.k);
            local_topk_buffer->reset();
        }

        // Process the scan job
        int64_t query_id = job.query_id;
        int64_t partition_id = job.partition_id;
        const float* query_vector = job.query_vector;

        // Retrieve partition data
        const float* partition_codes = (float *) partition_manager_->partitions_->get_codes(partition_id);
        const int64_t* partition_ids = partition_manager_->partitions_->get_ids(partition_id);
        int64_t partition_size = partition_manager_->partitions_->list_size(partition_id);

        // Perform the scan
        scan_list(query_vector,
                  partition_codes,
                  partition_ids,
                  partition_size,
                  d,
                  local_topk_buffer,
                  metric_);

        // Add local results to global topk buffer
        {
            auto ids = local_topk_buffer->get_topk_indices();
            auto distances = local_topk_buffer->get_topk();
            query_topk_buffers_[query_id]->batch_add(distances.data(), ids.data(), distances.size());
        }

    }
}

std::shared_ptr<SearchResult> QueryCoordinator::worker_scan(torch::Tensor x, torch::Tensor partition_ids, std::shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::worker_scan] partition_manager_ is null.");
    }

    if (!x.defined() || x.size(0) == 0) {
        // Return empty result
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0}, torch::kInt64);
        empty_result->distances = torch::empty({0}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }

    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = search_params->k;

    // Initialize Top-K buffers
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        query_topk_buffers_.resize(num_queries);
        for (int64_t q = 0; q < num_queries; ++q) {
            query_topk_buffers_[q] = std::make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
        }
    }

    // Dispatch scan jobs to workers
    auto start = std::chrono::high_resolution_clock::now();

    for (int64_t q = 0; q < num_queries; ++q) {
        for (int64_t p = 0; p < partition_ids.size(1); ++p) {
            int64_t partition_id = partition_ids[q][p].item<int64_t>();
            if (partition_id == -1) {
                continue; // Skip invalid partitions
            }

            // Determine which worker should handle this partition based on NUMA node
            // int numa_node_id = partition_manager_->get_numa_node_of_partition(partition_id);
            // int worker_id = numa_node_id % num_workers_;

            // chose a random worker for now
            int worker_id = partition_id % num_workers_;

            // Enqueue the scan job
            {
                std::lock_guard<std::mutex> lock(job_queue_mutexes_[worker_id]);
                job_queues_[worker_id].emplace(ScanJob{ q, partition_id });
            }
            job_queue_condition_vars_[worker_id].notify_one();
        }
    }

    // Wait for all jobs to be processed
    // This can be optimized using barriers or condition variables
    // For simplicity, we'll use a busy-wait loop with a timeout

    bool all_jobs_done = false;
    while (!all_jobs_done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep to reduce CPU usage
        all_jobs_done = true;
    }

    // Aggregate results
    auto [topk_ids, topk_distances] = std::make_pair(torch::empty({num_queries, k}, torch::kInt64),
                                                     torch::empty({num_queries, k}, torch::kFloat32));

    for (int64_t q = 0; q < num_queries; ++q) {
        auto distances = query_topk_buffers_[q]->get_topk();
        auto ids = query_topk_buffers_[q]->get_topk_indices();

        for (int i = 0; i < k; ++i) {
            if (i < ids.size()) {
                topk_ids[q][i] = ids[i];
                topk_distances[q][i] = distances[i];
            } else {
                topk_ids[q][i] = -1;
                topk_distances[q][i] = metric_ == faiss::METRIC_INNER_PRODUCT ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Prepare the search result
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_distances;
    search_result->timing_info = std::make_shared<SearchTimingInfo>();
    search_result->timing_info->partition_scan_time_us = duration_us;

    return search_result;
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