// query_coordinator.cpp

#include "query_coordinator.h"
#include <sys/fcntl.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cmath>
#include <partition_manager.h>
#include <quake_index.h>

// Constructor
QueryCoordinator::QueryCoordinator(shared_ptr<QuakeIndex> parent,
                                   shared_ptr<PartitionManager> partition_manager,
                                   MetricType metric,
                                   int num_workers)
    : parent_(parent),
      partition_manager_(partition_manager),
      metric_(metric),
      num_workers_(num_workers),
      stop_workers_(false) {
    if (num_workers_ > 0) {
        initialize_workers(num_workers_);
    }
}

// Destructor
QueryCoordinator::~QueryCoordinator() {
    shutdown_workers();
}

// Initialize Worker Threads
void QueryCoordinator::initialize_workers(int num_workers) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    // Reserve space for worker threads and job queues
    worker_threads_.reserve(num_workers);
    jobs_queue_.resize(num_workers);

    // Spawn worker threads
    for (int worker_id = 0; worker_id < num_workers; ++worker_id) {
        worker_threads_.emplace_back(&QueryCoordinator::partition_scan_worker_fn, this, worker_id);
    }

    // if the index is flat we should modify the partition manager to have a single partition
    if (parent_ == nullptr) {
        partition_manager_->distribute_flat(num_workers);
    } else {
        partition_manager_->distribute_partitions(num_workers);
    }

    workers_initialized_ = true;
}

// Shutdown Worker Threads
void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    // Signal all workers to stop by enqueueing a special shutdown job ID (-1)
    for (int worker_id = 0; worker_id < num_workers_; ++worker_id) {
        jobs_queue_[worker_id].enqueue(-1); // -1 is reserved for shutdown
    }

    // Join all worker threads
    for (auto &thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Clear worker-related data
    worker_threads_.clear();
    jobs_queue_.clear();
    stop_workers_.store(false);
    workers_initialized_ = false;

    // Clear any remaining jobs to prevent memory leaks
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        jobs_.clear();
        query_topk_buffers_.clear();
    }
}

// Worker Thread Function
void QueryCoordinator::partition_scan_worker_fn(int worker_id) {

    shared_ptr<TopkBuffer> local_topk_buffer;

    while (true) {
        int job_id;
        jobs_queue_[worker_id].wait_dequeue(job_id);

        // Check for shutdown signal
        if (job_id == -1) {
            break;
        }

        // Lookup the ScanJob
        ScanJob job;
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                std::cerr << "[partition_scan_worker_fn] Invalid job_id " << job_id << std::endl;
                continue;
            }
            job = it->second;
        }

        // Retrieve partition data
        const float *partition_codes = (float *) partition_manager_->partitions_->get_codes(job.partition_id);
        const int64_t *partition_ids = partition_manager_->partitions_->get_ids(job.partition_id);
        int64_t partition_size       = partition_manager_->partitions_->list_size(job.partition_id);

        if (!job.is_batched) {
            const float *query_vector = job.query_vector;
            if (!query_vector) {
                throw std::runtime_error("[QueryCoordinator::partition_scan_worker_fn] query_vector is null.");
            }

            if (local_topk_buffer == nullptr) {
                local_topk_buffer = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
            } else {
                local_topk_buffer->set_k(job.k);
                local_topk_buffer->reset();
            }

            // Perform the usual scan
            scan_list(query_vector,
                      partition_codes,
                      partition_ids,
                      partition_size,
                      partition_manager_->d(),
                      local_topk_buffer,
                      metric_);

            // Merge into global buffer for this query
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                if (query_topk_buffers_[job.query_ids[0]]) {
                    query_topk_buffers_[job.query_ids[0]]->batch_add(
                        local_topk_buffer->get_topk().data(),
                        local_topk_buffer->get_topk_indices().data(),
                        local_topk_buffer->curr_offset_
                    );
                }
                jobs_.erase(job_id);
            }
        } else {
            if (!job.query_vector || job.num_queries == 0) {
                throw std::runtime_error("[QueryCoordinator::partition_scan_worker_fn] Invalid batched job.");
            }

            // Create a local TopkBuffer for EACH query
            std::vector<std::shared_ptr<TopkBuffer>> local_buffers(job.num_queries);
            for (int64_t q = 0; q < job.num_queries; ++q) {
                local_buffers[q] = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
            }

            // Single batched pass over partition
            batched_scan_list(
                job.query_vector,
                partition_codes,
                partition_ids,
                job.num_queries,
                partition_size,
                partition_manager_->d(),
                local_buffers,
                metric_
            );

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                for (int64_t q = 0; q < job.num_queries; q++) {
                    int64_t query_id = job.query_ids[q];
                    if (query_topk_buffers_[query_id]) {
                        query_topk_buffers_[query_id]->batch_add(
                            local_buffers[q]->get_topk().data(),
                            local_buffers[q]->get_topk_indices().data(),
                            local_buffers[q]->curr_offset_
                        );
                    }
                }
                // Mark job complete
                jobs_.erase(job_id);
            }
        }
    }
}

// Worker-Based Scan Implementation
shared_ptr<SearchResult> QueryCoordinator::worker_scan(
    Tensor x,
    Tensor partition_ids,
    shared_ptr<SearchParams> search_params)
{
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::worker_scan] partition_manager_ is null.");
    }

    // Handle trivial case: no input queries
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0}, torch::kInt64);
        empty_result->distances = torch::empty({0}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }


    // Basic parameters
    int64_t num_queries = x.size(0);
    int64_t dimension   = x.size(1);
    int k               = search_params->k;
    int64_t nlist       = partition_manager_->nlist();

    // Initialize a Top-K buffer for each query
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        query_topk_buffers_.resize(num_queries);
        for (int64_t q = 0; q < num_queries; ++q) {
            query_topk_buffers_[q] =
                std::make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
        }
    }

    // We'll measure total partition-scan time
    auto start_time = std::chrono::high_resolution_clock::now();

    // ============================
    // 1) BATCHED-SCAN BRANCH
    // ============================
    if (search_params->batched_scan) {
        // Force partition_ids to shape [num_queries, num_partitions]
        if (partition_ids.dim() == 1) {
            partition_ids = partition_ids.unsqueeze(0)
                                       .expand({num_queries, partition_ids.size(0)});
        }
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();

        // Collect all unique partitions we need to scan
        std::unordered_map<int64_t, vector<int64_t>> per_partition_query_ids;
        for (int64_t q = 0; q < num_queries; q++) {
            for (int64_t p = 0; p < partition_ids.size(1); p++) {
                int64_t pid = partition_ids_accessor[q][p];
                if (pid < 0) {
                    continue;
                }
                per_partition_query_ids[pid].push_back(q);
            }
        }

        vector<int64_t> unique_partitions;
        for (const auto &kv : per_partition_query_ids) {
            unique_partitions.push_back(kv.first);
        }

        // Enqueue exactly one job per unique partition
        int job_counter = 0;
        for (auto partition_id : unique_partitions) {
            int job_id = job_counter++;

            int nq = per_partition_query_ids[partition_id].size();

            ScanJob scan_job;
            scan_job.is_batched   = true;          // <=== Key difference
            scan_job.partition_id = partition_id;
            scan_job.k            = k;
            scan_job.query_vector = x.data_ptr<float>();
            scan_job.num_queries  = nq;
            scan_job.query_ids    = per_partition_query_ids[partition_id];

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                jobs_[job_id] = scan_job;
            }

            // Assign a worker: e.g., partition_id mod num_workers_
            int worker_id = partition_id % num_workers_;
            jobs_queue_[worker_id].enqueue(job_id);
        }
    }

    // ============================
    // 2) SINGLE-QUERY SCAN BRANCH
    // ============================
    else {
        // If shape is 1D, we expand so that each query sees the same partitions
        if (partition_ids.dim() == 1) {
            partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
        }
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();

        // Create a job for each (query, partition)
        for (int64_t q = 0; q < num_queries; q++) {
            for (int64_t p = 0; p < partition_ids.size(1); p++) {
                int64_t partition_id = partition_ids_accessor[q][p];
                if (partition_id == -1) {
                    continue; // skip invalid
                }
                if (partition_id < 0 || partition_id >= nlist) {
                    std::cerr << "[worker_scan] Invalid partition_id "
                              << partition_id << std::endl;
                    continue;
                }

                // Generate a unique job_id
                // For big nlist, watch for overflow
                int job_id = static_cast<int>(q * nlist + partition_id);
                if (job_id < 0 || job_id >= static_cast<int>(num_queries * nlist)) {
                    std::cerr << "[worker_scan] Out-of-range job_id " << job_id << std::endl;
                    continue;
                }

                ScanJob scan_job;
                scan_job.is_batched   = false;
                scan_job.query_ids    = {q};
                scan_job.partition_id = partition_id;
                scan_job.k            = k;
                scan_job.query_vector = x[q].data_ptr<float>();
                scan_job.num_queries  = 1;

                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    jobs_[job_id] = scan_job;
                }

                int worker_id = partition_id % num_workers_;
                jobs_queue_[worker_id].enqueue(job_id);
            }
        }
    }

    // ============================
    // 3) WAIT FOR WORKERS
    // ============================
    while (true) {
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            if (jobs_.empty()) {
                break; // all jobs done
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    // ============================
    // 4) AGGREGATE RESULTS
    // ============================
    auto topk_ids       = torch::full({num_queries, k}, -1, torch::kInt64);
    auto topk_distances = torch::full({num_queries, k},
                            std::numeric_limits<float>::infinity(), torch::kFloat32);

    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (int64_t q = 0; q < num_queries; q++) {
            auto topk      = query_topk_buffers_[q]->get_topk();
            auto ids       = query_topk_buffers_[q]->get_topk_indices();
            auto distances = topk; // same vector

            for (int i = 0; i < k; i++) {
                if (i < (int)ids.size()) {
                    topk_ids[q][i]       = ids[i];
                    topk_distances[q][i] = distances[i];
                } else {
                    // if metric is inner product, fill with -infinity beyond topk
                    topk_ids[q][i]       = -1;
                    topk_distances[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT)
                        ? -std::numeric_limits<float>::infinity()
                        :  std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto scan_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                end_time - start_time).count();

    // Final SearchResult
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids                    = topk_ids;
    search_result->distances             = topk_distances;
    search_result->timing_info           = std::make_shared<SearchTimingInfo>();
    search_result->timing_info->partition_scan_time_us = scan_duration_us;

    return search_result;
}

// Serial Scan Implementation
shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids_to_scan,
                                                       shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        // Return empty result
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0}, torch::kInt64);
        empty_result->distances = torch::empty({0}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;

    // Prepare output tensors [num_queries, k]
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
        // All queries need to scan the same partitions
        partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({num_queries, partition_ids_to_scan.size(0)});
    }

    float *x_ptr = x.data_ptr<float>();
    float *out_dists_ptr = ret_dists.data_ptr<float>();
    int64_t *out_ids_ptr = ret_ids.data_ptr<int64_t>();
    auto partition_ids_accessor = partition_ids_to_scan.accessor<int64_t, 2>();

    // For each query
    for (int64_t q = 0; q < num_queries; q++) {
        auto topk_buf = make_shared<TopkBuffer>(k, is_descending);
        const float *query_vec = x_ptr + q * dimension;

        // For each partition in clustering
        for (size_t j = 0; j < partition_ids_to_scan.size(1); j++) {
            int64_t pi = partition_ids_accessor[q][j];

            if (pi == -1) {
                continue; // Skip invalid partitions
            }

            float *list_vectors = (float *) partition_manager_->partitions_->get_codes(pi);
            int64_t *list_ids = (int64_t *) partition_manager_->partitions_->get_ids(pi);
            int64_t list_size = partition_manager_->partitions_->list_size(pi);

            scan_list(query_vec,
                      list_vectors,
                      list_ids,
                      partition_manager_->partitions_->list_size(pi),
                      dimension,
                      topk_buf,
                      metric_);
        }

        std::vector<float> best_dists = topk_buf->get_topk();
        std::vector<int64_t> best_ids = topk_buf->get_topk_indices();
        int n_results = std::min<int>((int) best_dists.size(), k);

        // Populate the output tensors
        for (int i = 0; i < n_results; i++) {
            out_dists_ptr[q * k + i] = best_dists[i];
            out_ids_ptr[q * k + i] = best_ids[i];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto scan_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_info->partition_scan_time_us = scan_duration_us;

    // Prepare the search result
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = ret_ids;
    search_result->distances = ret_dists;
    search_result->timing_info = timing_info;

    return search_result;
}

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