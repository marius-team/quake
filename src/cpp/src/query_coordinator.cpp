// query_coordinator.cpp

#include "query_coordinator.h"
#include <sys/fcntl.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cmath>
#include <partition_manager.h>
#include <quake_index.h>
#include <geometry.h>
#include <parallel.h>

// Constructor
QueryCoordinator::QueryCoordinator(shared_ptr<QuakeIndex> parent,
                                   shared_ptr<PartitionManager> partition_manager,
                                   shared_ptr<MaintenancePolicy> maintenance_policy,
                                   MetricType metric,
                                   int num_workers)
    : parent_(parent),
      partition_manager_(partition_manager),
      maintenance_policy_(maintenance_policy),
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

    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_workers << " worker threads." <<
            std::endl;

    // Reserve space for worker threads and job queues
    worker_threads_.reserve(num_workers);
    jobs_queue_.resize(num_workers);

    std::cout << "[QueryCoordinator::initialize_workers] Creating " << num_workers << " worker threads." << std::endl;

    // Spawn worker threads
    for (int worker_id = 0; worker_id < num_workers; worker_id++) {
        worker_threads_.emplace_back(&QueryCoordinator::partition_scan_worker_fn, this, worker_id);
    }

    std::cout << "[QueryCoordinator::initialize_workers] Worker threads created." << std::endl;

    // if the index is flat we should modify the partition manager to have a single partition
    if (parent_ == nullptr) {
        partition_manager_->distribute_flat(num_workers);
    } else {
        partition_manager_->distribute_partitions(num_workers);
    }

    std::cout << "[QueryCoordinator::initialize_workers] Partitions distributed." << std::endl;

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
    for (auto &thread: worker_threads_) {
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
    // For non-batched jobs, we reuse a single buffer.
    shared_ptr<TopkBuffer> local_topk_buffer;
    // For batched jobs, we use a thread-local vector to avoid repeated allocation.
    thread_local std::vector<shared_ptr<TopkBuffer>> local_buffers;

    while (true) {
        int job_id;
        jobs_queue_[worker_id].wait_dequeue(job_id);

        // Shutdown signal: -1 indicates the worker should exit.
        if (job_id == -1) {
            break;
        }

        // Look up the job.
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

        // If the query has already been processed, skip this job.
        if (!query_topk_buffers_[job.query_ids[0]]->currently_processing_query()) {
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                jobs_.erase(job_id);
            }
            continue;
        }

        // Retrieve partition data.
        const float *partition_codes = (float *) partition_manager_->partitions_->get_codes(job.partition_id);
        const int64_t *partition_ids = (int64_t *) partition_manager_->partitions_->get_ids(job.partition_id);
        int64_t partition_size = partition_manager_->partitions_->list_size(job.partition_id);

        // Branch for non-batched jobs.
        if (!job.is_batched) {
            const float *query_vector = job.query_vector;
            if (!query_vector) {
                throw std::runtime_error("[partition_scan_worker_fn] query_vector is null.");
            }
            if (local_topk_buffer == nullptr) {
                local_topk_buffer = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
            } else {
                local_topk_buffer->set_k(job.k);
                local_topk_buffer->reset();
            }
            // Perform the scan on the partition.
            scan_list(query_vector, partition_codes, partition_ids,
                      partition_size, partition_manager_->d(), *local_topk_buffer, metric_);
            vector<float> topk = local_topk_buffer->get_topk();
            vector<int64_t> topk_indices = local_topk_buffer->get_topk_indices();
            int64_t n_results = topk_indices.size();

            // Merge local results into the global query buffer.
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                if (query_topk_buffers_[job.query_ids[0]]) {
                    query_topk_buffers_[job.query_ids[0]]->batch_add(topk.data(), topk_indices.data(), n_results);
                }
                jobs_.erase(job_id);
            }
        }
        // Batched job branch.
        else {
            if (!job.query_vector || job.num_queries == 0) {
                throw std::runtime_error("[partition_scan_worker_fn] Invalid batched job.");
            }

            // Use a thread_local vector to hold one buffer per query.
            if (local_buffers.size() < static_cast<size_t>(job.num_queries)) {
                local_buffers.resize(job.num_queries);
                for (int64_t q = 0; q < job.num_queries; ++q) {
                    local_buffers[q] = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
                }
            } else {
                for (int64_t q = 0; q < job.num_queries; ++q) {
                    local_buffers[q]->set_k(job.k);
                    local_buffers[q]->reset();
                }
            }
            // Process the batched job.
            batched_scan_list(job.query_vector, partition_codes, partition_ids,
                              job.num_queries, partition_size,
                              partition_manager_->d(), local_buffers, metric_);
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                for (int64_t q = 0; q < job.num_queries; q++) {
                    int64_t global_q = job.query_ids[q];
                    vector<float> topk = local_buffers[q]->get_topk();
                    vector<int64_t> topk_indices = local_buffers[q]->get_topk_indices();
                    int n_results = topk_indices.size();
                    if (query_topk_buffers_[global_q]) {
                        query_topk_buffers_[global_q]->batch_add(topk.data(), topk_indices.data(), n_results);
                    }
                }
                jobs_.erase(job_id);
            }
        }
    }
}

// Worker-Based Scan Implementation
shared_ptr<SearchResult> QueryCoordinator::worker_scan(
    Tensor x,
    Tensor partition_ids,
    shared_ptr<SearchParams> search_params) {
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
    int64_t dimension = x.size(1);
    int k = search_params->k;
    int64_t nlist = partition_manager_->nlist();
    bool use_aps = search_params->recall_target > 0.0 && !search_params->batched_scan;
    shared_ptr<SearchTimingInfo> timing_info = make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = nlist;
    timing_info->search_params = search_params;

    auto start_time = std::chrono::high_resolution_clock::now();
    // Initialize a Top-K buffer for each query using a global buffer pool.
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        // If our global buffer pool is smaller than needed, enlarge it.
        if (query_topk_buffers_.size() < static_cast<size_t>(num_queries)) {
            int old_size = query_topk_buffers_.size();
            query_topk_buffers_.resize(num_queries);
            for (int64_t q = old_size; q < num_queries; q++) {
                query_topk_buffers_[q] = std::make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
            }
        } else {
            // Otherwise, reset and update k for each existing buffer.
            for (int64_t q = 0; q < num_queries; q++) {
                query_topk_buffers_[q]->set_k(k);
                query_topk_buffers_[q]->reset();
            }
        }
        // Set the job count per query based on partition_ids shape.
        if (partition_ids.dim() == 1) {
            for (int64_t q = 0; q < num_queries; q++) {
                query_topk_buffers_[q]->set_jobs_left(partition_ids.size(0));
            }
        } else {
            for (int64_t q = 0; q < num_queries; q++) {
                query_topk_buffers_[q]->set_jobs_left(partition_ids.size(1));
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    timing_info->buffer_init_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).
            count();


    start_time = std::chrono::high_resolution_clock::now();
    // ============================
    // 1) BATCHED-SCAN BRANCH
    // ============================
    if (search_params->batched_scan) {
        // Force partition_ids to shape [num_queries, num_partitions]
        if (partition_ids.dim() == 1) {
            partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
        }
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();

        // Collect all unique partitions we need to scan
        std::unordered_map<int64_t, vector<int64_t> > per_partition_query_ids;
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
        for (const auto &kv: per_partition_query_ids) {
            unique_partitions.push_back(kv.first);
        }

        // Enqueue exactly one job per unique partition
        int job_counter = 0;
        for (auto partition_id: unique_partitions) {
            int job_id = job_counter++;

            int nq = per_partition_query_ids[partition_id].size();

            ScanJob scan_job;
            scan_job.is_batched = true; // <=== Key difference
            scan_job.partition_id = partition_id;
            scan_job.k = k;
            scan_job.query_vector = x.data_ptr<float>();
            scan_job.num_queries = nq;
            scan_job.query_ids = per_partition_query_ids[partition_id]; {
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
        vector<std::pair<int64_t, int64_t> > worker_ids_per_job_id;
        for (int64_t q = 0; q < num_queries; q++) {
            for (int64_t p = 0; p < partition_ids.size(1); p++) {
                int64_t partition_id = partition_ids_accessor[q][p];
                if (partition_id == -1) {
                    continue; // skip invalid
                }

                // Generate a unique job_id
                // For big nlist, watch for overflow
                int job_id = static_cast<int>(q * nlist + partition_id);

                ScanJob scan_job;
                scan_job.is_batched = false;
                scan_job.query_ids = {q};
                scan_job.partition_id = partition_id;
                scan_job.k = k;
                scan_job.query_vector = x[q].data_ptr<float>();
                scan_job.num_queries = 1; {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    jobs_[job_id] = scan_job;
                }

                int worker_id = partition_id % num_workers_;
                worker_ids_per_job_id.push_back({worker_id, job_id});
            }
        }
        for (const auto &pair: worker_ids_per_job_id) {
            jobs_queue_[pair.first].enqueue(pair.second);
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    timing_info->job_enqueue_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).
            count();

    auto last_flush_time = std::chrono::high_resolution_clock::now();

    // ============================
    // 3) WAIT FOR WORKERS
    // ============================
    start_time = std::chrono::high_resolution_clock::now();
    vector<Tensor> boundary_distances = vector<Tensor>(num_queries);
    if (use_aps) {
        for (int64_t q = 0; q < num_queries; q++) {
            Tensor cluster_centroids = parent_->get(partition_ids[q]);
            boundary_distances[q] = compute_boundary_distances(x[q],
                                                               cluster_centroids,
                                                               metric_ == faiss::METRIC_L2);
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    timing_info->boundary_distance_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time)
            .count();

    start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        {
            int time_since_last_flush_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - last_flush_time).count();
            if (use_aps && time_since_last_flush_us > search_params->aps_flush_period_us) {
                for (int64_t q = 0; q < num_queries; q++) {
                    shared_ptr<TopkBuffer> curr_buffer = query_topk_buffers_[q];
                    Tensor curr_boundary_distances = boundary_distances[q];

                    int partitions_scanned = curr_buffer->get_num_partitions_scanned();
                    if (curr_buffer->currently_processing_query() && partitions_scanned > 0 && partitions_scanned <
                        curr_boundary_distances.size(0)) {
                        float query_radius = curr_buffer->get_kth_distance();
                        Tensor partition_probabilities = compute_recall_profile(curr_boundary_distances,
                            query_radius,
                            dimension,
                            {},
                            search_params->use_precomputed,
                            metric_ == faiss::METRIC_L2);
                        Tensor recall_profile = torch::cumsum(partition_probabilities, 0);
                        last_flush_time = std::chrono::high_resolution_clock::now();
                        if (recall_profile[partitions_scanned - 1].item<float>() > search_params->recall_target) {
                            curr_buffer->set_processing_query(false);
                            break;
                        }
                    }
                }
            }

            std::lock_guard<std::mutex> lock(result_mutex_);
            if (jobs_.empty()) {
                break; // all jobs done
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    end_time = std::chrono::high_resolution_clock::now();
    timing_info->job_wait_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    // ============================
    // 4) AGGREGATE RESULTS
    // ============================
    start_time = std::chrono::high_resolution_clock::now();
    auto topk_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto topk_distances = torch::full({num_queries, k},
                                      std::numeric_limits<float>::infinity(), torch::kFloat32);
    auto topk_ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto topk_distances_accessor = topk_distances.accessor<float, 2>(); {
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (int64_t q = 0; q < num_queries; q++) {
            auto topk = query_topk_buffers_[q]->get_topk();
            auto ids = query_topk_buffers_[q]->get_topk_indices();
            auto distances = topk; // same vector

            for (int i = 0; i < k; i++) {
                if (i < (int) ids.size()) {
                    topk_ids_accessor[q][i] = ids[i];
                    topk_distances_accessor[q][i] = distances[i];
                } else {
                    // if metric is inner product, fill with -infinity beyond topk
                    topk_ids_accessor[q][i] = -1;
                    topk_distances_accessor[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT)
                                                        ? -std::numeric_limits<float>::infinity()
                                                        : std::numeric_limits<float>::infinity();
                }
            }
            timing_info->partitions_scanned = query_topk_buffers_[q]->get_num_partitions_scanned();
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    timing_info->result_aggregate_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).
            count();

    // Final SearchResult
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_distances;
    search_result->timing_info = timing_info;

    return search_result;
}

shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids_to_scan,
                                                         shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
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

    // Preallocate output tensors.
    auto ret_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto ret_dists = torch::full({num_queries, k},
                                  std::numeric_limits<float>::infinity(), torch::kFloat32);

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->search_params = search_params;

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    bool use_aps = (search_params->recall_target > 0.0 && parent_);

    // Ensure partition_ids is 2D.
    if (partition_ids_to_scan.dim() == 1) {
        partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({num_queries, partition_ids_to_scan.size(0)});
    }
    auto partition_ids_accessor = partition_ids_to_scan.accessor<int64_t, 2>();
    float* x_ptr = x.data_ptr<float>();

    // Allocate per-query result vectors.
    std::vector<std::vector<float>> all_topk_dists(num_queries);
    std::vector<std::vector<int64_t>> all_topk_ids(num_queries);

    // Use our custom parallel_for to process queries in parallel.
    parallel_for<int64_t>(0, num_queries, [&](int64_t q) {
        // Create a local TopK buffer for query q.
        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending);
        const float* query_vec = x_ptr + q * dimension;
        int num_parts = partition_ids_to_scan.size(1);
        // For each partition assigned to this query, scan and update the top-k buffer.
        for (int p = 0; p < num_parts; p++) {
            int64_t pid = partition_ids_accessor[q][p];
            if (pid == -1)
                continue;
            float* list_vectors = (float*) partition_manager_->partitions_->get_codes(pid);
            int64_t* list_ids = (int64_t*) partition_manager_->partitions_->get_ids(pid);
            int64_t list_size = partition_manager_->partitions_->list_size(pid);
            scan_list(query_vec, list_vectors, list_ids, list_size, (int)dimension, *topk_buf, metric_);
        }
        // Retrieve the top-k results for query q.
        all_topk_dists[q] = topk_buf->get_topk();
        all_topk_ids[q] = topk_buf->get_topk_indices();
    });

    // Aggregate per-query results into output tensors.
    auto ret_ids_accessor = ret_ids.accessor<int64_t, 2>();
    auto ret_dists_accessor = ret_dists.accessor<float, 2>();
    for (int64_t q = 0; q < num_queries; q++) {
        int n_results = std::min((int)all_topk_dists[q].size(), k);
        for (int i = 0; i < n_results; i++) {
            ret_dists_accessor[q][i] = all_topk_dists[q][i];
            ret_ids_accessor[q][i] = all_topk_ids[q][i];
        }
        for (int i = n_results; i < k; i++) {
            ret_ids_accessor[q][i] = -1;
            ret_dists_accessor[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT)
                                         ? -std::numeric_limits<float>::infinity()
                                         : std::numeric_limits<float>::infinity();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    timing_info->total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

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
    } else {
        auto parent_search_params = make_shared<SearchParams>();

        parent_search_params->recall_target = search_params->recall_target;
        parent_search_params->use_precomputed = search_params->use_precomputed;
        parent_search_params->recompute_threshold = search_params->recompute_threshold;
        parent_search_params->batched_scan = true;

        // if recall_target is set, we need an initial set of partitions to consider
        if (parent_search_params->recall_target > 0.0 && !search_params->batched_scan) {
            int initial_num_partitions_to_search = std::max(
                (int) (partition_manager_->nlist() * search_params->initial_search_fraction), 1);
            parent_search_params->k = initial_num_partitions_to_search;
        } else {
            parent_search_params->k = search_params->nprobe;
        }

        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        parent_timing_info = parent_search_result->timing_info;

        // if (maintenance_policy_ != nullptr) {
        //     for (int i = 0; i < partition_ids_to_scan.size(0); i++) {
        //         vector<int64_t> hit_partition_ids_vec = vector<int64_t>(partition_ids_to_scan[i].data_ptr<int64_t>(),
        //                                                                 partition_ids_to_scan[i].data_ptr<int64_t>() +
        //                                                                 partition_ids_to_scan[i].size(0));
        //         maintenance_policy_->increment_hit_count(hit_partition_ids_vec);
        //     }
        // }
    }

    auto search_result = scan_partitions(x, partition_ids_to_scan, search_params);
    search_result->timing_info->parent_info = parent_timing_info;

    auto end = std::chrono::high_resolution_clock::now();
    search_result->timing_info->total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).
            count();

    return search_result;
}

shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids,
                                                           shared_ptr<SearchParams> search_params) {
    if (workers_initialized_) {
        if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using worker-based scan." << std::endl;
        return worker_scan(x, partition_ids, search_params);
    } else {
        if (search_params->batched_scan) {
            if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using batched serial scan." << std::endl;
            return batched_serial_scan(x, partition_ids, search_params);
        } else {
            if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using serial scan." << std::endl;
            return serial_scan(x, partition_ids, search_params);
        }
    }
}

shared_ptr<SearchResult> QueryCoordinator::batched_serial_scan(
    Tensor x,
    Tensor partition_ids,
    shared_ptr<SearchParams> search_params) {

    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::batched_serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->ids = torch::empty({0}, torch::kInt64);
        empty_res->distances = torch::empty({0}, torch::kFloat32);
        empty_res->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_res;
    }

    // Timing info (could be extended as needed)
    auto timing_info = std::make_shared<SearchTimingInfo>();
    auto start = std::chrono::high_resolution_clock::now();

    int64_t num_queries = x.size(0);
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;

    // Global Top-K buffers: one for each query.
    vector<shared_ptr<TopkBuffer>> global_buffers = create_buffers(num_queries, k, (metric_ == faiss::METRIC_INNER_PRODUCT));

    // Ensure partition_ids is 2D. If it’s 1D, assume every query scans the same set.
    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
    }
    auto part_ids_accessor = partition_ids.accessor<int64_t, 2>();
    int num_parts = partition_ids.size(1);

    // Group queries by partition ID.
    std::unordered_map<int64_t, vector<int64_t>> queries_by_partition;
    for (int64_t q = 0; q < num_queries; q++) {
        for (int p = 0; p < num_parts; p++) {
            int64_t pid = part_ids_accessor[q][p];
            if (pid < 0) continue;
            queries_by_partition[pid].push_back(q);
        }
    }

    // For each unique partition, process the corresponding batch of queries.
    for (auto &entry : queries_by_partition) {
        int64_t pid = entry.first;
        vector<int64_t> query_indices = entry.second;

        // Create a tensor for the indices and then a subset of the queries.
        Tensor indices_tensor = torch::tensor(query_indices, torch::kInt64);
        Tensor x_subset = x.index_select(0, indices_tensor);
        int64_t batch_size = x_subset.size(0);

        // Get the partition’s data.
        const float *list_codes = (float *) partition_manager_->partitions_->get_codes(pid);
        const int64_t *list_ids = partition_manager_->partitions_->get_ids(pid);
        int64_t list_size = partition_manager_->partitions_->list_size(pid);
        int64_t d = partition_manager_->d();

        // Create temporary Top-K buffers for this sub-batch.
        vector<shared_ptr<TopkBuffer>> local_buffers = create_buffers(batch_size, k, (metric_ == faiss::METRIC_INNER_PRODUCT));

        // Perform a single batched scan on the partition.
        batched_scan_list(x_subset.data_ptr<float>(),
                          list_codes,
                          list_ids,
                          batch_size,
                          list_size,
                          d,
                          local_buffers,
                          metric_);

        // Merge the local results into the corresponding global buffers.
        for (int i = 0; i < batch_size; i++) {
            int global_q = query_indices[i];
            vector<float> local_dists = local_buffers[i]->get_topk();
            vector<int64_t> local_ids = local_buffers[i]->get_topk_indices();
            // Merge: global buffer adds the new candidate distances/ids.
            global_buffers[global_q]->batch_add(local_dists.data(), local_ids.data(), local_ids.size());
        }
    }

    // Aggregate the final results into output tensors.
    auto topk_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto topk_dists = torch::full({num_queries, k},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT ?
                                   -std::numeric_limits<float>::infinity() :
                                   std::numeric_limits<float>::infinity()), torch::kFloat32);
    auto topk_ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto topk_dists_accessor = topk_dists.accessor<float, 2>();

    for (int64_t q = 0; q < num_queries; q++) {
        vector<float> best_dists = global_buffers[q]->get_topk();
        vector<int64_t> best_ids = global_buffers[q]->get_topk_indices();
        int n_results = std::min((int) best_dists.size(), k);
        for (int i = 0; i < n_results; i++) {
            topk_ids_accessor[q][i] = best_ids[i];
            topk_dists_accessor[q][i] = best_dists[i];
        }
        // Fill in remaining slots with defaults.
        for (int i = n_results; i < k; i++) {
            topk_ids_accessor[q][i] = -1;
            topk_dists_accessor[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT) ?
                                        -std::numeric_limits<float>::infinity() :
                                        std::numeric_limits<float>::infinity();
        }
        // Optionally record per-query partition scan counts here.
    }

    auto end = std::chrono::high_resolution_clock::now();
    timing_info->total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Prepare and return the final search result.
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}