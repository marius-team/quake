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
    shared_ptr<TopkBuffer> local_topk_buffer;

    while (true) {
        int job_id;
        jobs_queue_[worker_id].wait_dequeue(job_id);

        // Check for shutdown signal
        if (job_id == -1) {
            break;
        }

        // Lookup the ScanJob
        ScanJob job; {
            std::lock_guard<std::mutex> lock(result_mutex_);
            auto it = jobs_.find(job_id);
            if (it == jobs_.end()) {
                std::cerr << "[partition_scan_worker_fn] Invalid job_id " << job_id << std::endl;
                continue;
            }
            job = it->second;
        }

        // check if query is done processing
        if (!query_topk_buffers_[job.query_ids[0]]->currently_processing_query()) {
            // mark job as done and continue
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                jobs_.erase(job_id);
            }
            continue;
        }

        // Retrieve partition data
        const float *partition_codes = (float *) partition_manager_->partitions_->get_codes(job.partition_id);
        const int64_t *partition_ids = partition_manager_->partitions_->get_ids(job.partition_id);
        int64_t partition_size = partition_manager_->partitions_->list_size(job.partition_id);

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
                      *local_topk_buffer,
                      metric_);

            vector<float> topk = local_topk_buffer->get_topk();
            vector<int64_t> topk_indices = local_topk_buffer->get_topk_indices();
            int64_t n_results = topk_indices.size();

            // Merge into global buffer for this query
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                if (query_topk_buffers_[job.query_ids[0]]) {
                    query_topk_buffers_[job.query_ids[0]]->batch_add(
                        topk.data(),
                        topk_indices.data(),
                        n_results
                    );
                }
                jobs_.erase(job_id);
            }
        } else {
            if (!job.query_vector || job.num_queries == 0) {
                throw std::runtime_error("[QueryCoordinator::partition_scan_worker_fn] Invalid batched job.");
            }

            // Create a local TopkBuffer for EACH query
            std::vector<std::shared_ptr<TopkBuffer> > local_buffers(job.num_queries);
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
            ); {
                std::lock_guard<std::mutex> lock(result_mutex_);
                for (int64_t q = 0; q < job.num_queries; q++) {
                    int64_t query_id = job.query_ids[q];
                    vector<float> topk = local_buffers[q]->get_topk();
                    vector<int64_t> topk_indices = local_buffers[q]->get_topk_indices();
                    int64_t n_results = topk_indices.size();
                    if (query_topk_buffers_[query_id]) {
                        query_topk_buffers_[query_id]->batch_add(
                            topk.data(),
                            topk_indices.data(),
                            n_results
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
    // Initialize a Top-K buffer for each query
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        query_topk_buffers_.resize(num_queries);
        for (int64_t q = 0; q < num_queries; q++) {
            if (partition_ids.dim() == 1) {
                query_topk_buffers_[q] = make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
                query_topk_buffers_[q]->set_jobs_left(partition_ids.size(0));
            } else {
                query_topk_buffers_[q] = make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
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
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
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

// Serial Scan Implementation
shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids_to_scan,
                                                       shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::serial_scan] partition_manager_ is null.");
    }

    if (debug_) {
        std::cout << "[QueryCoordinator::serial_scan] x: " << x.sizes() << std::endl;
        std::cout << "[QueryCoordinator::serial_scan] partition_ids_to_scan: " << partition_ids_to_scan.sizes() <<
                std::endl;
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
    timing_info->n_clusters = partition_manager_->nlist();

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    bool use_aps = (search_params->recall_target > 0.0 && parent_);

    if (partition_ids_to_scan.dim() == 1) {
        // All queries need to scan the same partitions
        partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({num_queries, partition_ids_to_scan.size(0)});
    }

    float *x_ptr = x.data_ptr<float>();
    float *out_dists_ptr = ret_dists.data_ptr<float>();
    int64_t *out_ids_ptr = ret_ids.data_ptr<int64_t>();
    auto partition_ids_accessor = partition_ids_to_scan.accessor<int64_t, 2>();

    auto end_time = std::chrono::high_resolution_clock::now();
    timing_info->buffer_init_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).
            count();

    int64_t boundary_distance_time_ns = 0;
    int64_t scan_duration_ns = 0;
    int64_t result_aggregate_time_ns = 0;
    int64_t recall_model_time_ns = 0;
    int64_t partitions_scanned = 0;
    // For each query
    for (int64_t q = 0; q < num_queries; q++) {
        auto topk_buf = make_shared<TopkBuffer>(k, is_descending);
        const float *query_vec = x_ptr + q * dimension;
        Tensor boundary_distances;
        Tensor partition_probabilities;
        float query_radius = 1000000.0;
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            query_radius = -1000000.0;
        }

        Tensor sort_args = torch::arange(partition_ids_to_scan.size(1), torch::kInt64);
        Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids_to_scan[q]);
        if (use_aps) {
            start_time = std::chrono::high_resolution_clock::now();
            Tensor cluster_centroids = parent_->get(partition_ids_to_scan[q]);
            boundary_distances = compute_boundary_distances(x[q],
                                                            cluster_centroids,
                                                            metric_ == faiss::METRIC_L2);

            // sort order by boundary distance
            sort_args = torch::argsort(boundary_distances, 0, false);

            end_time = std::chrono::high_resolution_clock::now();
            boundary_distance_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).
                    count();
        }
        auto sort_args_accessor = sort_args.accessor<int64_t, 1>();

        for (size_t j = 0; j < partition_ids_to_scan.size(1); j++) {
            int64_t pi = partition_ids_accessor[q][j];

            if (pi == -1) {
                continue; // Skip invalid partitions
            }

            start_time = std::chrono::high_resolution_clock::now();
            float *list_vectors = (float *) partition_manager_->partitions_->get_codes(pi);
            int64_t *list_ids = (int64_t *) partition_manager_->partitions_->get_ids(pi);
            int64_t list_size = partition_manager_->partitions_->list_size(pi);

            scan_list(query_vec,
                      list_vectors,
                      list_ids,
                      partition_manager_->partitions_->list_size(pi),
                      dimension,
                      *topk_buf,
                      metric_);
            partitions_scanned++;

            float curr_radius = topk_buf->get_kth_distance();
            float percent_change = abs(curr_radius - query_radius) / curr_radius;
            end_time = std::chrono::high_resolution_clock::now();
            scan_duration_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

            start_time = std::chrono::high_resolution_clock::now();
            if (use_aps) {
                if (percent_change > search_params->recompute_threshold) {
                    query_radius = curr_radius;
                    partition_probabilities = compute_recall_profile(boundary_distances,
                                                                     query_radius,
                                                                     dimension,
                                                                     partition_sizes,
                                                                     search_params->use_precomputed,
                                                                     metric_ == faiss::METRIC_L2).cumsum(0);
                }
                if (partition_probabilities[j].item<float>() >= search_params->recall_target) {
                    break;
                }
            }
            end_time = std::chrono::high_resolution_clock::now();
            recall_model_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        }

        start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> best_dists = topk_buf->get_topk();
        std::vector<int64_t> best_ids = topk_buf->get_topk_indices();
        int n_results = std::min<int>((int) best_dists.size(), k);

        // Populate the output tensors
        for (int i = 0; i < n_results; i++) {
            out_dists_ptr[q * k + i] = best_dists[i];
            out_ids_ptr[q * k + i] = best_ids[i];
        }
        end_time = std::chrono::high_resolution_clock::now();
        result_aggregate_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }
    // Prepare the search result
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = ret_ids;
    search_result->distances = ret_dists;
    search_result->timing_info = timing_info;
    search_result->timing_info->boundary_distance_time_ns = boundary_distance_time_ns;
    search_result->timing_info->job_wait_time_ns = scan_duration_ns;
    search_result->timing_info->job_enqueue_time_ns = recall_model_time_ns;
    search_result->timing_info->result_aggregate_time_ns = result_aggregate_time_ns;
    search_result->timing_info->partitions_scanned = (int) ((float) partitions_scanned / num_queries);

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

        if (maintenance_policy_ != nullptr) {
            for (int i = 0; i < partition_ids_to_scan.size(0); i++) {
                vector<int64_t> hit_partition_ids_vec = vector<int64_t>(partition_ids_to_scan[i].data_ptr<int64_t>(),
                                                                        partition_ids_to_scan[i].data_ptr<int64_t>() +
                                                                        partition_ids_to_scan[i].size(0));
                maintenance_policy_->increment_hit_count(hit_partition_ids_vec);
            }
        }
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
        return worker_scan(x, partition_ids, search_params);
    } else {
        if (search_params->batched_scan) {
            return batched_serial_scan(x, partition_ids, search_params);
        } else {
            return serial_scan(x, partition_ids, search_params);
        }
    }
}

shared_ptr<SearchResult> QueryCoordinator::batched_serial_scan(Tensor x, Tensor partition_ids,
                                                               shared_ptr<SearchParams> search_params) {
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
        vector<shared_ptr<TopkBuffer> > buffers = create_buffers(x.size(0), search_params->k,
                                                                 metric_ == faiss::METRIC_INNER_PRODUCT);

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
        search_result->timing_info = timing_info;

        return search_result;
    } else {
        return serial_scan(x, partition_ids, search_params);
    }
}
