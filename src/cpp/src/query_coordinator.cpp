// query_coordinator.cpp

#include "query_coordinator.h"
#include <sys/fcntl.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm> // For std::remove, std::min, std::find, std::distance
#include "partition_manager.h"
#include "quake_index.h"
#include "geometry.h"
#include "parallel.h"

using high_resolution_clock = std::chrono::high_resolution_clock;
using microseconds = std::chrono::microseconds;
using nanoseconds = std::chrono::nanoseconds;


// Constructor
QueryCoordinator::QueryCoordinator(std::shared_ptr<QuakeIndex> parent,
                                   std::shared_ptr<PartitionManager> partition_manager,
                                   std::shared_ptr<MaintenancePolicy> maintenance_policy,
                                   MetricType metric,
                                   int num_workers,
                                   bool use_numa)
        : parent_(parent),
          partition_manager_(partition_manager),
          maintenance_policy_(maintenance_policy),
          metric_(metric),
          num_workers_(num_workers),
          workers_initialized_(false) {

    if (num_workers_ > 0) {
        initialize_workers(num_workers_, use_numa);
    }
}

// Destructor
QueryCoordinator::~QueryCoordinator() {
    shutdown_workers();
}

const int DEFAULT_K_FOR_RESOURCES = 10;
const int DEFAULT_DIM_FOR_RESOURCES = 128;


void QueryCoordinator::allocate_core_resources(int core_idx, int k_default, int d_default) {
    CoreResources &res = core_resources_[core_idx];
    res.core_id = core_idx;

    if (res.local_query_buffer.empty()) {
        res.local_query_buffer.resize(static_cast<size_t>(d_default * sizeof(float)));
    }

    if (res.topk_buffer_pool.empty()) {
        res.topk_buffer_pool.push_back(std::make_shared<TopkBuffer>(k_default, metric_ == faiss::METRIC_INNER_PRODUCT));
    } else {
        res.topk_buffer_pool[0]->set_k(k_default);
    }
}

void QueryCoordinator::initialize_workers(int num_cores, bool use_numa) {
    if (workers_initialized_) {
        // std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }
    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_cores << " worker threads with use_numa=" << use_numa << std::endl;

    if (partition_manager_) {
        partition_manager_->distribute_partitions(num_cores, use_numa);
    } else {
        std::cerr << "[QueryCoordinator::initialize_workers] Warning: partition_manager_ is null during worker initialization." << std::endl;
    }

    core_resources_.resize(num_cores);
    worker_threads_.resize(num_cores);

    int dim_for_alloc = partition_manager_ ? partition_manager_->d() : DEFAULT_DIM_FOR_RESOURCES;

    for (int i = 0; i < num_cores; i++) {
        if (!set_thread_affinity(i)) {
            // std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
        }
        allocate_core_resources(i, DEFAULT_K_FOR_RESOURCES, dim_for_alloc);
        worker_threads_[i] = std::thread(&QueryCoordinator::partition_scan_worker_fn, this, i);
    }
    workers_initialized_ = true;
    stop_workers_.store(false);
}

void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_ || stop_workers_.load(std::memory_order_relaxed)) {
        return;
    }
    stop_workers_.store(true, std::memory_order_release);

    for (size_t i = 0; i < core_resources_.size(); ++i) {
        if(core_resources_[i].job_queue.size_approx() < 2 * num_workers_) { // Avoid over-enqueueing if already many jobs
            ScanJob termination_job;
            termination_job.partition_id = -1;
            core_resources_[i].job_queue.enqueue(termination_job);
        }
    }

    for (auto &thr : worker_threads_) {
        if (thr.joinable()) {
            thr.join();
        }
    }
    worker_threads_.clear();
    workers_initialized_ = false;
}

// Reverted partition_scan_worker_fn: workers directly update global_topk_buffer_pool and set job_flags_
void QueryCoordinator::partition_scan_worker_fn(int core_index) {
    CoreResources &res = core_resources_[core_index];

    if (!set_thread_affinity(core_index)) {
        // std::cout << "[QueryCoordinator::partition_scan_worker_fn] Failed to set thread affinity on core " << core_index << std::endl;
    }

    while (true) {
        ScanJob job;
        res.job_queue.wait_dequeue(job);

        if (job.partition_id == -1 || stop_workers_.load(std::memory_order_relaxed)) {
            break;
        }

        if (job.query_ids.empty()) {
            std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Job for partition " << job.partition_id << " has no query_ids." << std::endl;
            continue;
        }
        int64_t current_global_query_id = job.query_ids[0]; // For non-batched, or the first query in a worker's batch context

        const float *partition_codes_ptr = nullptr;
        const int64_t *partition_ids_ptr = nullptr;
        int64_t partition_size = 0;

        try {
            partition_codes_ptr = (float *)partition_manager_->partition_store_->get_codes(job.partition_id);
            partition_ids_ptr = (int64_t *)partition_manager_->partition_store_->get_ids(job.partition_id);
            partition_size = partition_manager_->partition_store_->list_size(job.partition_id);
        } catch (const std::runtime_error &e) {
            std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Error accessing partition " << job.partition_id << ": " << e.what() << std::endl;
            // Send an invalid/empty result for each query this job was supposed to serve
            for(int64_t q_id : job.query_ids) {
                query_result_queues_[q_id].enqueue(PartitionScanResult(q_id, job.partition_id, false));
            }
            continue;
        }

        if (partition_size == 0 || partition_codes_ptr == nullptr) {
            // Enqueue an empty but valid result to signal completion of this job
            for(int64_t q_id : job.query_ids) {
                query_result_queues_[q_id].enqueue(PartitionScanResult(q_id, job.partition_id, {}, {}));
            }
            continue;
        }

        int current_dim = partition_manager_->d();

        if (!job.is_batched) {
            // Ensure local_query_buffer is large enough
            size_t required_query_buffer_size = static_cast<size_t>(current_dim * sizeof(float));
            if (res.local_query_buffer.size() < required_query_buffer_size) {
                res.local_query_buffer.resize(required_query_buffer_size);
            }
            std::memcpy(res.local_query_buffer.data(), job.query_vector, required_query_buffer_size);

            if (res.topk_buffer_pool.empty()) { // Should be pre-allocated
                res.topk_buffer_pool.push_back(std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT));
            }
            std::shared_ptr<TopkBuffer> local_topk_buffer = res.topk_buffer_pool[0];
            local_topk_buffer->set_k(job.k);
            local_topk_buffer->reset();

            scan_list((float *)res.local_query_buffer.data(),
                      partition_codes_ptr, partition_ids_ptr,
                      partition_size, current_dim,
                      *local_topk_buffer, metric_);

            query_result_queues_[current_global_query_id].enqueue(
                    PartitionScanResult(current_global_query_id, job.partition_id, local_topk_buffer->get_topk(), local_topk_buffer->get_topk_indices())
            );

        } else { // Batched job for this worker (multiple queries, one partition)
            size_t required_query_batch_buffer_size = static_cast<size_t>(job.num_queries * current_dim * sizeof(float));
            if (res.local_query_buffer.size() < required_query_batch_buffer_size) {
                res.local_query_buffer.resize(required_query_batch_buffer_size);
            }

            // job.query_vector points to the start of the *entire* batch of queries for worker_scan
            // job.query_ids contains the *global indices* of queries to process in *this specific ScanJob*
            for (int i = 0; i < job.num_queries; ++i) {
                int64_t global_q_idx = job.query_ids[i];
                std::memcpy(res.local_query_buffer.data() + i * current_dim * sizeof(float), // Dest in worker's local batch buffer
                            job.query_vector + global_q_idx * current_dim,                   // Source from the full query tensor x_ptr
                            current_dim * sizeof(float));
            }

            if (res.topk_buffer_pool.size() < static_cast<size_t>(job.num_queries)) {
                res.topk_buffer_pool.resize(job.num_queries);
            }
            for (int i = 0; i < job.num_queries; ++i) {
                if (!res.topk_buffer_pool[i]) {
                    res.topk_buffer_pool[i] = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
                } else {
                    res.topk_buffer_pool[i]->set_k(job.k);
                    res.topk_buffer_pool[i]->reset();
                }
            }

            batched_scan_list((float*)res.local_query_buffer.data(), // This is the worker's prepared batch of queries
                              partition_codes_ptr, partition_ids_ptr,
                              job.num_queries, // Number of queries in *this ScanJob's batch*
                              partition_size, current_dim,
                              res.topk_buffer_pool, // Pass the pool of local buffers for this worker
                              metric_);

            for (int i = 0; i < job.num_queries; ++i) {
                int64_t global_q_idx = job.query_ids[i];
                query_result_queues_[global_q_idx].enqueue(
                        PartitionScanResult(global_q_idx, job.partition_id, res.topk_buffer_pool[i]->get_topk(), res.topk_buffer_pool[i]->get_topk_indices())
                );
            }
        }
        // job_process_time_ns += duration_cast<nanoseconds>(high_resolution_clock::now() - job_process_start).count();
    }
}


// Worker-Based Scan Implementation
std::shared_ptr<SearchResult> QueryCoordinator::worker_scan(
        Tensor x, // Query vectors [num_queries_total, dim]
        Tensor partition_ids_to_scan_all_queries, // Partition IDs for each query [num_queries_total, nprobe_max]
        std::shared_ptr<SearchParams> search_params) {

    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::worker_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0, search_params->k}, torch::kInt64); // Ensure correct shape for empty
        empty_result->distances = torch::empty({0, search_params->k}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }

    int64_t num_queries_total = x.size(0);
    int64_t dimension = x.size(1);
    int k = search_params->k;
    bool use_aps = (search_params->recall_target > 0.0 && !search_params->batched_scan && parent_ != nullptr);

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries_total;
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->search_params = search_params;

    float *x_ptr = x.data_ptr<float>();
    auto start_overall_scan_time = high_resolution_clock::now();

    if (partition_ids_to_scan_all_queries.dim() == 1) {
        partition_ids_to_scan_all_queries = partition_ids_to_scan_all_queries.unsqueeze(0).expand({num_queries_total, partition_ids_to_scan_all_queries.size(0)});
    }
    auto p_ids_acc = partition_ids_to_scan_all_queries.accessor<int64_t, 2>();

    // Initialize global buffers and result queues
    {
        std::lock_guard<std::mutex> lock_pools(global_pool_mutex_); // Protects global_topk_buffer_pool_
        std::lock_guard<std::mutex> lock_queues(result_queues_mutex_); // Protects query_result_queues_

        if (global_topk_buffer_pool_.size() < static_cast<size_t>(num_queries_total)) {
            global_topk_buffer_pool_.resize(num_queries_total);
        }
        if (query_result_queues_.size() < static_cast<size_t>(num_queries_total)) {
            query_result_queues_.resize(num_queries_total);
        }

        for (int64_t q = 0; q < num_queries_total; ++q) {
            if (!global_topk_buffer_pool_[q]) {
                global_topk_buffer_pool_[q] = std::make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
            } else {
                global_topk_buffer_pool_[q]->set_k(k);
                global_topk_buffer_pool_[q]->reset();
            }
            global_topk_buffer_pool_[q]->set_processing_query(true);

            PartitionScanResult ignored_item; // For draining the queue
            while(query_result_queues_[q].try_dequeue(ignored_item)); // Drain existing items
        }
    }
    timing_info->buffer_init_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start_overall_scan_time).count();

    auto job_enqueue_start_time = high_resolution_clock::now();
    std::vector<std::atomic<int>> expected_results_per_query(num_queries_total); // How many partition results each query expects
    std::vector<std::atomic<int>> received_results_per_query(num_queries_total); // How many results have been dequeued

    for(int64_t i=0; i<num_queries_total; ++i) { // Initialize atomic counters
        expected_results_per_query[i] = 0;
        received_results_per_query[i] = 0;
    }

    // Dispatch jobs
    if (search_params->batched_scan) {
        std::unordered_map<int64_t, std::vector<int64_t>> partition_to_queries_map;
        for (int64_t q_global = 0; q_global < num_queries_total; ++q_global) {
            int valid_jobs_for_this_q = 0;
            for (int64_t p_idx = 0; p_idx < partition_ids_to_scan_all_queries.size(1); ++p_idx) {
                int64_t pid = p_ids_acc[q_global][p_idx];
                if (pid == -1) continue;
                partition_to_queries_map[pid].push_back(q_global);
                valid_jobs_for_this_q++;
            }
            expected_results_per_query[q_global].store(valid_jobs_for_this_q);
            global_topk_buffer_pool_[q_global]->set_jobs_left(valid_jobs_for_this_q); // Initialize jobs_left for TopkBuffer
        }

        for(auto const& [pid, query_indices_for_pid] : partition_to_queries_map) {
            ScanJob job;
            job.is_batched = true;
            job.partition_id = pid;
            job.k = k;
            job.query_vector = x_ptr;
            job.num_queries = query_indices_for_pid.size();
            job.query_ids = query_indices_for_pid;

            int core_id = partition_manager_->get_partition_core_id(pid);
            if (core_id < 0 || core_id >= num_workers_) core_id = pid % num_workers_; // Simple fallback
            core_resources_[core_id].job_queue.enqueue(job);
        }
    } else { // Non-batched job dispatch
        for (int64_t q_global = 0; q_global < num_queries_total; ++q_global) {
            int valid_jobs_for_this_q = 0;
            for (int64_t p_idx = 0; p_idx < partition_ids_to_scan_all_queries.size(1); ++p_idx) {
                int64_t pid = p_ids_acc[q_global][p_idx];
                if (pid == -1) continue;
                valid_jobs_for_this_q++;
                ScanJob job;
                job.is_batched = false;
                job.query_ids = {q_global};
                job.partition_id = pid;
                job.k = k;
                job.query_vector = x_ptr + q_global * dimension;
                job.num_queries = 1;

                int core_id = partition_manager_->get_partition_core_id(pid);
                if (core_id < 0 || core_id >= num_workers_) core_id = pid % num_workers_; // Simple fallback
                core_resources_[core_id].job_queue.enqueue(job);
            }
            expected_results_per_query[q_global].store(valid_jobs_for_this_q);
            global_topk_buffer_pool_[q_global]->set_jobs_left(valid_jobs_for_this_q); // Initialize jobs_left for TopkBuffer
        }
    }
    timing_info->job_enqueue_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - job_enqueue_start_time).count();

    auto last_aps_flush_time = high_resolution_clock::now();
    std::vector<std::vector<float>> boundary_distances_all_queries(num_queries_total);
    if (use_aps) {
        for (int64_t q = 0; q < num_queries_total; ++q) {
            std::vector<int64_t> pids_for_this_query_vec;
            for (int64_t p_idx = 0; p_idx < partition_ids_to_scan_all_queries.size(1); ++p_idx) {
                if (p_ids_acc[q][p_idx] != -1) pids_for_this_query_vec.push_back(p_ids_acc[q][p_idx]);
            }
            if (!pids_for_this_query_vec.empty() && parent_ && parent_->partition_manager_) {
                std::vector<float*> cluster_centroids = parent_->partition_manager_->get_vectors(pids_for_this_query_vec);
                // Filter out nullptrs from cluster_centroids before passing to compute_boundary_distances
                cluster_centroids.erase(std::remove(cluster_centroids.begin(), cluster_centroids.end(), nullptr), cluster_centroids.end());
                if (!cluster_centroids.empty()) { // Only compute if there are valid centroids
                    boundary_distances_all_queries[q] = compute_boundary_distances(x[q], cluster_centroids, metric_ == faiss::METRIC_L2);
                } else if (!pids_for_this_query_vec.empty()) { // If there were PIDs but no valid centroids found
                    std::cerr << "[QueryCoordinator::worker_scan] APS: No valid centroids for query " << q << " for PIDs." << std::endl;
                }
            }
        }
    }

    auto job_processing_loop_start_time = high_resolution_clock::now();
    int total_expected_results_overall = 0;
    for(int64_t q=0; q<num_queries_total; ++q) total_expected_results_overall += expected_results_per_query[q].load();
    int total_processed_results_overall = 0;

    while(total_processed_results_overall < total_expected_results_overall) {
        bool work_done_in_iteration = false;
        for (int64_t q = 0; q < num_queries_total; ++q) {
            if (!global_topk_buffer_pool_[q]->currently_processing_query()) {
                continue;
            }

            PartitionScanResult result_item;
            // Try to dequeue one result for this query
            if (query_result_queues_[q].try_dequeue(result_item)) {
                work_done_in_iteration = true;
                total_processed_results_overall++;
                received_results_per_query[q]++;

                if (result_item.is_valid && (!result_item.distances.empty() || !result_item.indices.empty())) {
                    global_topk_buffer_pool_[q]->batch_add(
                            result_item.distances.data(),
                            result_item.indices.data(),
                            result_item.indices.size()
                    );
                } else {
                    global_topk_buffer_pool_[q]->record_empty_job(); // This also decrements jobs_left_
                }
            }
        }

        // APS Logic
        if (use_aps && duration_cast<microseconds>(high_resolution_clock::now() - last_aps_flush_time).count() > search_params->aps_flush_period_us) {
            for (int64_t q = 0; q < num_queries_total; ++q) {
                if (!global_topk_buffer_pool_[q]->currently_processing_query() || boundary_distances_all_queries[q].empty()) {
                    continue;
                }
                auto curr_buffer = global_topk_buffer_pool_[q];
                int num_partitions_effectively_scanned = curr_buffer->get_num_partitions_scanned(); // Based on non-empty batch_adds

                if (num_partitions_effectively_scanned > 0 && num_partitions_effectively_scanned < (int)boundary_distances_all_queries[q].size()) {
                    float radius = curr_buffer->get_kth_distance(); // Calls flush
                    // Simplified recall profile for now. A more robust one would map merged results to original ranks.
                    std::vector<float> probs = compute_recall_profile(boundary_distances_all_queries[q], radius, dimension, {}, search_params->use_precomputed, metric_ == faiss::METRIC_L2);

                    float estimated_recall = 0.0f;

                    // TODO adjust reordering
                    for(int i=0; i < num_partitions_effectively_scanned && i < probs.size(); ++i) {
                        estimated_recall += probs[i];
                    }

                    if (estimated_recall >= search_params->recall_target) {
                        int jobs_remaining = curr_buffer->jobs_left_.load(std::memory_order_relaxed);
                        if (jobs_remaining > 0) {
                            // Adjust total_expected_results_overall because these jobs won't be processed.
                            total_expected_results_overall -= jobs_remaining;
                            curr_buffer->record_skipped_jobs(jobs_remaining); // Sets jobs_left to 0 or less
                        }
                        curr_buffer->set_processing_query(false);
                    }
                }
            }
            last_aps_flush_time = high_resolution_clock::now();
        }
        if (!work_done_in_iteration && total_processed_results_overall < total_expected_results_overall) {
            // Avoid busy spinning if queues are temporarily empty but work is expected
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    timing_info->job_wait_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - job_processing_loop_start_time).count();

    // Aggregate results
    auto result_aggregation_start_time = high_resolution_clock::now();
    auto topk_ids = torch::full({num_queries_total, k}, -1, torch::kInt64);
    auto topk_dists = torch::full({num_queries_total, k},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                                  torch::kFloat32);
    auto ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto dists_accessor = topk_dists.accessor<float, 2>();

    for (int64_t q = 0; q < num_queries_total; ++q) {
        global_topk_buffer_pool_[q]->flush(); // Ensure final state is sorted
        std::vector<float> final_dists = global_topk_buffer_pool_[q]->get_topk();
        std::vector<int64_t> final_ids = global_topk_buffer_pool_[q]->get_topk_indices();
        int num_results_for_q = std::min((int)final_ids.size(), k);
        for (int i = 0; i < num_results_for_q; ++i) {
            ids_accessor[q][i] = final_ids[i];
            dists_accessor[q][i] = final_dists[i];
        }
        timing_info->partitions_scanned += global_topk_buffer_pool_[q]->get_num_partitions_scanned();
        // Maintenance policy hit tracking could be added here if needed,
        // based on which partitions contributed to the final top-k.
    }
    timing_info->result_aggregate_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - result_aggregation_start_time).count();

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}

std::shared_ptr<SearchResult> QueryCoordinator::search(Tensor x, std::shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::search] partition_manager_ is null.");
    }
    x = x.contiguous(); // Ensure tensor is contiguous

    auto overall_search_start_time = high_resolution_clock::now();
    std::shared_ptr<SearchTimingInfo> parent_timing_info = nullptr; // Initialize to nullptr

    Tensor partition_ids_to_scan;
    Tensor partition_distances; // For SPANN

    bool use_aps = search_params->recall_target > 0.0 && !search_params->batched_scan && parent_ != nullptr;

    if (parent_ == nullptr) {
        // Flat index or top level: scan all (or a subset of) partitions
        partition_ids_to_scan = partition_manager_->get_partition_ids(); // Gets all PIDs
        if (x.size(0) > 1 && partition_ids_to_scan.dim() == 1) { // If multiple queries and PIDs are 1D
            partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({x.size(0), partition_ids_to_scan.size(0)});
        } else if (x.size(0) == 1 && partition_ids_to_scan.dim() == 1) { // Single query, PIDs 1D
            partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0); // Make it [1, num_pids]
        }
    } else { // Hierarchical index: query parent first
        auto parent_search_params = search_params->parent_params ?
                                    search_params->parent_params :
                                    std::make_shared<SearchParams>(); // Default if not provided

        // Determine k for parent search
        if (use_aps && !search_params->batched_scan) { // APS logic (use_aps defined in worker_scan context)
            parent_search_params->k = std::max(
                    (int)(partition_manager_->nlist() * search_params->initial_search_fraction), 1
            );
        } else {
            parent_search_params->k = std::min(search_params->nprobe, (int)partition_manager_->nlist());
        }
        if (parent_search_params->k == 0 && partition_manager_->nlist() > 0) parent_search_params->k = 1; // Ensure k is at least 1 if there are partitions


        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        partition_distances = parent_search_result->distances; // For SPANN
        parent_timing_info = parent_search_result->timing_info;
    }

    // SPANN pruning (if applicable)
    if (search_params->use_spann && partition_distances.defined() && partition_distances.numel() > 0) {
        // Ensure partition_distances is not empty and has a second dimension
        if (partition_distances.size(1) > 0) {
            Tensor relative_distances = partition_distances / partition_distances.select(1, 0).unsqueeze(1); // Use unsqueeze(1) for broadcasting
            Tensor mask = relative_distances.ge(search_params->spann_eps); // Greater than or equal
            partition_ids_to_scan.masked_fill_(mask, -1);
        } else {
            // std::cout << "[QueryCoordinator::search] SPANN: partition_distances has no second dimension, skipping pruning." << std::endl;
        }
    }

    auto scan_result = scan_partitions(x, partition_ids_to_scan, search_params);

    if (parent_timing_info) { // Check if parent_timing_info was populated
        scan_result->timing_info->parent_info = parent_timing_info;
    }
    scan_result->timing_info->total_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - overall_search_start_time).count();

    return scan_result;
}


std::shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids_to_scan, std::shared_ptr<SearchParams> search_params) {
    if (workers_initialized_) {
        // if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using worker-based scan." << std::endl;
        return worker_scan(x, partition_ids_to_scan, search_params);
    } else {
        if (search_params->batched_scan) { // This flag indicates if the *lowest level scan* should be batched
            // if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using batched serial scan." << std::endl;
            return batched_serial_scan(x, partition_ids_to_scan, search_params);
        } else {
            // if (debug_) std::cout << "[QueryCoordinator::scan_partitions] Using serial scan." << std::endl;
            return serial_scan(x, partition_ids_to_scan, search_params);
        }
    }
}

std::shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids_to_scan_all_queries,
                                                            std::shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0,search_params->k}, torch::kInt64);
        empty_result->distances = torch::empty({0,search_params->k}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        empty_result->timing_info->n_queries = 0;
        return empty_result;
    }

    auto start_time = high_resolution_clock::now();

    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = search_params->k;

    auto ret_ids = torch::full({num_queries, k}, -1L, torch::kInt64);
    auto ret_dists = torch::full({num_queries, k},
                                 (metric_ == faiss::METRIC_INNER_PRODUCT) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                                 torch::kFloat32);

    auto ids_acc = ret_ids.accessor<int64_t,2>();
    auto dists_acc = ret_dists.accessor<float,2>();

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = partition_manager_->nlist();
    timing_info->search_params = search_params;
    timing_info->partitions_scanned = 0;

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    bool use_aps = (search_params->recall_target > 0.0 && !search_params->batched_scan && parent_ != nullptr);


    if (partition_ids_to_scan_all_queries.dim() == 1) {
        partition_ids_to_scan_all_queries = partition_ids_to_scan_all_queries.unsqueeze(0).expand({num_queries, partition_ids_to_scan_all_queries.size(0)});
    }
    auto p_ids_acc = partition_ids_to_scan_all_queries.accessor<int64_t, 2>();
    float *x_ptr = x.data_ptr<float>();

    std::atomic<long> total_partitions_scanned_atomic{0};

    // If search_params->num_threads > 1, parallelize over queries. Otherwise, serial loop.
    int num_threads_for_queries = (search_params->num_threads > 0) ? search_params->num_threads : 1;


    parallel_for<int64_t>(0, num_queries, [&](int64_t q_idx) {
        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending);
        const float* query_vec_ptr = x_ptr + q_idx * dimension;
        int num_partitions_to_consider_for_this_query = partition_ids_to_scan_all_queries.size(1);

        std::vector<float> boundary_distances_for_q;
        std::vector<int64_t> pids_in_aps_order_for_q;
        std::unordered_map<int64_t, int> pid_to_aps_rank_for_q;


        if (use_aps) {
            for(int p_rank=0; p_rank < num_partitions_to_consider_for_this_query; ++p_rank) {
                int64_t pid = p_ids_acc[q_idx][p_rank];
                if (pid != -1) pids_in_aps_order_for_q.push_back(pid);
            }
            if(!pids_in_aps_order_for_q.empty() && parent_ && parent_->partition_manager_){
                std::vector<float*> cluster_centroids = parent_->partition_manager_->get_vectors(pids_in_aps_order_for_q);
                std::vector<float*> valid_centroids;
                std::vector<int64_t> pids_for_valid_centroids;

                for(size_t i=0; i<cluster_centroids.size(); ++i) {
                    if(cluster_centroids[i] != nullptr) {
                        valid_centroids.push_back(cluster_centroids[i]);
                        pids_for_valid_centroids.push_back(pids_in_aps_order_for_q[i]);
                    }
                }

                if(!valid_centroids.empty()){
                    boundary_distances_for_q = compute_boundary_distances(x[q_idx], valid_centroids, metric_ == faiss::METRIC_L2);
                    // Update pids_in_aps_order_for_q to only those that had valid centroids and thus have a boundary distance
                    pids_in_aps_order_for_q = pids_for_valid_centroids;
                    for(size_t i=0; i < pids_in_aps_order_for_q.size(); ++i) {
                        pid_to_aps_rank_for_q[pids_in_aps_order_for_q[i]] = i;
                    }
                } else {
                    pids_in_aps_order_for_q.clear(); // No valid centroids, so no APS possible with this data
                }
            } else {
                pids_in_aps_order_for_q.clear(); // Conditions for APS not met
            }
        }

        int scanned_count_for_q = 0;
        std::vector<bool> aps_candidate_scanned_flags;
        if (use_aps && !pids_in_aps_order_for_q.empty()) {
            aps_candidate_scanned_flags.resize(pids_in_aps_order_for_q.size(), false);
        }

        for (int p_orig_rank = 0; p_orig_rank < num_partitions_to_consider_for_this_query; ++p_orig_rank) {
            int64_t pid = p_ids_acc[q_idx][p_orig_rank];
            if (pid == -1) continue;

            const float *list_vectors = nullptr;
            const int64_t *list_ids_ptr = nullptr;
            int64_t list_size = 0;
            try {
                list_vectors = (float *)partition_manager_->partition_store_->get_codes(pid);
                list_ids_ptr = partition_manager_->partition_store_->get_ids(pid);
                list_size = partition_manager_->partition_store_->list_size(pid);
            } catch (const std::runtime_error& e) { continue; }


            if (list_size > 0 && list_vectors != nullptr) {
                scan_list(query_vec_ptr, list_vectors, list_ids_ptr, list_size, dimension, *topk_buf, metric_);
                scanned_count_for_q++;
                if (use_aps) {
                    auto it = pid_to_aps_rank_for_q.find(pid);
                    if (it != pid_to_aps_rank_for_q.end()) {
                        int aps_rank = it->second;
                        if (aps_rank < static_cast<int>(aps_candidate_scanned_flags.size())) {
                            aps_candidate_scanned_flags[aps_rank] = true;
                        }
                    }
                }
            }

            if (use_aps && !boundary_distances_for_q.empty() && scanned_count_for_q > 0) {
                // APS check should happen periodically, not necessarily after every single partition.
                // For serial scan, checking after each scan is fine.
                float radius = topk_buf->get_kth_distance();
                std::vector<float> probs = compute_recall_profile(boundary_distances_for_q, radius, dimension, {}, search_params->use_precomputed, metric_ == faiss::METRIC_L2);

                float estimated_recall = 0.0f;
                for(size_t aps_rank_idx = 0; aps_rank_idx < aps_candidate_scanned_flags.size(); ++aps_rank_idx) {
                    if (aps_candidate_scanned_flags[aps_rank_idx] && aps_rank_idx < probs.size()) {
                        estimated_recall += probs[aps_rank_idx];
                    }
                }

                if (estimated_recall >= search_params->recall_target) {
                    break;
                }
            }
        }
        total_partitions_scanned_atomic += scanned_count_for_q;

        std::vector<float> final_dists = topk_buf->get_topk();
        std::vector<int64_t> final_ids = topk_buf->get_topk_indices();
        int num_results = std::min((int)final_ids.size(), k);
        for(int i=0; i<num_results; ++i) {
            ids_acc[q_idx][i] = final_ids[i];
            dists_acc[q_idx][i] = final_dists[i];
        }
    }, num_threads_for_queries);

    timing_info->partitions_scanned = total_partitions_scanned_atomic.load();
    timing_info->total_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count();

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = ret_ids;
    search_result->distances = ret_dists;
    search_result->timing_info = timing_info;
    return search_result;
}


std::shared_ptr<SearchResult> QueryCoordinator::batched_serial_scan(
        Tensor x,
        Tensor partition_ids_to_scan_all_queries,
        std::shared_ptr<SearchParams> search_params) {
    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::batched_serial_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0,search_params->k}, torch::kInt64);
        empty_result->distances = torch::empty({0,search_params->k}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        empty_result->timing_info->n_queries = 0;
        return empty_result;
    }

    auto start_time = high_resolution_clock::now();
    int64_t num_queries = x.size(0);
    int k_val = search_params->k;
    int64_t dim = x.size(1);

    std::vector<std::shared_ptr<TopkBuffer>> global_query_buffers =
            create_buffers(num_queries, k_val, (metric_ == faiss::METRIC_INNER_PRODUCT));

    if (partition_ids_to_scan_all_queries.dim() == 1) {
        partition_ids_to_scan_all_queries = partition_ids_to_scan_all_queries.unsqueeze(0).expand({num_queries, partition_ids_to_scan_all_queries.size(0)});
    }
    auto p_ids_acc = partition_ids_to_scan_all_queries.accessor<int64_t, 2>();

    std::unordered_map<int64_t, std::vector<int64_t>> p_to_q_map;
    std::atomic<long> total_scans_performed_atomic{0}; // Sum of (num_queries_for_pid * 1_partition)

    for (int64_t q_idx = 0; q_idx < num_queries; ++q_idx) {
        for (int64_t p_col = 0; p_col < partition_ids_to_scan_all_queries.size(1); ++p_col) {
            int64_t pid = p_ids_acc[q_idx][p_col];
            if (pid != -1) {
                p_to_q_map[pid].push_back(q_idx);
            }
        }
    }

    std::vector<std::pair<int64_t, std::vector<int64_t>>> p_to_q_vec;
    p_to_q_vec.reserve(p_to_q_map.size());
    for(const auto& entry : p_to_q_map) {
        p_to_q_vec.push_back(entry);
    }

    int num_threads_for_partitions = (search_params->num_threads > 0) ? search_params->num_threads : 1;


    parallel_for<int64_t>(0, p_to_q_vec.size(), [&](int64_t i) {
        int64_t current_pid = p_to_q_vec[i].first;
        const std::vector<int64_t>& query_indices_for_this_pid = p_to_q_vec[i].second;

        if (query_indices_for_this_pid.empty()) return;

        const float *list_codes_ptr = nullptr;
        const int64_t *list_ids_ptr = nullptr;
        int64_t list_size = 0;
        int64_t current_d = 0;

        try {
            list_codes_ptr = (float *)partition_manager_->partition_store_->get_codes(current_pid);
            list_ids_ptr = partition_manager_->partition_store_->get_ids(current_pid);
            list_size = partition_manager_->partition_store_->list_size(current_pid);
            current_d = partition_manager_->d();
        } catch (const std::runtime_error& e) { return; }


        if (list_size == 0 || list_codes_ptr == nullptr) return;

        Tensor x_subset = torch::empty({(int64_t)query_indices_for_this_pid.size(), dim}, x.options());
        for(size_t q_sub_idx = 0; q_sub_idx < query_indices_for_this_pid.size(); ++q_sub_idx) {
            x_subset[q_sub_idx] = x[query_indices_for_this_pid[q_sub_idx]];
        }

        std::vector<std::shared_ptr<TopkBuffer>> local_scan_buffers =
                create_buffers(query_indices_for_this_pid.size(), k_val, (metric_ == faiss::METRIC_INNER_PRODUCT));

        batched_scan_list(x_subset.data_ptr<float>(),
                          list_codes_ptr, list_ids_ptr,
                          query_indices_for_this_pid.size(),
                          list_size, current_d,
                          local_scan_buffers, metric_);

        total_scans_performed_atomic += query_indices_for_this_pid.size();

        for (size_t q_sub_idx = 0; q_sub_idx < query_indices_for_this_pid.size(); ++q_sub_idx) {
            int64_t global_q_idx = query_indices_for_this_pid[q_sub_idx];
            std::vector<float> dists = local_scan_buffers[q_sub_idx]->get_topk();
            std::vector<int64_t> ids = local_scan_buffers[q_sub_idx]->get_topk_indices();
            if (!ids.empty()) {
                global_query_buffers[global_q_idx]->batch_add(dists.data(), ids.data(), ids.size());
            } else { // Still record that this query processed this (empty) partition for its accounting
                global_query_buffers[global_q_idx]->record_empty_job();
            }
        }
    }, num_threads_for_partitions);

    auto topk_ids = torch::full({num_queries, k_val}, -1L, torch::kInt64);
    auto topk_dists = torch::full({num_queries, k_val},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                                  torch::kFloat32);
    auto ids_acc = topk_ids.accessor<int64_t,2>();
    auto dists_acc = topk_dists.accessor<float,2>();

    long total_partitions_merged_in_buffers = 0;
    for(int64_t q_idx=0; q_idx < num_queries; ++q_idx) {
        global_query_buffers[q_idx]->flush();
        std::vector<float> final_dists = global_query_buffers[q_idx]->get_topk();
        std::vector<int64_t> final_ids = global_query_buffers[q_idx]->get_topk_indices();
        int num_results = std::min((int)final_ids.size(), k_val);
        for(int i=0; i<num_results; ++i) {
            ids_acc[q_idx][i] = final_ids[i];
            dists_acc[q_idx][i] = final_dists[i];
        }
        total_partitions_merged_in_buffers += global_query_buffers[q_idx]->get_num_partitions_scanned();
    }

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->search_params = search_params;
    // timing_info->partitions_scanned: For batched serial, each query effectively "sees" results from
    // the partitions it was assigned. total_scans_performed_atomic counts (query, partition_scan) events.
    // A good average is total_scans_performed_atomic / num_queries.
    if (num_queries > 0) {
        timing_info->partitions_scanned = total_scans_performed_atomic.load() / num_queries;
    } else {
        timing_info->partitions_scanned = 0;
    }
    timing_info->total_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count();

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}