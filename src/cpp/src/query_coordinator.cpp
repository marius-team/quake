// query_coordinator.cpp

#include "query_coordinator.h"
#include <sys/fcntl.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm> // For std::remove, std::min
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
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
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
            std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
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
        ScanJob termination_job;
        termination_job.partition_id = -1;
        core_resources_[i].job_queue.enqueue(termination_job);
    }

    for (auto &thr : worker_threads_) {
        if (thr.joinable()) {
            thr.join();
        }
    }
    worker_threads_.clear();
    workers_initialized_ = false;
}

void QueryCoordinator::partition_scan_worker_fn(int core_index) {
    CoreResources &res = core_resources_[core_index];

    if (!set_thread_affinity(core_index)) {
        std::cout << "[QueryCoordinator::partition_scan_worker_fn] Failed to set thread affinity on core " << core_index << std::endl;
    }

    while (true) {
        ScanJob job;
        res.job_queue.wait_dequeue(job);

        if (job.partition_id == -1 || stop_workers_.load(std::memory_order_relaxed)) {
            break;
        }

        if (job.query_ids.empty()) {
            // std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Job for partition " << job.partition_id << " has no query_ids." << std::endl;
            continue;
        }

        // Check if the primary query this job might serve is still active in the global pool
        // This is a best-effort check to potentially avoid work if APS has already terminated the query.
        // The coordinator will ultimately discard results if the query is no longer active.
        if (job.query_ids[0] < static_cast<int64_t>(global_topk_buffer_pool_.size()) &&
            global_topk_buffer_pool_[job.query_ids[0]] &&
            !global_topk_buffer_pool_[job.query_ids[0]]->currently_processing_query()) {
            // Enqueue an empty/invalid result to signal structural completion for all affected queries.
            for(int64_t q_id : job.query_ids) {
                if (q_id < 0 || q_id >= static_cast<int64_t>(query_result_queues_.size())) continue;
                query_result_queues_[q_id].enqueue(PartitionScanResult(q_id, job.partition_id, false));
            }
            continue;
        }

        const float *partition_codes_ptr = nullptr;
        const int64_t *partition_ids_ptr = nullptr;
        int64_t partition_size = 0;

        try {
            partition_codes_ptr = (float *)partition_manager_->partition_store_->get_codes(job.partition_id);
            partition_ids_ptr = (int64_t *)partition_manager_->partition_store_->get_ids(job.partition_id);
            partition_size = partition_manager_->partition_store_->list_size(job.partition_id);
        } catch (const std::runtime_error &e) {
            // std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Error accessing partition " << job.partition_id << ": " << e.what() << std::endl;
            for(int64_t q_id : job.query_ids) {
                if (q_id < 0 || q_id >= static_cast<int64_t>(query_result_queues_.size())) continue;
                query_result_queues_[q_id].enqueue(PartitionScanResult(q_id, job.partition_id, false));
            }
            continue;
        }

        if (partition_size == 0 || partition_codes_ptr == nullptr) {
            for(int64_t q_id : job.query_ids) {
                if (q_id < 0 || q_id >= static_cast<int64_t>(query_result_queues_.size())) continue;
                query_result_queues_[q_id].enqueue(PartitionScanResult(q_id, job.partition_id, {}, {}));
            }
            continue;
        }

        int current_dim = partition_manager_->d();

        if (!job.is_batched) {
            int64_t global_query_id = job.query_ids[0];
            if (global_query_id < 0 || global_query_id >= static_cast<int64_t>(query_result_queues_.size())) {
                // std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Invalid global_query_id " << global_query_id << " for non-batched job." << std::endl;
                continue;
            }

            size_t required_query_buffer_size = static_cast<size_t>(current_dim * sizeof(float));
            if (res.local_query_buffer.size() < required_query_buffer_size) {
                res.local_query_buffer.resize(required_query_buffer_size);
            }
            std::memcpy(res.local_query_buffer.data(), job.query_vector, required_query_buffer_size);

            if (res.topk_buffer_pool.empty()) {
                res.topk_buffer_pool.push_back(std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT));
            }
            std::shared_ptr<TopkBuffer> local_topk_buffer = res.topk_buffer_pool[0];
            local_topk_buffer->set_k(job.k);
            local_topk_buffer->reset();

            scan_list((float *)res.local_query_buffer.data(),
                      partition_codes_ptr, partition_ids_ptr,
                      partition_size, current_dim,
                      *local_topk_buffer, metric_);

            query_result_queues_[global_query_id].enqueue(
                    PartitionScanResult(global_query_id, job.partition_id, local_topk_buffer->get_topk(), local_topk_buffer->get_topk_indices())
            );

        } else {
            size_t required_query_batch_buffer_size = static_cast<size_t>(job.num_queries * current_dim * sizeof(float));
            if (res.local_query_buffer.size() < required_query_batch_buffer_size) {
                res.local_query_buffer.resize(required_query_batch_buffer_size);
            }

            for (int i = 0; i < job.num_queries; ++i) {
                int64_t global_q_idx = job.query_ids[i];
                std::memcpy(res.local_query_buffer.data() + i * current_dim * sizeof(float),
                            job.query_vector + global_q_idx * current_dim,
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

            batched_scan_list((float*)res.local_query_buffer.data(),
                              partition_codes_ptr, partition_ids_ptr,
                              job.num_queries,
                              partition_size, current_dim,
                              res.topk_buffer_pool,
                              metric_);

            for (int i = 0; i < job.num_queries; ++i) {
                int64_t global_q_idx = job.query_ids[i];
                if (global_q_idx < 0 || global_q_idx >= static_cast<int64_t>(query_result_queues_.size())) {
                    // std::cerr << "[QueryCoordinator::partition_scan_worker_fn] Invalid global query_id " << global_q_idx << " in batched job result enqueue." << std::endl;
                    continue;
                }
                query_result_queues_[global_q_idx].enqueue(
                        PartitionScanResult(global_q_idx, job.partition_id, res.topk_buffer_pool[i]->get_topk(), res.topk_buffer_pool[i]->get_topk_indices())
                );
            }
        }
    }
}

std::shared_ptr<SearchResult> QueryCoordinator::worker_scan(
        Tensor x,
        Tensor partition_ids_to_scan_all_queries,
        std::shared_ptr<SearchParams> search_params) {

    if (!partition_manager_) {
        throw std::runtime_error("[QueryCoordinator::worker_scan] partition_manager_ is null.");
    }
    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0, search_params->k}, torch::kInt64);
        empty_result->distances = torch::empty({0, search_params->k}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        empty_result->timing_info->n_queries = 0;
        return empty_result;
    }

    int64_t num_queries_total = x.size(0);
    int64_t dimension = x.size(1);
    int k = search_params->k;
    // This `use_aps` flag is for the current level being scanned by workers
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

    {
        std::lock_guard<std::mutex> lock_pools(global_pool_mutex_);
        std::lock_guard<std::mutex> lock_queues(result_queues_mutex_);

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

            PartitionScanResult ignored_item;
            while(query_result_queues_[q].try_dequeue(ignored_item));
        }
    }
    timing_info->buffer_init_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start_overall_scan_time).count();

    auto job_enqueue_start_time = high_resolution_clock::now();
    std::vector<int> expected_results_per_query(num_queries_total, 0);
    std::vector<int> received_results_per_query(num_queries_total, 0);
    // For APS: Store the original list of PIDs to scan for each query to map back for recall profile
    std::vector<std::vector<int64_t>> pids_for_aps_per_query(num_queries_total);


    if (search_params->batched_scan) { // This is about how jobs are sent to workers for *this level*
        std::unordered_map<int64_t, std::vector<int64_t>> partition_to_queries_map;
        for (int64_t q_global = 0; q_global < num_queries_total; ++q_global) {
            int valid_jobs_for_this_q = 0;
            for (int64_t p_idx = 0; p_idx < partition_ids_to_scan_all_queries.size(1); ++p_idx) {
                int64_t pid = p_ids_acc[q_global][p_idx];
                if (pid == -1) continue;
                partition_to_queries_map[pid].push_back(q_global);
                if(use_aps) pids_for_aps_per_query[q_global].push_back(pid);
                valid_jobs_for_this_q++;
            }
            expected_results_per_query[q_global] = valid_jobs_for_this_q;
            if (q_global < static_cast<int64_t>(global_topk_buffer_pool_.size()) && global_topk_buffer_pool_[q_global]) {
                global_topk_buffer_pool_[q_global]->set_jobs_left(valid_jobs_for_this_q);
            }
        }

        for(auto const& [pid, query_indices_for_pid] : partition_to_queries_map) {
            ScanJob job;
            job.is_batched = true; // Worker will process this job as a batch of queries for one partition
            job.partition_id = pid;
            job.k = k;
            job.query_vector = x_ptr;
            job.num_queries = query_indices_for_pid.size();
            job.query_ids = query_indices_for_pid; // Global indices of queries

            int core_id = partition_manager_->get_partition_core_id(pid);
            if (core_id < 0 || core_id >= num_workers_) core_id = pid % std::max(1, num_workers_);
            core_resources_[core_id].job_queue.enqueue(job);
        }
    } else {
        for (int64_t q_global = 0; q_global < num_queries_total; ++q_global) {
            int valid_jobs_for_this_q = 0;
            for (int64_t p_idx = 0; p_idx < partition_ids_to_scan_all_queries.size(1); ++p_idx) {
                int64_t pid = p_ids_acc[q_global][p_idx];
                if (pid == -1) continue;
                if(use_aps) pids_for_aps_per_query[q_global].push_back(pid);
                valid_jobs_for_this_q++;
                ScanJob job;
                job.is_batched = false;
                job.query_ids = {q_global};
                job.partition_id = pid;
                job.k = k;
                job.query_vector = x_ptr + q_global * dimension;
                job.num_queries = 1;

                int core_id = partition_manager_->get_partition_core_id(pid);
                if (core_id < 0 || core_id >= num_workers_) core_id = pid % std::max(1, num_workers_);
                core_resources_[core_id].job_queue.enqueue(job);
            }
            expected_results_per_query[q_global] = valid_jobs_for_this_q;
            if (q_global < static_cast<int64_t>(global_topk_buffer_pool_.size()) && global_topk_buffer_pool_[q_global]) {
                global_topk_buffer_pool_[q_global]->set_jobs_left(valid_jobs_for_this_q);
            }
        }
    }
    timing_info->job_enqueue_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - job_enqueue_start_time).count();

    auto last_aps_flush_time = high_resolution_clock::now();
    std::vector<std::vector<float>> boundary_distances_all_queries(num_queries_total);
    // Map from global query index to (map from original_pid to its rank in the pids_for_aps_per_query[q] list)
    std::vector<std::unordered_map<int64_t, int>> pid_to_rank_for_aps_per_query(num_queries_total);


    if (use_aps) { // Precompute boundary distances based on the actual PIDs chosen for each query
        for (int64_t q = 0; q < num_queries_total; ++q) {
            if (!pids_for_aps_per_query[q].empty() && parent_ && parent_->partition_manager_) {
                std::vector<float*> cluster_centroids = parent_->partition_manager_->get_vectors(pids_for_aps_per_query[q]);
                std::vector<float*> valid_centroids;
                std::vector<int64_t> valid_pids_for_centroids; // Store PIDs corresponding to valid_centroids

                for(size_t i=0; i < cluster_centroids.size(); ++i) {
                    if(cluster_centroids[i] != nullptr) {
                        valid_centroids.push_back(cluster_centroids[i]);
                        valid_pids_for_centroids.push_back(pids_for_aps_per_query[q][i]);
                    }
                }

                if (!valid_centroids.empty()) {
                    boundary_distances_all_queries[q] = compute_boundary_distances(x[q], valid_centroids, metric_ == faiss::METRIC_L2);
                    // Build the pid_to_rank map for this query
                    for(size_t i=0; i < valid_pids_for_centroids.size(); ++i) {
                        pid_to_rank_for_aps_per_query[q][valid_pids_for_centroids[i]] = i;
                    }
                } else if (!pids_for_aps_per_query[q].empty()) {
                    // std::cerr << "[QueryCoordinator::worker_scan] APS: No valid centroids for query " << q << "." << std::endl;
                }
            }
        }
    }

    auto job_processing_loop_start_time = high_resolution_clock::now();
    int total_expected_results_overall = 0;
    for(int64_t q=0; q<num_queries_total; ++q) total_expected_results_overall += expected_results_per_query[q];

    std::vector<bool> query_aps_terminated_flags(num_queries_total, false); // Use regular bool vector, not atomic here

    int total_received_results_overall = 0;
    // Store which PIDs have been merged for each query for accurate APS recall estimation
    std::vector<std::vector<bool>> merged_partition_flags_for_aps(num_queries_total);
    if(use_aps) {
        for(int64_t q=0; q < num_queries_total; ++q) {
            if(!boundary_distances_all_queries[q].empty()){
                merged_partition_flags_for_aps[q].resize(boundary_distances_all_queries[q].size(), false);
            }
        }
    }


    while(total_received_results_overall < total_expected_results_overall) {
        bool work_done_in_iteration = false;
        for (int64_t q = 0; q < num_queries_total; ++q) {
            if (query_aps_terminated_flags[q] ||
                (global_topk_buffer_pool_[q] && !global_topk_buffer_pool_[q]->currently_processing_query())) {
                continue;
            }

            if (received_results_per_query[q] < expected_results_per_query[q]) {
                PartitionScanResult result_item;
                if (query_result_queues_[q].try_dequeue(result_item)) {
                    work_done_in_iteration = true;

                    if (global_topk_buffer_pool_[q]->currently_processing_query()) {
                        if (result_item.is_valid) { // is_valid is true if scan happened, even if empty results
                            if(!result_item.distances.empty() || !result_item.indices.empty()) {
                                global_topk_buffer_pool_[q]->batch_add(
                                        result_item.distances.data(),
                                        result_item.indices.data(),
                                        result_item.indices.size()
                                );
                            } else { // Valid scan, but no results from partition (e.g. empty partition)
                                global_topk_buffer_pool_[q]->record_empty_job();
                            }
                            // For APS, mark this partition (if it's in the APS candidate list) as processed
                            if(use_aps && !pid_to_rank_for_aps_per_query[q].empty()) {
                                auto it = pid_to_rank_for_aps_per_query[q].find(result_item.original_partition_id);
                                if (it != pid_to_rank_for_aps_per_query[q].end()) {
                                    int rank = it->second;
                                    if (rank < static_cast<int>(merged_partition_flags_for_aps[q].size())) {
                                        merged_partition_flags_for_aps[q][rank] = true;
                                    }
                                }
                            }
                        } else { // is_valid is false, meaning scan itself failed for this partition (e.g. couldn't access data)
                            global_topk_buffer_pool_[q]->record_empty_job(); // Still counts as a job attempt
                        }
                    }
                    // Increment received_results_per_query[q] only after processing the item from the queue.
                    // This must happen regardless of whether the query was APS terminated *between enqueue and dequeue*.
                    // The total_received_results_overall ensures the loop terminates.
                    if (received_results_per_query[q] < expected_results_per_query[q]) { // check to prevent over-increment if APS terminates mid-way
                        received_results_per_query[q]++;
                        total_received_results_overall++;
                    }
                }
            }
        }

        if (use_aps && duration_cast<microseconds>(high_resolution_clock::now() - last_aps_flush_time).count() > search_params->aps_flush_period_us) {
            for (int64_t q = 0; q < num_queries_total; ++q) {
                if (query_aps_terminated_flags[q] || !global_topk_buffer_pool_[q]->currently_processing_query() || boundary_distances_all_queries[q].empty()) {
                    continue;
                }
                auto curr_buffer = global_topk_buffer_pool_[q];
                // We need get_num_partitions_scanned() to reflect actual data merges for APS decision
                int num_partitions_merged_for_recall_calc = 0;
                for(bool flag : merged_partition_flags_for_aps[q]) {
                    if (flag) num_partitions_merged_for_recall_calc++;
                }


                if (num_partitions_merged_for_recall_calc > 0) {
                    float radius = curr_buffer->get_kth_distance();

                    std::vector<float> probs = compute_recall_profile(
                            boundary_distances_all_queries[q],
                            radius, dimension,
                            {}, // partition_sizes_vec - can be omitted if compute_recall_profile handles it or doesn't need it
                            search_params->use_precomputed,
                            metric_ == faiss::METRIC_L2
                    );

                    float estimated_recall = 0.0f;
                    for(size_t i=0; i < merged_partition_flags_for_aps[q].size() && i < probs.size(); ++i) {
                        if(merged_partition_flags_for_aps[q][i]) { // Only sum probs for partitions whose results have been merged
                            estimated_recall += probs[i];
                        }
                    }

                    if (estimated_recall >= search_params->recall_target) {
                        curr_buffer->set_processing_query(false); // Stop this query
                        query_aps_terminated_flags[q] = true;

                        // Calculate jobs that were dispatched but their results will now be ignored/not pulled
                        int jobs_dispatched_for_q = expected_results_per_query[q];
                        int jobs_already_received_for_q = received_results_per_query[q];
                        int jobs_to_ignore_or_drain = jobs_dispatched_for_q - jobs_already_received_for_q;

                        total_expected_results_overall -= jobs_to_ignore_or_drain; // Adjust overall counter

                        // Correctly update jobs_left in the buffer
                        // It should reflect how many of the *initial* jobs are now skipped.
                        // current jobs_left = initial_jobs_left - (merged_valid + merged_empty)
                        // We want to skip all jobs that are not yet merged.
                        int current_jobs_left_in_buffer = curr_buffer->jobs_left_.load(std::memory_order_relaxed);
                        if (current_jobs_left_in_buffer > 0) {
                            curr_buffer->record_skipped_jobs(current_jobs_left_in_buffer);
                        }
                    }
                }
            }
            last_aps_flush_time = high_resolution_clock::now();
        }
        if (!work_done_in_iteration && total_received_results_overall < total_expected_results_overall) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    timing_info->job_wait_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - job_processing_loop_start_time).count();

    auto result_aggregation_start_time = high_resolution_clock::now();
    auto topk_ids = torch::full({num_queries_total, k}, -1L, torch::kInt64);
    auto topk_dists = torch::full({num_queries_total, k},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                                  torch::kFloat32);
    auto ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto dists_accessor = topk_dists.accessor<float, 2>();

    for (int64_t q = 0; q < num_queries_total; ++q) {
        if (global_topk_buffer_pool_[q]) {
            global_topk_buffer_pool_[q]->flush();
            std::vector<float> final_dists = global_topk_buffer_pool_[q]->get_topk();
            std::vector<int64_t> final_ids = global_topk_buffer_pool_[q]->get_topk_indices();
            int num_results_for_q = std::min((int)final_ids.size(), k);
            for (int i = 0; i < num_results_for_q; ++i) {
                ids_accessor[q][i] = final_ids[i];
                dists_accessor[q][i] = final_dists[i];
            }
            timing_info->partitions_scanned += global_topk_buffer_pool_[q]->get_num_partitions_scanned();
        }
    }
    timing_info->result_aggregate_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - result_aggregation_start_time).count();

    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}


std::shared_ptr<SearchResult> QueryCoordinator::search(Tensor x, std::shared_ptr<SearchParams> search_params) {
    if (!partition_manager_ && workers_initialized_) {
        throw std::runtime_error("[QueryCoordinator::search] partition_manager_ is null but workers are initialized.");
    }
    x = x.contiguous();

    auto overall_search_start_time = high_resolution_clock::now();
    std::shared_ptr<SearchTimingInfo> parent_timing_info = nullptr;

    Tensor partition_ids_to_scan;
    Tensor partition_distances;

    bool use_aps_for_current_level = (search_params->recall_target > 0.0 &&
                                      (search_params->batched_scan == false) && // APS usually for non-batched at current level
                                      parent_ != nullptr);


    if (parent_ == nullptr) {
        if (!partition_manager_) throw std::runtime_error("Search called on coordinator with no parent and no partition manager.");

        int num_p_to_get = partition_manager_->nlist();
        if (search_params->nprobe > 0 && search_params->nprobe < num_p_to_get) {
            num_p_to_get = search_params->nprobe;
            // TODO: Implement a way to get top NPROBE PIDs if parent is null (e.g. all, or first N, or random N)
            // For now, get_partition_ids() returns all. We might need to slice it.
            std::cout << "[QueryCoordinator::search] Warning: nprobe < total partitions for flat index. Scanning all partitions for now." << std::endl;
        }
        partition_ids_to_scan = partition_manager_->get_partition_ids();

        if (partition_ids_to_scan.numel() == 0 && partition_manager_->nlist() > 0) {
            std::cerr << "[QueryCoordinator::search] Warning: get_partition_ids() is empty but nlist=" << partition_manager_->nlist() << std::endl;
        }

        if (x.size(0) >= 1 && partition_ids_to_scan.dim() == 1) {
            partition_ids_to_scan = partition_ids_to_scan.unsqueeze(0).expand({x.size(0), partition_ids_to_scan.size(0)});
        } else if (partition_ids_to_scan.numel() == 0 && x.size(0) >=1 ){
            partition_ids_to_scan = torch::empty({x.size(0), 0}, torch::kInt64);
        }
    } else {
        auto parent_search_params = search_params->parent_params ?
                                    search_params->parent_params :
                                    std::make_shared<SearchParams>();

        if (parent_search_params->k == 0) {
            if (use_aps_for_current_level) {
                parent_search_params->k = std::max(
                        (int)( (partition_manager_ ? partition_manager_->nlist() : 1) * search_params->initial_search_fraction), 1
                );
            } else {
                int nlist_current = partition_manager_ ? partition_manager_->nlist() : 1;
                if (nlist_current == 0 && search_params->nprobe > 0) nlist_current = search_params->nprobe; // if current level has no list but nprobe is set.
                else if (nlist_current == 0) nlist_current = 1;


                parent_search_params->k = std::min(search_params->nprobe, nlist_current);
            }
        }
        if (parent_search_params->k == 0) parent_search_params->k = 1;


        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        partition_distances = parent_search_result->distances;
        parent_timing_info = parent_search_result->timing_info;
    }

    if (search_params->use_spann && partition_distances.defined() && partition_distances.numel() > 0 && partition_ids_to_scan.numel() > 0) {
        if (partition_distances.size(0) == partition_ids_to_scan.size(0) && partition_distances.size(1) == partition_ids_to_scan.size(1) && partition_distances.size(1) > 0) {
            Tensor first_dist = partition_distances.select(1, 0).unsqueeze(1);
            if (metric_ == faiss::METRIC_L2) first_dist = torch::clamp_min(first_dist, 1e-9f); // Avoid div by zero for L2
            else if (metric_ == faiss::METRIC_INNER_PRODUCT) {
                // For IP, if first_dist is 0 or negative, this relative logic is problematic.
                // SPANN is more designed for L2. We might need to skip if first_dist is not positive.
                // Or use absolute values if that makes sense for SPANN with IP.
                // For now, proceed with caution for IP.
            }

            Tensor relative_distances = partition_distances / (first_dist + 1e-9f); // Add epsilon for stability
            Tensor mask = relative_distances.ge(search_params->spann_eps);
            partition_ids_to_scan.masked_fill_(mask, -1);
        } else {
            // std::cout << "[QueryCoordinator::search] SPANN: Mismatch in dimensions or empty distances/pids, skipping pruning." << std::endl;
        }
    }

    auto scan_result = scan_partitions(x, partition_ids_to_scan, search_params);

    if (parent_timing_info) {
        scan_result->timing_info->parent_info = parent_timing_info;
    }
    scan_result->timing_info->total_time_ns = duration_cast<nanoseconds>(high_resolution_clock::now() - overall_search_start_time).count();

    return scan_result;
}


std::shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids_to_scan, std::shared_ptr<SearchParams> search_params) {
    if (workers_initialized_) {
        return worker_scan(x, partition_ids_to_scan, search_params);
    } else {
        if (search_params->batched_scan) {
            return batched_serial_scan(x, partition_ids_to_scan, search_params);
        } else {
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
    timing_info->partitions_scanned = 0; // Initialize

    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    // APS for serial scan uses the same conditions as worker_scan
    bool use_aps = (search_params->recall_target > 0.0 && !search_params->batched_scan && parent_ != nullptr);


    if (partition_ids_to_scan_all_queries.dim() == 1) {
        partition_ids_to_scan_all_queries = partition_ids_to_scan_all_queries.unsqueeze(0).expand({num_queries, partition_ids_to_scan_all_queries.size(0)});
    }
    auto p_ids_acc = partition_ids_to_scan_all_queries.accessor<int64_t, 2>();
    float *x_ptr = x.data_ptr<float>();

    std::atomic<long> total_partitions_scanned_atomic{0};

    parallel_for<int64_t>(0, num_queries, [&](int64_t q_idx) {
        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending);
        const float* query_vec_ptr = x_ptr + q_idx * dimension;
        int num_partitions_for_this_query = partition_ids_to_scan_all_queries.size(1);

        std::vector<float> boundary_distances_for_q;
        std::vector<int64_t> pids_in_aps_order_for_q; // Store PIDs in the order they appear for boundary_distances

        if (use_aps) {
            for(int p=0; p<num_partitions_for_this_query; ++p) {
                if (p_ids_acc[q_idx][p] != -1) pids_in_aps_order_for_q.push_back(p_ids_acc[q_idx][p]);
            }
            if(!pids_in_aps_order_for_q.empty() && parent_ && parent_->partition_manager_){
                std::vector<float*> cluster_centroids = parent_->partition_manager_->get_vectors(pids_in_aps_order_for_q);
                std::vector<float*> valid_centroids; // Centroids corresponding to pids_in_aps_order_for_q after filtering nulls
                std::vector<int64_t> pids_for_valid_centroids;

                for(size_t i=0; i<cluster_centroids.size(); ++i) {
                    if(cluster_centroids[i] != nullptr) {
                        valid_centroids.push_back(cluster_centroids[i]);
                        pids_for_valid_centroids.push_back(pids_in_aps_order_for_q[i]);
                    }
                }
                pids_in_aps_order_for_q = pids_for_valid_centroids; // Update to only contain PIDs with valid centroids

                if(!valid_centroids.empty()){
                    boundary_distances_for_q = compute_boundary_distances(x[q_idx], valid_centroids, metric_ == faiss::METRIC_L2);
                }
            }
        }

        int scanned_count_for_q = 0;
        for (int p = 0; p < num_partitions_for_this_query; ++p) { // Iterate through the original list of PIDs
            int64_t pid = p_ids_acc[q_idx][p];
            if (pid == -1) continue;

            const float *list_vectors = (float *)partition_manager_->partition_store_->get_codes(pid);
            const int64_t *list_ids_ptr = partition_manager_->partition_store_->get_ids(pid);
            int64_t list_size = partition_manager_->partition_store_->list_size(pid);

            if (list_size > 0 && list_vectors != nullptr) {
                scan_list(query_vec_ptr, list_vectors, list_ids_ptr, list_size, dimension, *topk_buf, metric_);
                scanned_count_for_q++;
            }

            if (use_aps && !boundary_distances_for_q.empty() && scanned_count_for_q > 0) {
                float radius = topk_buf->get_kth_distance(); // Calls flush
                // Recall profile uses boundary distances for PIDs that had valid centroids
                std::vector<float> probs = compute_recall_profile(boundary_distances_for_q, radius, dimension, {}, search_params->use_precomputed, metric_ == faiss::METRIC_L2);

                float estimated_recall = 0.0f;
                // Sum probs for partitions scanned so far that are in the APS ordered list
                int aps_relevant_scanned_count = 0;
                for(int scanned_p_idx = 0; scanned_p_idx <= p; ++scanned_p_idx) { // Check all PIDs up to current
                    int64_t current_scanned_pid = p_ids_acc[q_idx][scanned_p_idx];
                    if (current_scanned_pid == -1) continue;

                    auto it = std::find(pids_in_aps_order_for_q.begin(), pids_in_aps_order_for_q.end(), current_scanned_pid);
                    if (it != pids_in_aps_order_for_q.end()) {
                        int rank_in_aps_list = std::distance(pids_in_aps_order_for_q.begin(), it);
                        if (rank_in_aps_list < static_cast<int>(probs.size())) {
                            estimated_recall += probs[rank_in_aps_list];
                        }
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
    }, search_params->num_threads);

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
    std::atomic<long> total_partitions_scanned_across_queries{0}; // For timing_info

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

    parallel_for<int64_t>(0, p_to_q_vec.size(), [&](int64_t i) {
        int64_t current_pid = p_to_q_vec[i].first;
        const std::vector<int64_t>& query_indices_for_this_pid = p_to_q_vec[i].second;

        if (query_indices_for_this_pid.empty()) return;

        const float *list_codes_ptr = (float *)partition_manager_->partition_store_->get_codes(current_pid);
        const int64_t *list_ids_ptr = partition_manager_->partition_store_->get_ids(current_pid);
        int64_t list_size = partition_manager_->partition_store_->list_size(current_pid);
        int64_t current_d = partition_manager_->d();

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

        total_partitions_scanned_across_queries += query_indices_for_this_pid.size(); // Each query effectively "scans" this one partition

        for (size_t q_sub_idx = 0; q_sub_idx < query_indices_for_this_pid.size(); ++q_sub_idx) {
            int64_t global_q_idx = query_indices_for_this_pid[q_sub_idx];
            std::vector<float> dists = local_scan_buffers[q_sub_idx]->get_topk();
            std::vector<int64_t> ids = local_scan_buffers[q_sub_idx]->get_topk_indices();
            if (!ids.empty()) {
                // global_query_buffers access needs to be thread-safe if multiple parallel_for iterations
                // could target the same global_q_idx (not the case here, as parallel_for is over unique PIDs).
                // However, TopkBuffer itself has a mutex for batch_add.
                global_query_buffers[global_q_idx]->batch_add(dists.data(), ids.data(), ids.size());
            }
        }
    }, search_params->num_threads);

    auto topk_ids = torch::full({num_queries, k_val}, -1L, torch::kInt64);
    auto topk_dists = torch::full({num_queries, k_val},
                                  (metric_ == faiss::METRIC_INNER_PRODUCT) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity(),
                                  torch::kFloat32);
    auto ids_acc = topk_ids.accessor<int64_t,2>();
    auto dists_acc = topk_dists.accessor<float,2>();

    for(int64_t q_idx=0; q_idx < num_queries; ++q_idx) {
        global_query_buffers[q_idx]->flush();
        std::vector<float> final_dists = global_query_buffers[q_idx]->get_topk();
        std::vector<int64_t> final_ids = global_query_buffers[q_idx]->get_topk_indices();
        int num_results = std::min((int)final_ids.size(), k_val);
        for(int i=0; i<num_results; ++i) {
            ids_acc[q_idx][i] = final_ids[i];
            dists_acc[q_idx][i] = final_dists[i];
        }
    }

    auto timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->search_params = search_params;
    // This is an approximation: total PIDs processed / num_queries.
    // A more accurate per-query scanned count would require summing unique PIDs scanned per query.
    if (num_queries > 0) {
        timing_info->partitions_scanned = total_partitions_scanned_across_queries.load() / num_queries;
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