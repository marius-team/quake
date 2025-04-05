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
#include <arrow/compute/api_vector.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <torch/torch.h>

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
      workers_initialized_(false) {

    if (num_workers_ > 0) {
        initialize_workers(num_workers_);
    }
}

// Destructor
QueryCoordinator::~QueryCoordinator() {
    shutdown_workers();
}

void QueryCoordinator::allocate_core_resources(int core_idx, int num_queries, int k, int d) {
    CoreResources &res = core_resources_[core_idx];
    res.core_id = core_idx;
    res.local_query_buffer.resize(num_queries * d * sizeof(float));
    res.topk_buffer_pool.resize(num_queries);
    for (int q = 0; q < num_queries; ++q) {
        res.topk_buffer_pool[q] = make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
        res.job_queue = moodycamel::BlockingConcurrentQueue<ScanJob>();
    }

}

// Initialize Worker Threads
void QueryCoordinator::initialize_workers(int num_cores) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_cores << " worker threads." <<
            std::endl;

    partition_manager_->distribute_partitions(num_cores);

    core_resources_.resize(num_cores);
    worker_threads_.resize(num_cores);
    worker_job_counter_.reserve(num_cores);
    for (int i = 0; i < num_cores; i++) {
        if (!set_thread_affinity(i)) {
            std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
        }
        allocate_core_resources(i, 1, 10, partition_manager_->d());
        worker_threads_[i] = std::thread(&QueryCoordinator::partition_scan_worker_fn, this, i);
        worker_job_counter_[i] = 0;
    }
    workers_initialized_ = true;
}

// Shutdown Worker Threads
void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    stop_workers_.store(true);
    // Enqueue a special shutdown job for each core.
    for (auto &res : core_resources_) {
        ScanJob termination_job;
        termination_job.partition_id = -1;
        res.job_queue.enqueue(termination_job);
    }
    // Join all worker threads.
    for (auto &thr : worker_threads_) {
        if (thr.joinable())
            thr.join();
    }
    worker_threads_.clear();
    workers_initialized_ = false;
}

// Worker Thread Function
void QueryCoordinator::partition_scan_worker_fn(int core_index) {

    CoreResources &res = core_resources_[core_index];

    if (!set_thread_affinity(core_index)) {
        std::cout << "[QueryCoordinator::partition_scan_worker_fn] Failed to set thread affinity on core " << core_index << std::endl;
    }



    while (true) {
        ScanJob job;

        auto job_wait_start = std::chrono::high_resolution_clock::now();
        res.job_queue.wait_dequeue(job);
        auto job_wait_end = std::chrono::high_resolution_clock::now();

        job_pull_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(job_wait_end - job_wait_start).count();

        auto job_process_start = std::chrono::high_resolution_clock::now();
        shared_ptr<TopkBuffer> local_topk_buffer = res.topk_buffer_pool[0];

        // Shutdown signal: -1 indicates the worker should exit.
        if (job.partition_id == -1) {
            break;
        }

        // Ignore this job if the global buffer is not processing queries.
        if (!global_topk_buffer_pool_[job.query_ids[0]]->currently_processing_query()) {
            // decrement the job counter
            global_topk_buffer_pool_[job.query_ids[0]]->record_empty_job();
            continue;
        }

        worker_job_counter_[core_index]++;

        // Retrieve partition data.
        const float *partition_codes = (float *) partition_manager_->partition_store_->get_codes(job.partition_id);
        const int64_t *partition_ids = (int64_t *) partition_manager_->partition_store_->get_ids(job.partition_id);
        int64_t partition_size = partition_manager_->partition_store_->list_size(job.partition_id);

        // Branch for non-batched jobs.
        if (!job.is_batched) {

            // Allocate a thread-local query buffer if needed.
            if (res.local_query_buffer.size() < partition_manager_->d() * sizeof(float)) {
                res.local_query_buffer.resize(partition_manager_->d() * sizeof(float));
            }

            // Copy the contents of the query vector to the local buffer using memcpy.
            if (memcpy(res.local_query_buffer.data(), job.query_vector, partition_manager_->d() * sizeof(float)) == nullptr) {
                throw std::runtime_error("[partition_scan_worker_fn] memcpy failed.");
            }

            if (local_topk_buffer == nullptr) {
                throw std::runtime_error("[partition_scan_worker_fn] local_topk_buffer is null.");
            } else {
                local_topk_buffer->set_k(job.k);
                local_topk_buffer->reset();
            }
            // Perform the scan on the partition.
            scan_list((float *) res.local_query_buffer.data(),
                partition_codes,
                partition_ids,
                      partition_size,
                      partition_manager_->d(),
                      *local_topk_buffer,
                      metric_);

            vector<float> topk = local_topk_buffer->get_topk();
            vector<int64_t> topk_indices = local_topk_buffer->get_topk_indices();
            int64_t n_results = topk_indices.size();

            // Merge local results into the global query buffer.
            global_topk_buffer_pool_[job.query_ids[0]]->batch_add(topk.data(), topk_indices.data(), n_results);
            job_flags_[job.query_ids[0]][job.rank] = true;
        }
        // Batched job branch.
        else {
            if (!job.query_vector || job.num_queries == 0) {
                throw std::runtime_error("[partition_scan_worker_fn] Invalid batched job.");
            }

            // Allocate a thread-local query buffer if needed.
            if (res.local_query_buffer.size() < partition_manager_->d() * sizeof(float) * job.num_queries) {
                res.local_query_buffer.resize(partition_manager_->d() * sizeof(float) * job.num_queries);
            }

            int64_t d = partition_manager_->d();
            std::vector<float> query_subset(job.num_queries * d);
            for (int i = 0; i < job.num_queries; i++) {
                int64_t global_q = job.query_ids[i];
                memcpy(&query_subset[i * d],
                       job.query_vector + global_q * d,
                       d * sizeof(float));
            }
            // Then copy query_subset into your local buffer if needed:
            if(memcpy(res.local_query_buffer.data(),
                   query_subset.data(),
                   query_subset.size() * sizeof(float)) == nullptr) {
                throw std::runtime_error("[partition_scan_worker_fn] memcpy failed.");
            }

            // Use a thread_local vector to hold one buffer per query.
            if (res.topk_buffer_pool.size() < static_cast<size_t>(job.num_queries)) {
                res.topk_buffer_pool.resize(job.num_queries);
                for (int64_t q = 0; q < job.num_queries; ++q) {
                    res.topk_buffer_pool[q] = std::make_shared<TopkBuffer>(job.k, metric_ == faiss::METRIC_INNER_PRODUCT);
                }
            } else {
                for (int64_t q = 0; q < job.num_queries; ++q) {
                    res.topk_buffer_pool[q]->set_k(job.k);
                    res.topk_buffer_pool[q]->reset();
                }
            }

            // Process the batched job.
            batched_scan_list((float *) res.local_query_buffer.data(),
                partition_codes,
                partition_ids,
                              job.num_queries,
                              partition_size,
                              partition_manager_->d(),
                              res.topk_buffer_pool,
                              metric_);

            vector<vector<float>> topk_list(job.num_queries);
            vector<vector<int64_t>> topk_indices_list(job.num_queries);
            for (int64_t q = 0; q < job.num_queries; q++) {
                topk_list[q] = res.topk_buffer_pool[q]->get_topk();
                topk_indices_list[q] = res.topk_buffer_pool[q]->get_topk_indices();
            }

            for (int64_t q = 0; q < job.num_queries; q++) {
                int64_t global_q = job.query_ids[q];
                int n_results = topk_indices_list[q].size();
                global_topk_buffer_pool_[global_q]->batch_add(topk_list[q].data(), topk_indices_list[q].data(), n_results);
            }
        }
        auto job_process_end = std::chrono::high_resolution_clock::now();
        job_process_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(job_process_end - job_process_start).count();
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

    if (!x.defined() || x.size(0) == 0) {
        auto empty_result = std::make_shared<SearchResult>();
        empty_result->ids = torch::empty({0}, torch::kInt64);
        empty_result->distances = torch::empty({0}, torch::kFloat32);
        empty_result->timing_info = std::make_shared<SearchTimingInfo>();
        return empty_result;
    }

    int64_t num_queries = x.size(0);
    int64_t dimension = x.size(1);
    int k = search_params->k;
    int64_t nlist = partition_manager_->nlist();
    bool use_aps = (search_params->recall_target > 0.0 && !search_params->batched_scan);
    auto timing_info = make_shared<SearchTimingInfo>();
    timing_info->n_queries = num_queries;
    timing_info->n_clusters = nlist;
    timing_info->search_params = search_params;

    float *x_ptr = x.data_ptr<float>();

    auto start_time = high_resolution_clock::now();

    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
    }

    job_flags_.clear();
    job_flags_.resize(num_queries);
    for (int64_t q = 0; q < num_queries; q++) {
        job_flags_[q] = vector<std::atomic<bool>>(partition_ids.size(1));
        for (int64_t p = 0; p < partition_ids.size(1); p++) {
            job_flags_[q][p] = false;
        }
    }

    job_pull_time_ns = 0;
    job_process_time_ns = 0;

    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (global_topk_buffer_pool_.size() < static_cast<size_t>(num_queries)) {
            std::cout << "Resizing query_topk_buffers_ from " << global_topk_buffer_pool_.size()
                      << " to " << num_queries << std::endl;
            int old_size = global_topk_buffer_pool_.size();
            global_topk_buffer_pool_.resize(num_queries);
            for (int64_t q = old_size; q < num_queries; q++) {
                global_topk_buffer_pool_[q] = std::make_shared<TopkBuffer>(k, metric_ == faiss::METRIC_INNER_PRODUCT);
            }
        } else {
            for (int64_t q = 0; q < num_queries; q++) {
                global_topk_buffer_pool_[q]->set_k(k);
                global_topk_buffer_pool_[q]->reset();
                global_topk_buffer_pool_[q]->set_processing_query(true);
            }
        }
        for (int64_t q = 0; q < num_queries; q++) {
            global_topk_buffer_pool_[q]->set_jobs_left(partition_ids.size(1));
        }
    }
    auto end_time = high_resolution_clock::now();
    timing_info->buffer_init_time_ns =
        duration_cast<nanoseconds>(end_time - start_time).count();

    start_time = high_resolution_clock::now();
    if (search_params->batched_scan) {
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();

        std::unordered_map<int64_t, vector<int64_t>> per_partition_query_ids;
        for (int64_t q = 0; q < num_queries; q++) {
            for (int64_t p = 0; p < partition_ids.size(1); p++) {
                int64_t pid = partition_ids_accessor[q][p];
                if (pid < 0) continue;
                per_partition_query_ids[pid].push_back(q);
            }
        }
        for (auto &kv : per_partition_query_ids) {
            ScanJob job;
            job.is_batched = true;
            job.partition_id = kv.first;
            job.k = k;
            job.query_vector = x.data_ptr<float>();
            job.num_queries = kv.second.size();
            job.query_ids = kv.second;
            int core_id = partition_manager_->get_partition_core_id(kv.first);
            core_resources_[core_id].job_queue.enqueue(job);
        }
    } else {
        auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();

        int64_t start = 0;
        int64_t end = num_queries;
        parallel_for(start, end, [&](int64_t q) {
            for (int64_t p = 0; p < partition_ids.size(1); p++) {
                int64_t pid = partition_ids_accessor[q][p];
                if (pid == -1) continue;

                ScanJob job;
                job.is_batched = false;
                job.query_ids = {q};
                job.partition_id = pid;
                job.k = k;
                job.query_vector = x_ptr + q * dimension;
                job.num_queries = 1;
                job.rank = p;

                int core_id = partition_manager_->get_partition_core_id(pid);
                core_resources_[core_id].job_queue.enqueue(job);
            }
            }, search_params->num_threads);
    }
    end_time = high_resolution_clock::now();
    timing_info->job_enqueue_time_ns = duration_cast<nanoseconds>(end_time - start_time).count();

    auto last_flush_time = high_resolution_clock::now();
    vector<vector<float>> boundary_distances(num_queries);
    if (use_aps) {
        for (int64_t q = 0; q < num_queries; q++) {
            vector<int64_t> partition_ids_to_scan_vec = vector<int64_t>(partition_ids[q].data_ptr<int64_t>(),
                                                                partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));
            vector<float *> cluster_centroids = parent_->partition_manager_->get_vectors(partition_ids_to_scan_vec);
            boundary_distances[q] = compute_boundary_distances(x[q],
                                                                cluster_centroids,
                                                                metric_ == faiss::METRIC_L2);
        }
    }

    start_time = high_resolution_clock::now();
    last_flush_time = high_resolution_clock::now();

    vector<float> query_radius(num_queries, 0.0f);
    vector<vector<float>> probs(num_queries);

    while (true) {
        // check if all jobs have been processed
        bool all_done = true;
        for (int64_t q = 0; q < num_queries; q++) {
            all_done = all_done && (global_topk_buffer_pool_[q]->jobs_left_ == 0);
        }

        if (all_done) {
            break;
        }

        // check if recall target has been reached
        if (use_aps && duration_cast<microseconds>(high_resolution_clock::now() - last_flush_time).count()
             > search_params->aps_flush_period_us) {
            for (int64_t q = 0; q < num_queries; q++) {
                auto curr_buffer = global_topk_buffer_pool_[q];
                int scanned = curr_buffer->get_num_partitions_scanned();
                if (curr_buffer->currently_processing_query() &&
                    scanned > 0 && scanned < (int) boundary_distances[q].size()) {
                    float radius = curr_buffer->get_kth_distance();

                    if (query_radius[q] != radius) {
                        query_radius[q] = radius;

                        // recompute recall profile if the radius has changed
                        probs[q] = compute_recall_profile(boundary_distances[q],
                                                                    radius,
                                                                    dimension,
                                                                    {},
                                                                    search_params->use_precomputed,
                                                                    metric_ == faiss::METRIC_L2);
                    }

                    float cum = 0.0f;
                    for (int i = 0; i < partition_ids.size(1); i++) {
                        if (job_flags_[q][i]) {
                            cum += probs[q][i];
                        }
                    }
                    if (cum > search_params->recall_target) {
                        curr_buffer->set_processing_query(false);
                    }
                }
            }
            last_flush_time = high_resolution_clock::now();
        }
        std::this_thread::sleep_for(microseconds(1));
    }
    end_time = high_resolution_clock::now();
    timing_info->job_wait_time_ns =
        duration_cast<nanoseconds>(end_time - start_time).count();

    // Aggregate results.
    start_time = high_resolution_clock::now();
    auto topk_ids = torch::full({num_queries, k}, -1, torch::kInt64);
    auto topk_dists = torch::full({num_queries, k},
                                  std::numeric_limits<float>::infinity(), torch::kFloat32);
    auto ids_accessor = topk_ids.accessor<int64_t, 2>();
    auto dists_accessor = topk_dists.accessor<float, 2>();
    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        for (int64_t q = 0; q < num_queries; q++) {
            auto topk = global_topk_buffer_pool_[q]->get_topk();
            auto ids = global_topk_buffer_pool_[q]->get_topk_indices();
            for (int i = 0; i < k; i++) {
                if (i < (int) ids.size()) {
                    ids_accessor[q][i] = ids[i];
                    dists_accessor[q][i] = topk[i];
                } else {
                    ids_accessor[q][i] = -1;
                    dists_accessor[q][i] = (metric_ == faiss::METRIC_INNER_PRODUCT)
                                             ? -std::numeric_limits<float>::infinity()
                                             : std::numeric_limits<float>::infinity();
                }
            }
            timing_info->partitions_scanned = global_topk_buffer_pool_[q]->get_num_partitions_scanned();
        }
    }
    end_time = high_resolution_clock::now();
    timing_info->result_aggregate_time_ns = duration_cast<nanoseconds>(end_time - start_time).count();
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;

    return search_result;
}

bool* create_bitmap(std::unordered_map<int64_t, int64_t> id_to_price, int64_t* list_ids, 
                                        int64_t num_ids, shared_ptr<SearchParams> search_params) {
    
    bool* bitmap = new bool[num_ids];

    for (int64_t i = 0; i < num_ids; i++) {
        int64_t id = list_ids[i];
        if (id_to_price.count(id) && id_to_price[id] <= search_params->price_threshold) {
            bitmap[i] = 1;
        } else {
            bitmap[i] = 0;
        }
    }

    return bitmap;
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

    auto start_time = high_resolution_clock::now();

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
    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({num_queries, partition_ids.size(0)});
    }
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 2>();
    float *x_ptr = x.data_ptr<float>();

    // Allocate per-query result vectors.
    vector<vector<float>> all_topk_dists(num_queries);
    vector<vector<int64_t>> all_topk_ids(num_queries);

    // Use our custom parallel_for to process queries in parallel.
    parallel_for<int64_t>(0, num_queries, [&](int64_t q) {
        // Create a local TopK buffer for query q.
        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending);
        const float* query_vec = x_ptr + q * dimension;
        int num_parts = partition_ids.size(1);

        vector<float> boundary_distances;
        vector<float> partition_probs;
        float query_radius = 1000000.0;
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            query_radius = -1000000.0;
        }

        Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids[q]);
        if (use_aps) {
            vector<int64_t> partition_ids_to_scan_vec = std::vector<int64_t>(partition_ids[q].data_ptr<int64_t>(),
                                                                partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));
            vector<float *> cluster_centroids = parent_->partition_manager_->get_vectors(partition_ids_to_scan_vec);
            boundary_distances = compute_boundary_distances(x[q],
                                                            cluster_centroids,
                                                            metric_ == faiss::METRIC_L2);
        }
        for (int p = 0; p < num_parts; p++) {
            int64_t pi = partition_ids_accessor[q][p];

            if (pi == -1) {
                continue; // Skip invalid partitions
            }

            start_time = std::chrono::high_resolution_clock::now();
            float *list_vectors = (float *) partition_manager_->partitions_->get_codes(pi);
            int64_t *list_ids = (int64_t *) partition_manager_->partitions_->get_ids(pi);
            std::shared_ptr<arrow::Table> partition_attributes_table = 
                            partition_manager_->partitions_->partitions_[pi]->attributes_table_;
            int64_t list_size = partition_manager_->partitions_->list_size(pi);

            std::shared_ptr<arrow::Int64Array> id_array = nullptr;
            std::shared_ptr<arrow::DoubleArray> price_array = nullptr;

            std::unordered_map<int64_t, int64_t> id_to_price;

            if (partition_attributes_table != nullptr) {
                id_array = std::static_pointer_cast<arrow::Int64Array>(partition_attributes_table->GetColumnByName("id")->chunk(0));
                price_array = std::static_pointer_cast<arrow::DoubleArray>(partition_attributes_table->GetColumnByName("price")->chunk(0));
                int64_t length = id_array->length();
                for (int64_t i = 0; i < id_array->length(); i++) {
                    id_to_price[id_array->Value(i)] = price_array->Value(i);
                }
            }
            
            bool* bitmap = nullptr;            

            if (search_params->filteringType == FilteringType::PRE_FILTERING) {
                bitmap = create_bitmap(id_to_price, list_ids, list_size, search_params);
            }

            scan_list(query_vec,
                      list_vectors,
                      list_ids,
                      partition_manager_->partition_store_->list_size(pi),
                      dimension,
                      *topk_buf,
                      metric_,
                      bitmap);
            if (search_params->filteringType ==FilteringType::POST_FILTERING) {
                auto scanned_vectors = topk_buf->topk_;
                int buffer_size = topk_buf->curr_offset_;
                for (int i = 0;i < buffer_size; i++) {
                    auto vector_id = scanned_vectors[i].second;
                    if (id_to_price.count(vector_id) and id_to_price[vector_id] > search_params->price_threshold) {
                        topk_buf->remove(i);
                    }
                }
            }

            float curr_radius = topk_buf->get_kth_distance();
            float percent_change = abs(curr_radius - query_radius) / curr_radius;

            start_time = high_resolution_clock::now();
            if (use_aps) {
                if (percent_change > search_params->recompute_threshold) {
                    query_radius = curr_radius;

                    partition_probs = compute_recall_profile(boundary_distances,
                                                             query_radius,
                                                             dimension,
                                                             {},
                                                             search_params->use_precomputed,
                                                             metric_ == faiss::METRIC_L2);
                }
                float recall_estimate = 0.0;
                for (int i = 0; i < p; i++) {
                    recall_estimate += partition_probs[i];
                }
                if (recall_estimate >= search_params->recall_target) {
                    break;
                }
            }
        }
        // Retrieve the top-k results for query q.
        all_topk_dists[q] = topk_buf->get_topk();
        all_topk_ids[q] = topk_buf->get_topk_indices();
    }, search_params->num_threads);

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

    auto end_time = high_resolution_clock::now();
    timing_info->total_time_ns = duration_cast<nanoseconds>(end_time - start_time).count();

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
    auto start = high_resolution_clock::now();

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
            parent_search_params->k = std::min(search_params->nprobe, (int) partition_manager_->nlist());
        }

        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        parent_timing_info = parent_search_result->timing_info;
    }

    auto search_result = scan_partitions(x, partition_ids_to_scan, search_params);

    // if (search_params->filteringType == FilteringType::POST_FILTERING and search_params->price_threshold != INT_MAX and parent_!=nullptr) {
    //     auto id_data = search_result->ids.data<int64_t>();
    //     auto distance_data = search_result->distances.data<float>();
    //     int64_t num_results = search_result->ids.size(0);

    //     std::vector<int64_t> filtered_ids;
    //     std::vector<float> filtered_distances;

    //     std::shared_ptr<arrow::ChunkedArray> id_column = attributes_table->GetColumnByName("id");
    //     std::shared_ptr<arrow::ChunkedArray> price_column = attributes_table->GetColumnByName("price");

    //     for (int64_t i = 0; i < num_results; i++) {
    //         int64_t id = id_data[i];

    //         // Search for this ID in the attribute table
    //         std::shared_ptr<arrow::Datum> found_row;
    //         auto id_scalar = arrow::MakeScalar(id);
    //         auto price_scalar = arrow::MakeScalar(search_params->price_threshold);

    //         auto equal_condition  = arrow::compute::CallFunction("equal", {id_column->chunk(0), id_scalar});
    //         auto less_equal_condition = arrow::compute::CallFunction("less_equal", {price_column->chunk(0), price_scalar});

    //         auto combined_condition = arrow::compute::CallFunction("and", {equal_condition.ValueOrDie(), less_equal_condition.ValueOrDie()});

    //         auto mask_table = std::static_pointer_cast<arrow::BooleanArray>(combined_condition->make_array());

    //         auto filter_result = arrow::compute::Filter(attributes_table, combined_condition.ValueOrDie());

    //         if (filter_result.ok()) {
    //             filtered_ids.push_back(id);
    //             filtered_distances.push_back(distance_data[i]);
    //         }
    //     }

    //     search_result->ids = torch::tensor(filtered_ids, torch::kInt64);
    //     search_result->distances = torch::tensor(filtered_distances, torch::kFloat);
    // }
    
    search_result->timing_info->parent_info = parent_timing_info;

    auto end = high_resolution_clock::now();
    search_result->timing_info->total_time_ns = duration_cast<nanoseconds>(end - start).
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
    auto start = high_resolution_clock::now();

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

    std::vector<std::pair<int64_t, std::vector<int64_t>>> queries_vec;
    queries_vec.reserve(queries_by_partition.size());
    for (const auto &entry : queries_by_partition) {
        queries_vec.push_back(entry);
    }

    parallel_for((int64_t) 0, (int64_t) queries_by_partition.size(), [&](int64_t i) {
        int64_t pid = queries_vec[i].first;
        auto query_indices = queries_vec[i].second;

        // Create a tensor for the indices and then a subset of the queries.
        Tensor indices_tensor = torch::tensor(query_indices, torch::kInt64);
        Tensor x_subset = x.index_select(0, indices_tensor);
        int64_t batch_size = x_subset.size(0);

        // Get the partition’s data.
        const float *list_codes = (float *) partition_manager_->partition_store_->get_codes(pid);
        const int64_t *list_ids = partition_manager_->partition_store_->get_ids(pid);
        int64_t list_size = partition_manager_->partition_store_->list_size(pid);
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


    }, search_params->num_threads);

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

    auto end = high_resolution_clock::now();
    timing_info->total_time_ns = duration_cast<nanoseconds>(end - start).count();

    // Prepare and return the final search result.
    auto search_result = std::make_shared<SearchResult>();
    search_result->ids = topk_ids;
    search_result->distances = topk_dists;
    search_result->timing_info = timing_info;
    return search_result;
}
