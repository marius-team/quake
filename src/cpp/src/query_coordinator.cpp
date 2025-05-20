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

void QueryCoordinator::allocate_core_resources(int core_idx,
                                               int num_queries,
                                               int k,
                                               int d)
{
    auto& CR = core_resources_[core_idx];
    CR.core_id = core_idx;
    CR.topk_buffer_pool.clear();

    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(core_idx);
#endif

    // job queue remains default-constructed
    numa_resources_.resize(get_num_numa_nodes());
    auto& numa_res = numa_resources_[numa_node];
    size_t bytes = size_t(num_queries) * d * sizeof(float);
    if (numa_res.buffer_size != bytes) {
        quake_free(numa_res.local_query_buffer, numa_res.buffer_size);
        numa_res.local_query_buffer = static_cast<float*>(quake_alloc(bytes, numa_node));
        numa_res.buffer_size = bytes;
    }
}

void QueryCoordinator::partition_scan_worker_fn(int core_index) {
    CoreResources &res = core_resources_[core_index];
    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(core_index);
#endif
    NUMAResources &nr = numa_resources_[numa_node];

    set_thread_affinity(core_index);

    while (!stop_workers_) {
        int64_t jid = 0;
        if (!res.job_queue.try_dequeue(jid)) {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
            continue;
        }

        process_scan_job(job_buffer_[jid], res);
    }
}

void QueryCoordinator::process_scan_job(ScanJob job,
                                        CoreResources &res) {
    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(res.core_id);
#endif
    NUMAResources &nr = numa_resources_[numa_node];

    // fetch codes & ids
    const float   *codes = (float*)partition_manager_->partition_store_->get_codes(job.partition_id);
    const int64_t *ids   = (int64_t*)partition_manager_->partition_store_->get_ids(job.partition_id);

    int64_t part_size = 0;
    try {
        part_size = partition_manager_->partition_store_->list_size(job.partition_id);
    } catch (...) {
        std::cerr << "[partition_scan_worker_fn] Partition " << job.partition_id << " not found.\n";
        result_queue_.enqueue(ResultJob{job.query_id, job.rank, {}, {}});
        return;
    }

    if (part_size == 0) {
        // empty => enqueue zero‐work per query
        if (job.is_batched) {
            for (int64_t q : *job.query_ids) {
                result_queue_.enqueue(ResultJob{(int)q, 0, {}, {}});
            }
        } else {
            result_queue_.enqueue(ResultJob{job.query_id, job.rank, {}, {}});
        }
        return;
    }

    if (!job.is_batched) {
        handle_nonbatched_job(job, res, nr);
    } else {
        handle_batched_job(job, res, nr);
    }
}

void QueryCoordinator::handle_nonbatched_job(const ScanJob &job,
                                             CoreResources &res,
                                             NUMAResources &nr) {

    // ensure buffers
    if (res.topk_buffer_pool.size() < 1) {
        res.topk_buffer_pool.resize(1);
        res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                job.k,
                metric_ == faiss::METRIC_INNER_PRODUCT,
                /*cap=*/100 * job.k,
                /*node=*/cpu_numa_node(res.core_id)
        );
    } else if (res.topk_buffer_pool[0]->k_ != job.k) {
        // check capacity
        if (res.topk_buffer_pool[0]->capacity_ < job.k) {
            res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                    job.k,
                    metric_ == faiss::METRIC_INNER_PRODUCT,
                    /*cap=*/100 * job.k,
                    /*node=*/cpu_numa_node(res.core_id)
            );
        }
        res.topk_buffer_pool[0]->set_k(job.k);
    }

    auto buf = res.topk_buffer_pool[0];
    res.topk_buffer_pool[0]->reset();

    try {
        // It's good practice to re-fetch or validate partition data here if it could
        // have changed since process_scan_job's initial checks, especially with maintenance.
        const float* codes = (float*)partition_manager_->partition_store_->get_codes(job.partition_id);
        const int64_t* ids = (int64_t*)partition_manager_->partition_store_->get_ids(job.partition_id);
        int64_t part_size = partition_manager_->partition_store_->list_size(job.partition_id);
        int D = partition_manager_->d();

        // Defensive check for partition validity right before scan
        if (!codes || !ids || part_size <= 0) {
            std::cerr << "[QueryCoordinator::handle_nonbatched_job] Partition " << job.partition_id
                      << " invalid or empty before scan for query " << job.query_id
                      << ". Enqueuing empty result.\n";
            result_queue_.enqueue(ResultJob{job.query_id, job.rank, {}, {}});
            return; // Important to return after enqueueing the placeholder
        }

        scan_list(nr.local_query_buffer, // Or job.query_vector depending on NUMA strategy
                  codes,
                  ids,
                  part_size,
                  D,
                  *buf,
                  metric_);

        // If scan_list completes, enqueue its results
        auto tv = buf->get_topk();
        auto ti = buf->get_topk_indices();
        result_queue_.enqueue(ResultJob{job.query_id, job.rank, std::move(tv), std::move(ti)});

    } catch (const std::exception& e) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id << ": " << e.what()
                  << ". Enqueuing empty result.\n";
        result_queue_.enqueue(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    } catch (...) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Unknown exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id
                  << ". Enqueuing empty result.\n";
        result_queue_.enqueue(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    }
}

void QueryCoordinator::handle_batched_job(const ScanJob &job,
                                          CoreResources &res,
                                          NUMAResources &nr) {
    // ensure buffers
    if (res.topk_buffer_pool.size() < (size_t)job.num_queries) {
        res.topk_buffer_pool.resize(job.num_queries);
        for (int64_t i = 0; i < job.num_queries; ++i) {
            res.topk_buffer_pool[i] = std::make_shared<TopkBuffer>(
                    job.k,
                    metric_ == faiss::METRIC_INNER_PRODUCT,
                    /*cap=*/num_workers_ * job.k,
                    /*node=*/cpu_numa_node(res.core_id)
            );
        }

        // realloc batch query buffer
        if (res.batch_queries != nullptr) {
            quake_free(res.batch_queries,
                       size_t(job.num_queries) * partition_manager_->d() * sizeof(float));
        }
        res.batch_queries = static_cast<float*>(quake_alloc(
                size_t(job.num_queries) * partition_manager_->d() * sizeof(float),
                cpu_numa_node(res.core_id)));

        // realloc batch distances and ids
        if (res.batch_distances != nullptr) {
            quake_free(res.batch_distances,
                       size_t(job.num_queries) * job.k * sizeof(float));
        }
        if (res.batch_ids != nullptr) {
            quake_free(res.batch_ids,
                       size_t(job.num_queries) * job.k * sizeof(int64_t));
        }
        res.batch_distances = static_cast<float*>(quake_alloc(
                size_t(job.num_queries) * job.k * sizeof(float),
                cpu_numa_node(res.core_id)));
        res.batch_ids = static_cast<int64_t*>(quake_alloc(
                size_t(job.num_queries) * job.k * sizeof(int64_t),
                cpu_numa_node(res.core_id)));
    } else {
        for (int64_t i = 0; i < job.num_queries; ++i) {
            res.topk_buffer_pool[i]->set_k(job.k);
            res.topk_buffer_pool[i]->reset();
        }

        // set batch distances and ids to max/min and -1
        for (int64_t i = 0; i < job.num_queries * job.k; ++i) {
            res.batch_distances[i] = metric_ == faiss::METRIC_INNER_PRODUCT ?
                                     -std::numeric_limits<float>::infinity() :
                                     std::numeric_limits<float>::infinity();
            res.batch_ids[i] = -1;
        }
    }

    bool scan_successful = false;
    try {
        const float* codes = (float*)partition_manager_->partition_store_->get_codes(job.partition_id);
        const int64_t* ids = (int64_t*)partition_manager_->partition_store_->get_ids(job.partition_id);
        int64_t part_size = partition_manager_->partition_store_->list_size(job.partition_id);
        int D = partition_manager_->d();

        if (!codes || !ids || part_size <= 0) {
            std::cerr << "[QueryCoordinator::handle_batched_job] Partition " << job.partition_id
                      << " invalid or empty before batched scan. Enqueuing empty results for batch.\n";
            // Let success remain false, loop below will send empty results
            throw std::runtime_error("Batched scan pre-check failed for partition data");
        }

        // Gather queries into res.batch_queries (as in your original code)
        for (int64_t i = 0; i < job.num_queries; ++i) {
            std::memcpy(res.batch_queries + i * D,
                        job.query_vector + job.query_ids->at(i) * D,
                        D * sizeof(float));
        }

        batched_scan_list(res.batch_queries,
                          codes,
                          ids,
                          job.num_queries,
                          part_size,
                          D,
                          res.topk_buffer_pool, // Vector of TopkBuffer shared_ptrs
                          metric_,
                          res.batch_distances,  // Scratch space for distances
                          res.batch_ids);       // Scratch space for IDs
        scan_successful = true;

    } catch (const std::exception& e) {
        std::cerr << "[QueryCoordinator::handle_batched_job] Exception during batched scan for partition "
                  << job.partition_id << ": " << e.what()
                  << ". Will enqueue empty results for affected queries.\n";
        // scan_successful remains false
    } catch (...) {
        std::cerr << "[QueryCoordinator::handle_batched_job] Unknown exception during batched scan for partition "
                  << job.partition_id
                  << ". Will enqueue empty results for affected queries.\n";
        // scan_successful remains false
    }

    // Enqueue results (actual or empty) for every query in this batch job
    std::vector<ResultJob> results_batch;
    results_batch.reserve(job.num_queries);
    for (int64_t i = 0; i < job.num_queries; ++i) {
        int global_query_id = job.query_ids->at(i);
        int rank_for_this_query = job.ranks->at(i); // Rank of this partition scan for this specific query

        if (scan_successful) {
            // Ensure buffer pool access is safe if it could have been resized or is smaller than expected
            if (i < res.topk_buffer_pool.size() && res.topk_buffer_pool[i]) {
                auto tv = res.topk_buffer_pool[i]->get_topk();
                auto ti = res.topk_buffer_pool[i]->get_topk_indices();
                results_batch.emplace_back(ResultJob{global_query_id, rank_for_this_query, std::move(tv), std::move(ti)});
            } else {
                std::cerr << "[QueryCoordinator::handle_batched_job] Error accessing TopK buffer for successful scan, query "
                          << global_query_id << ", buffer index " << i << ". Sending empty result.\n";
                results_batch.emplace_back(ResultJob{global_query_id, rank_for_this_query, {}, {}});
            }
        } else {
            // Scan failed, enqueue an empty result for this query's part of the batch
            results_batch.emplace_back(ResultJob{global_query_id, rank_for_this_query, {}, {}});
        }
    }
    if (!results_batch.empty()) {
        result_queue_.enqueue_bulk(std::make_move_iterator(results_batch.begin()), results_batch.size());
    }
}

void QueryCoordinator::init_global_buffers(int64_t nQ,
                                           int K,
                                           Tensor &partition_ids,
                                           shared_ptr<SearchParams> params) {
    std::lock_guard<std::mutex> lg(global_mutex_);
    // resize or reset
    if (global_topk_buffer_pool_.size() < (size_t) nQ) {
        size_t old = global_topk_buffer_pool_.size();
        global_topk_buffer_pool_.resize(nQ);
        for (int64_t q = old; q < nQ; ++q) {
            global_topk_buffer_pool_[q] = std::make_shared<TopkBuffer>(
                    K,
                    metric_ == faiss::METRIC_INNER_PRODUCT,
                    /*cap=*/num_workers_ * K * 2,
                    /*node=*/0
            );
        }
    } else {
        for (int64_t q = 0; q < nQ; ++q) {
            global_topk_buffer_pool_[q]->set_k(K);
            global_topk_buffer_pool_[q]->reset();
        }
    }
}

void QueryCoordinator::copy_query_to_numa(const float *xptr, int64_t nQ, int64_t D) {

    for (int node = 0; node < get_num_numa_nodes(); ++node) {
        auto &nr = numa_resources_[node];
        if (nr.buffer_size < size_t(nQ) * size_t(D) * sizeof(float)) {
            quake_free(nr.local_query_buffer, nr.buffer_size);
            nr.local_query_buffer = static_cast<float*>(quake_alloc(
                    size_t(nQ) * size_t(D) * sizeof(float),
                    node));
            nr.buffer_size = size_t(nQ) * size_t(D) * sizeof(float);
        }
        std::memcpy(nr.local_query_buffer,
                    xptr,
                    size_t(nQ) * size_t(D) * sizeof(float));
    }
}

void QueryCoordinator::enqueue_scan_jobs(Tensor x,
                                         Tensor partition_ids,
                                         shared_ptr<SearchParams> params)
{
    int64_t nQ = x.size(0), D = x.size(1);
    float *xptr = x.data_ptr<float>();

    auto partition_ids_acc = partition_ids.accessor<int64_t,2>();

    // flatten jobs
    next_job_id_ = 0;
    job_flags_.clear();
    job_flags_.resize(nQ);
    for (int64_t q = 0; q < nQ; ++q) {
        job_flags_[q] = vector<std::atomic<bool>>(partition_ids.size(1));
        for (int p = 0; p < partition_ids.size(1); ++p) {
            job_flags_[q][p].store(false);
            if (partition_ids_acc[q][p] < 0) job_flags_[q][p] = true;
        }
    }
    job_buffer_.clear();

    auto pid_acc = partition_ids.accessor<int64_t,2>();
    if (!params->batched_scan) {
        // one job per (q,p)
        for (int64_t q = 0; q < nQ; ++q) {
            const float* qptr = xptr + q*D;
            for (int p = 0; p < partition_ids.size(1); ++p) {
                int64_t pid = pid_acc[q][p];
                if (pid < 0) continue;
                ScanJob job;
                job.is_batched    = false;
                job.query_id      = (int)q;
                job.partition_id  = pid;
                job.k             = params->k;
                job.query_vector  = qptr;
                job.rank          = p;
                job_buffer_.push_back(job);
                core_resources_[pid % num_workers_].job_queue.enqueue(next_job_id_);
                next_job_id_++;
            }
        }
    } else {
        job_buffer_.resize(partition_manager_->nlist());
        auto pids = partition_manager_->get_partition_ids();
        auto pids_acc = pids.accessor<int64_t, 1>();

        if (pids.size(0) == partition_ids.size(1)) {
            vector<int> all_query_ids = std::vector<int>(x.size(0));
            std::iota(all_query_ids.begin(), all_query_ids.end(), 0);
            shared_ptr<vector<int>> all_query_ids_ptr = make_shared<vector<int>>(all_query_ids);
            for (int64_t i = 0; i < pids.size(0); i++) {
                ScanJob job;
                job.is_batched = true;
                job.partition_id = pids_acc[i];
                job.k = params->k;
                job.query_vector = x.data_ptr<float>();
                job.num_queries = x.size(0);
                job.query_ids = all_query_ids_ptr;
                job.ranks = make_shared<vector<int>>(x.size(0), i);
                int core_id = partition_manager_->get_partition_core_id(pids_acc[i]);
                if (core_id < 0) {
                    throw std::runtime_error("[QueryCoordinator::worker_scan] Invalid core ID.");
                }
                job_buffer_[next_job_id_] = job;
                core_resources_[core_id].job_queue.enqueue(next_job_id_);
                next_job_id_++;
            }

        } else {
            std::unordered_map<int64_t, shared_ptr<vector<std::pair<int, int>>>> per_partition_query_ids; // for batched scan
            for (int64_t q = 0; q < nQ; q++) {
                for (int64_t p = 0; p < partition_ids.size(1); p++) {
                    int64_t pid = pid_acc[q][p];
                    if (pid < 0) continue;
                    if (per_partition_query_ids[pid] == nullptr) {
                        per_partition_query_ids[pid] = make_shared<vector<std::pair<int, int>>>();
                    }
                    per_partition_query_ids[pid]->push_back({q, p});
                }
            }
            for (auto &kv : per_partition_query_ids) {

                auto qids_and_ranks = kv.second;
                vector<int> qids(qids_and_ranks->size());
                vector<int> ranks(qids_and_ranks->size());
                for (size_t i = 0; i < qids_and_ranks->size(); ++i) {
                    qids[i] = (*qids_and_ranks)[i].first;
                    ranks[i] = (*qids_and_ranks)[i].second;
                }

                ScanJob job;
                job.is_batched = true;
                job.partition_id = kv.first;
                job.k = params->k;
                job.query_vector = x.data_ptr<float>();
                job.num_queries = kv.second->size();
                job.query_ids = make_shared<vector<int>>(qids);
                job.ranks = make_shared<vector<int>>(ranks);
                int core_id = partition_manager_->get_partition_core_id(kv.first);
                if (core_id < 0) {
                    throw std::runtime_error("[QueryCoordinator::worker_scan] Invalid core ID.");
                }
                job_buffer_[next_job_id_] = job;
                core_resources_[core_id].job_queue.enqueue(next_job_id_);
                next_job_id_++;
            }
        }
    }
}

void QueryCoordinator::drain_and_apply_aps(Tensor x,
                                           Tensor partition_ids,
                                           bool use_aps,
                                           float recall_target,
                                           float aps_flush_period_us,
                                           shared_ptr<SearchTimingInfo> timing)
{
    // precompute boundary distances if needed
    int nQ = x.size(0);
    int D = x.size(1);
    int nprobe = partition_ids.size(1);
    vector<vector<float>> boundary_dist(nQ);
    vector<float> query_radius(nQ, 0.0f);
    vector<vector<float>> probs(nQ);
    if (use_aps) {

        for (int64_t q = 0; q < nQ; ++q) {
            vector<int64_t> partition_ids_to_scan_vec = std::vector<int64_t>(partition_ids[q].data_ptr<int64_t>(),
                                                                             partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));
            vector<float *> cluster_centroids = parent_->partition_manager_->get_vectors(partition_ids_to_scan_vec);
            boundary_dist[q] = compute_boundary_distances(x[q], cluster_centroids,  metric_ == faiss::METRIC_L2);
        }

    }

    auto last_flush = high_resolution_clock::now();

    while (true) {
        // 1) drain all ready results
        ResultJob rj;
        while (result_queue_.try_dequeue(rj)) {
            global_topk_buffer_pool_[rj.query_id]
                    ->batch_add(rj.distances.data(),
                                rj.indices.data(),
                                (int)rj.indices.size());

            job_flags_[rj.query_id][rj.rank] = true;
        }
        // 2) check done
        bool all_done = true;
        for (int64_t q = 0; q < nQ; ++q) {
            for (int64_t p = 0; p < nprobe; ++p) {
                if (!job_flags_[q][p]) {
                    all_done = false;
                    break;
                }
            }
        }
        if (all_done) break;

        // 3) APS early-stop
        if (use_aps && duration_cast<microseconds>(high_resolution_clock::now() - last_flush).count()
                       > aps_flush_period_us)
        {
            for (int64_t q = 0; q < nQ; ++q) {
                auto buf = global_topk_buffer_pool_[q];
                    float r = buf->get_kth_distance();
                    if (r != query_radius[q]) {
                        query_radius[q] = r;
                        probs[q] = compute_recall_profile(
                                boundary_dist[q],
                                r,
                                D,
                                {},
                                true,
                                metric_==faiss::METRIC_L2
                        );
                    }
                    float cum=0;
                    for (int i=0; i<(int)job_flags_[q].size(); ++i)
                        if (job_flags_[q][i]) cum += probs[q][i];
                    if (cum > recall_target) {
                        // set all jobs to done
                        for (int i=0; i<(int)job_flags_[q].size(); ++i)
                            job_flags_[q][i] = true;
                    }
                }
            }
            last_flush = high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

std::shared_ptr<SearchResult>
QueryCoordinator::aggregate_scan_results(int64_t nQ,
                                         int K,
                                         shared_ptr<SearchTimingInfo> timing)
{
    // build output tensors
    auto out_ids   = torch::full({nQ,K}, -1, torch::kInt64);
    auto out_dists = torch::full({nQ,K},
                                 (metric_==faiss::METRIC_INNER_PRODUCT
                                  ? -std::numeric_limits<float>::infinity()
                                  :  std::numeric_limits<float>::infinity()),
                                 torch::kFloat32);

    {
        std::lock_guard<std::mutex> lg(global_mutex_);
        auto id_acc = out_ids.accessor<int64_t,2>();
        auto d_acc  = out_dists.accessor<float,2>();
        for (int64_t q = 0; q < nQ; ++q) {
            auto tv = global_topk_buffer_pool_[q]->get_topk();
            auto ti = global_topk_buffer_pool_[q]->get_topk_indices();
            for (int i = 0; i < (int)ti.size() && i < K; ++i) {
                id_acc[q][i]  = ti[i];
                d_acc [q][i]  = tv[i];
            }
        }
    }

    auto res = std::make_shared<SearchResult>();
    res->ids        = out_ids;
    res->distances  = out_dists;
    res->timing_info = timing;
    return res;
}

std::shared_ptr<SearchResult> QueryCoordinator::worker_scan(
        Tensor x,
        Tensor partition_ids,
        std::shared_ptr<SearchParams> params)
{
    int64_t nQ = x.size(0), D = x.size(1);
    int     K  = params->k;
    bool    use_aps = (params->recall_target>0 && !params->batched_scan && parent_);

    auto timing = std::make_shared<SearchTimingInfo>();
    timing->n_queries  = nQ;
    timing->n_clusters = partition_manager_->nlist();
    timing->search_params = params;

    auto s1 = high_resolution_clock::now();

    // 1) init global buffers & jobs_left
    init_global_buffers(nQ, K, partition_ids, params);

    auto s2 = high_resolution_clock::now();

    // 2) copy query vec to NUMA buffers
    copy_query_to_numa(x.data_ptr<float>(), nQ, D);

    auto s3 = high_resolution_clock::now();

    // 3) enqueue jobs
    enqueue_scan_jobs(x, partition_ids, params);

    auto s4 = high_resolution_clock::now();

    // 4) drain results + APS
    drain_and_apply_aps(x, partition_ids, use_aps, params->recall_target,
                        params->aps_flush_period_us, timing);

    auto s5 = high_resolution_clock::now();

    auto res = aggregate_scan_results(nQ, K, timing);

    auto s6 = high_resolution_clock::now();

    res->timing_info->buffer_init_time_ns =
            duration_cast<nanoseconds>(s2 - s1).count();
//    res->timing_info->copy_query_time_ns =
//            duration_cast<nanoseconds>(s3 - s2).count();
    res->timing_info->job_enqueue_time_ns =
            duration_cast<nanoseconds>(s4 - s3).count();
    res->timing_info->job_wait_time_ns =
            duration_cast<nanoseconds>(s5 - s4).count();
    res->timing_info->result_aggregate_time_ns =
            duration_cast<nanoseconds>(s6 - s5).count();

    return res;
}

// Initialize Worker Threads
void QueryCoordinator::initialize_workers(int num_cores, bool use_numa) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_cores << " worker threads with use_numa=" << use_numa <<
            std::endl;

    partition_manager_->distribute_partitions(num_cores, use_numa);
    core_resources_.resize(num_cores);
    worker_threads_.resize(num_cores);
    stop_workers_.store(false);
    for (int i = 0; i < num_cores; i++) {
        if (!set_thread_affinity(i)) {
            std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
        }
        allocate_core_resources(i, 1, 10, partition_manager_->d());
        worker_threads_[i] = std::thread(&QueryCoordinator::partition_scan_worker_fn, this, i);
    }

    workers_initialized_ = true;

    // set main thread on separate thread from workers
    int num_cores_on_machine = std::thread::hardware_concurrency();
    set_thread_affinity(num_cores % num_cores_on_machine);
}

// Shutdown Worker Threads
void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    stop_workers_.store(true);
    // Enqueue a special shutdown job for each core.
    for (auto &res : core_resources_) {
        res.job_queue.enqueue(-1);
    }
    // Join all worker threads.
    for (auto &thr : worker_threads_) {
        if (thr.joinable())
            thr.join();
    }
    worker_threads_.clear();
    workers_initialized_ = false;
}

shared_ptr<SearchResult> QueryCoordinator::serial_scan(Tensor x, Tensor partition_ids,
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

        auto t1 = high_resolution_clock::now();

        auto topk_buf = std::make_shared<TopkBuffer>(k, is_descending,
                                                     /*cap=*/10 * k,
                                                     /*node=*/0);
        const float* query_vec = x_ptr + q * dimension;
        int num_parts = partition_ids.size(1);

        vector<float> boundary_distances;
        vector<float> partition_probs;
        float query_radius = 1000000.0;
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            query_radius = -1000000.0;
        }

        auto t2 = high_resolution_clock::now();



        Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids[q]);
        vector<int64_t> partition_sizes_vec = vector<int64_t>(partition_sizes.data_ptr<int64_t>(),
                                                              partition_sizes.data_ptr<int64_t>() + partition_sizes.size(0));
        auto t3 = high_resolution_clock::now();
        if (use_aps) {
            vector<int64_t> partition_ids_to_scan_vec = std::vector<int64_t>(partition_ids[q].data_ptr<int64_t>(),
                                                                partition_ids[q].data_ptr<int64_t>() + partition_ids[q].size(0));

            vector<float *> cluster_centroids = parent_->partition_manager_->get_vectors(partition_ids_to_scan_vec);
            t3 = high_resolution_clock::now();

            // trim nullptrs
            cluster_centroids.erase(std::remove(cluster_centroids.begin(), cluster_centroids.end(), nullptr),
                                     cluster_centroids.end());


            boundary_distances = compute_boundary_distances(x[q],
                                                            cluster_centroids,
                                                            metric_ == faiss::METRIC_L2);
        }
        auto t4 = high_resolution_clock::now();

        int64_t scan_time = 0;
        int64_t aps_time = 0;

        vector<int64_t> scanned_ids;

        for (int p = 0; p < num_parts; p++) {

            auto curr_time = high_resolution_clock::now();
            int64_t pi = partition_ids_accessor[q][p];

            if (pi == -1) {
                continue; // Skip invalid partitions
            }

            start_time = high_resolution_clock::now();
            float *list_vectors = (float *) partition_manager_->partition_store_->get_codes(pi);
            int64_t *list_ids = (int64_t *) partition_manager_->partition_store_->get_ids(pi);
            int64_t list_size = partition_manager_->partition_store_->list_size(pi);
            scan_list(query_vec,
                      list_vectors,
                      list_ids,
                      partition_manager_->partition_store_->list_size(pi),
                      dimension,
                      *topk_buf,
                      metric_);
            scanned_ids.push_back(pi);

            float curr_radius = topk_buf->get_kth_distance();
            float percent_change = abs(curr_radius - query_radius) / curr_radius;

            auto end_time = high_resolution_clock::now();

            scan_time += duration_cast<nanoseconds>(end_time - start_time).count();

            start_time = high_resolution_clock::now();
            bool first_list = (p == 0);
            if (use_aps && curr_radius != 0) {
                if (first_list || percent_change > search_params->recompute_threshold) {
                    query_radius = curr_radius;

                    if (search_params->use_auncel) {
                        partition_probs = compute_recall_profile_auncel(boundary_distances,
                            query_radius,
                            search_params->k,
                            search_params->auncel_a,
                            search_params->auncel_b);
                    } else {
                        partition_probs = compute_recall_profile(boundary_distances,
                                                                 query_radius,
                                                                 dimension,
                                                                 partition_sizes_vec,
                                                                 search_params->use_precomputed,
                                                                 metric_ == faiss::METRIC_L2);
                    }
                }
                float recall_estimate = 0.0;
                for (int i = 0; i < p + 1; i++) {
                    recall_estimate += partition_probs[i];
                }
                end_time = high_resolution_clock::now();
                aps_time += duration_cast<nanoseconds>(end_time - start_time).count();
                if (recall_estimate >= search_params->recall_target) {
                    break;
                }
            }
        }

        timing_info->partitions_scanned = scanned_ids.size();

        if (search_params->track_hits && maintenance_policy_) {
            maintenance_policy_->record_query_hits(std::vector<int64_t>(scanned_ids.begin(), scanned_ids.end()));
        }

        // Retrieve the top-k results for query q.
        all_topk_dists[q] = topk_buf->get_topk();
        all_topk_ids[q] = topk_buf->get_topk_indices();
        auto t5 = high_resolution_clock::now();

        // std::cout << "Query " << q << " times: " << duration_cast<microseconds>(t2 - t1).count() << " "
        //           << duration_cast<microseconds>(t3 - t2).count() << " "
        //           << duration_cast<microseconds>(t4 - t3).count() << " "
        //           << duration_cast<microseconds>(t5 - t4).count() << std::endl;
        // std::cout << "Scan time: " << scan_time / 1000.0 << " APS time: " << aps_time / 1000.0 << std::endl;
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
    Tensor partition_distances;
    if (parent_ == nullptr) {
        // scan all partitions for each query
        partition_ids_to_scan = partition_manager_->get_partition_ids();
    } else {
        auto parent_search_params = make_shared<SearchParams>();
        if (search_params->parent_params == nullptr) {
            parent_search_params->recall_target = .99;
            parent_search_params->use_precomputed = search_params->use_precomputed;
            parent_search_params->recompute_threshold = search_params->recompute_threshold;
            parent_search_params->initial_search_fraction = .5;
            parent_search_params->batched_scan = false;
        } else {
            parent_search_params = search_params->parent_params;
        }

        // if recall_target is set, we need an initial set of partitions to consider
        if (search_params->recall_target > 0.0 && !search_params->batched_scan) {
            int initial_num_partitions_to_search = std::max(
                (int) (partition_manager_->nlist() * search_params->initial_search_fraction), 1);
            parent_search_params->k = initial_num_partitions_to_search;
        } else {
            parent_search_params->k = std::min(search_params->nprobe, (int) partition_manager_->nlist());
        }

        auto parent_search_result = parent_->search(x, parent_search_params);
        partition_ids_to_scan = parent_search_result->ids;
        partition_distances = parent_search_result->distances;
        parent_timing_info = parent_search_result->timing_info;
    }

    if (search_params->use_spann && partition_distances.defined()) {
        // prune partitions based on relative distance compared to nearest centroid
        partition_distances = partition_distances / partition_distances.select(1, 0).unsqueeze(0);

        Tensor mask = partition_distances.ge(search_params->spann_eps);

        // set mask partition ids to -1
        partition_ids_to_scan.masked_fill_(mask, -1);
    }

    auto search_result = scan_partitions(x, partition_ids_to_scan, search_params);
    search_result->timing_info->parent_info = parent_timing_info;

    auto end = high_resolution_clock::now();
    search_result->timing_info->total_time_ns = duration_cast<nanoseconds>(end - start).
            count();

    return search_result;
}

shared_ptr<SearchResult> QueryCoordinator::scan_partitions(Tensor x, Tensor partition_ids,
                                                           shared_ptr<SearchParams> search_params) {

    if (partition_ids.dim() == 0) {
        throw std::runtime_error("[QueryCoordinator::scan_partitions] partition_ids is empty.");
    }
    if (partition_ids.dim() == 1) {
        partition_ids = partition_ids.unsqueeze(0).expand({x.size(0), partition_ids.size(0)});
    }
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
