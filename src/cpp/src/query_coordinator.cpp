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
//#include "parallel_hashmap/btree.h"

static void ensure_blas_buffers(QueryCoordinator::CoreResources& res,
                                size_t max_q,
                                size_t db_bs,
                                int    node)
{
    const size_t ip_need = db_bs * max_q;
    if (res.blas_ip_capacity < ip_need) {
        std::cerr << "Reallocating BLAS buffers: " << res.blas_ip_capacity
                  << " -> " << ip_need << std::endl;
        if (res.blas_ip_block) quake_free(res.blas_ip_block, res.blas_ip_capacity * sizeof(float));
        res.blas_ip_block    = static_cast<float*>(quake_alloc(ip_need * sizeof(float), node));
        res.blas_ip_capacity = ip_need;
    }
    if (res.blas_norms_x_cap < max_q) {
        if (res.blas_norms_x) quake_free(res.blas_norms_x, res.blas_norms_x_cap * sizeof(float));
        res.blas_norms_x     = static_cast<float*>(quake_alloc(max_q * sizeof(float), node));
        res.blas_norms_x_cap = max_q;
    }
    if (res.blas_norms_y_cap < db_bs) {
        if (res.blas_norms_y) quake_free(res.blas_norms_y, res.blas_norms_y_cap * sizeof(float));
        res.blas_norms_y     = static_cast<float*>(quake_alloc(db_bs * sizeof(float), node));
        res.blas_norms_y_cap = db_bs;
    }
}

// Constructor
QueryCoordinator::QueryCoordinator(shared_ptr<QuakeIndex> parent,
                                   shared_ptr<PartitionManager> partition_manager,
                                   shared_ptr<MaintenancePolicy> maintenance_policy,
                                   MetricType metric,
                                   int num_workers,
                                   bool use_numa,
                                   int num_merge_workers)
    : parent_(parent),
      partition_manager_(partition_manager),
      maintenance_policy_(maintenance_policy),
      metric_(metric),
      num_workers_(num_workers),
    num_merge_workers_(num_merge_workers),
      workers_initialized_(false) {

    if (num_workers_ > 0) {
        initialize_workers(num_workers_, num_merge_workers_, use_numa);
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

    // --- ZERO‐INITIALIZE our batched‐query buffers so we never free garbage pointers ---
    CR.batch_queries   = nullptr;

    CR.blas_ip_block = nullptr;
    CR.blas_ip_capacity = 0;
    CR.blas_norms_x = nullptr;
    CR.blas_norms_x_cap = 0;
    CR.blas_norms_y = nullptr;
    CR.blas_norms_y_cap = 0;

    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(core_idx);
#endif

    // job queue remains default‐constructed
    numa_resources_.resize(get_num_numa_nodes());
    auto& numa_res = numa_resources_[numa_node];
    size_t bytes = size_t(num_queries) * d * sizeof(float);
    if (numa_res.buffer_size != bytes) {
        quake_free(numa_res.local_query_buffer, numa_res.buffer_size);
        numa_res.local_query_buffer = static_cast<float*>(quake_alloc(bytes, numa_node));
        numa_res.buffer_size = bytes;
    }
}


// The heart of it: one function, two instantiations.
template <typename Compare>
void QueryCoordinator::merge_worker_fn(int mid) {
    auto& MR = merge_res_[mid];
    ResultJob rj;

    while (true) {
        MR.queue.wait_dequeue(rj);
        if (rj.query_id == -1)  // poison pill
            return;

        // single cast, based on the Compare template
        using Handler = typename faiss::HeapBlockResultHandler<Compare>::SingleResultHandler;
        auto* h = static_cast<Handler*>(MR.handlers[rj.query_id]);

        // feed all partial results
        for (size_t i = 0; i < rj.distances.size(); ++i) {
            h->add_result(rj.distances[i], rj.indices[i]);
        }
        // update pivot
        query_dist_pivots_[rj.query_id].store(h->threshold,
                                              std::memory_order_relaxed);

        // once all ranks for this query are in, finalize & sort
        if (!job_flags_[rj.query_id][rj.rank]) {
            job_flags_[rj.query_id][rj.rank] = true;
            if (--per_query_total_left_[rj.query_id] == 0) {
                h->end();

                // pack into pairs for sorting
                int k = h->k;
                std::vector<std::pair<float,int64_t>> result;
                result.reserve(k);
                for (int i = 0; i < k; ++i) {
                    result.emplace_back(h->heap_dis[i], h->heap_ids[i]);
                }

                // for CMin (inner-product) we want descending distances
                // for CMax (L2) we want ascending distances
                auto cmp = [](auto& a, auto& b) {
                    return Compare::cmp(a.first, b.first);
                };
                std::sort(result.begin(), result.end(), cmp);

                // write them back
                for (int i = 0; i < k; ++i) {
                    h->heap_dis[i] = result[i].first;
                    h->heap_ids[i] = result[i].second;
                }
            }
        }
        --total_left_;
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

    int i = 0;
    res.wait_time_ns = 0;
    res.process_time_ns = 0;
    res.enqueue_time_ns = 0;
    res.job_time_ns = 0;

    while (!stop_workers_) {
        int64_t jid = 0;

        auto start = std::chrono::high_resolution_clock::now();
        nr.job_queue.wait_dequeue(jid);
        auto end = std::chrono::high_resolution_clock::now();

        res.wait_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        if (jid == -1) {
            break;
        }

        auto s2 = std::chrono::high_resolution_clock::now();
        process_scan_job(job_buffer_[jid], res);
        i++;
        end = std::chrono::high_resolution_clock::now();

        res.process_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - s2).count();
        res.job_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // std::cout << "[partition_scan_worker_fn] Core: " << core_index
        //           << ", Job ID: " << jid
        //           << ", Processed: " << i
        //           << ", Wait time: " << res.wait_time_ns / 1e6 << " ms"
        //           << ", Process time: " << res.process_time_ns / 1e6 << " ms"
        //           << ", Enqueue time: " << res.enqueue_time_ns / 1e6 << " ms"
        //           << ", Job time: " << res.job_time_ns / 1e6 << " ms" << std::endl;
    }
}

void QueryCoordinator::process_scan_job(ScanJob job,
                                        CoreResources &res) {

    auto start = std::chrono::high_resolution_clock::now();
    int numa_node = 0;
#ifdef QUAKE_USE_NUMA
    numa_node = cpu_numa_node(res.core_id);
#endif
    NUMAResources &nr = numa_resources_[numa_node];

    // Attempt to fetch partition data; if the list doesn't exist, catch and enqueue empty results.
    const float   *codes = nullptr;
    const int64_t *ids   = nullptr;
    int64_t part_size    = 0;
    try {
        codes     = (float *)(partition_manager_->partition_store_->get_codes(job.partition_id));
        ids       = (int64_t *) partition_manager_->partition_store_->get_ids(job.partition_id);
        part_size = partition_manager_->partition_store_->list_size(job.partition_id);
    } catch (const std::exception &e) {
        std::cerr << "[process_scan_job] Partition " << job.partition_id
                  << " invalid: " << e.what() << ". Returning empty result(s).\n";
        if (job.is_batched) {
            for (int64_t i = 0; i < job.num_queries; ++i) {
                enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
            }
        } else {
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        }
        return;
    }

    if (part_size == 0) {
        // empty => enqueue zero‐work per query
        if (job.is_batched) {
            for (int64_t i = 0; i < job.num_queries; ++i) {
                enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
            }
        } else {
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
        }
        return;
    }
    auto end = std::chrono::high_resolution_clock::now();
    res.process_preamble_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (!job.is_batched) {
        handle_nonbatched_job(job, res, nr);
    } else {
        handle_batched_job(job, res, nr);
    }
    res.queries_counter += job.num_queries;
    res.job_counter++;
}

void QueryCoordinator::handle_nonbatched_job(const ScanJob &job,
                                             CoreResources &res,
                                             NUMAResources &nr) {

    auto start = std::chrono::high_resolution_clock::now();
    // ensure buffers
    if (res.topk_buffer_pool.size() < 1) {
        res.topk_buffer_pool.resize(1);
        res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                job.k,
                metric_ == faiss::METRIC_INNER_PRODUCT,
                /*cap=*/std::min(100 * job.k, 10000),
                /*node=*/cpu_numa_node(res.core_id)
        );
    } else if (res.topk_buffer_pool[0]->k() != job.k) {
        // check capacity
        if (res.topk_buffer_pool[0]->capacity() < job.k) {
            res.topk_buffer_pool[0] = std::make_shared<TopkBuffer>(
                    job.k,
                    metric_ == faiss::METRIC_INNER_PRODUCT,
                    /*cap=*/std::min(100 * job.k, 10000),
                    /*node=*/cpu_numa_node(res.core_id)
            );
        }
        res.topk_buffer_pool[0]->set_k(job.k);
    }

    auto buf = res.topk_buffer_pool[0];
    res.topk_buffer_pool[0]->reset();

    try {
        const float* codes = (float*)partition_manager_->partition_store_->get_codes(job.partition_id);
        const int64_t* ids = (int64_t*)partition_manager_->partition_store_->get_ids(job.partition_id);
        int64_t part_size = partition_manager_->partition_store_->list_size(job.partition_id);
        int D = partition_manager_->d();

        // Defensive check for partition validity right before scan
        if (!codes || !ids || part_size <= 0) {
            std::cerr << "[QueryCoordinator::handle_nonbatched_job] Partition " << job.partition_id
                      << " invalid or empty before scan for query " << job.query_id
                      << ". Enqueuing empty result.\n";
            enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}});
            return; // Important to return after enqueueing the placeholder
        }

        scan_list(nr.local_query_buffer + (job.query_id * D),
                  codes,
                  ids,
                  part_size,
                  D,
                  *buf,
                  metric_,
                  query_dist_pivots_[job.query_id].load(std::memory_order_relaxed));

        auto end = std::chrono::high_resolution_clock::now();

        res.scan_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();

        // If scan_list completes, enqueue its results
        auto tv = buf->get_topk(false);
        auto ti = buf->get_topk_indices(false);
        enqueue_result_job(ResultJob{job.query_id, job.rank, std::move(tv), std::move(ti)});

        end = std::chrono::high_resolution_clock::now();
        res.enqueue_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();



    } catch (const std::exception& e) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id << ": " << e.what()
                  << ". Enqueuing empty result.\n";
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    } catch (...) {
        std::cerr << "[QueryCoordinator::handle_nonbatched_job] Unknown exception during scan for partition "
                  << job.partition_id << ", query " << job.query_id
                  << ". Enqueuing empty result.\n";
        enqueue_result_job(ResultJob{job.query_id, job.rank, {}, {}}); // Enqueue empty result on error
    }
}

void QueryCoordinator::handle_batched_job(const ScanJob &job,
                                          CoreResources &res,
                                          NUMAResources &nr) {

    auto start = std::chrono::high_resolution_clock::now();
    auto s1 = std::chrono::high_resolution_clock::now();
    // Total queries, Top-K, dimension, NUMA node
    int64_t Q    = job.num_queries;
    int     K    = job.k;
    int     D    = partition_manager_->d();
    int     node = cpu_numa_node(res.core_id);

    // Fetch partition data
    const float   *codes     = (float *) partition_manager_->partition_store_->get_codes(job.partition_id);
    const int64_t *ids       = partition_manager_->partition_store_->get_ids(job.partition_id);
    int64_t        part_size = partition_manager_->partition_store_->list_size(job.partition_id);
    if (!codes || !ids || part_size <= 0) {
        for (int64_t i = 0; i < Q; ++i) {
            enqueue_result_job(ResultJob{(*job.query_ids)[i], (*job.ranks)[i], {}, {}});
        }
        return;
    }

    // 1) Prepare per-thread buffers *once*
    size_t cap = std::min(100 * K, 10000);
    int64_t queries_req = Q;

    if (res.topk_buffer_pool.size() < (size_t)queries_req) {
        res.topk_buffer_pool.resize(queries_req);
        for (size_t i = 0; i < (size_t)queries_req; ++i) {
            res.topk_buffer_pool[i] =
                    std::make_shared<TopkBuffer>(K,
                                                 metric_ == faiss::METRIC_INNER_PRODUCT,
                                                 cap,
                                                 node);
        }
    }

    ensure_blas_buffers(res, Q, BLAS_DB_BS, node);

    size_t max_q = size_t(queries_req) * D;
    if (res.batch_q_capacity < max_q) {
        quake_free(res.batch_queries, res.batch_q_capacity * sizeof(float));
        res.batch_queries    = static_cast<float*>(quake_alloc(max_q * sizeof(float), node));
        res.batch_q_capacity = max_q;
    }

    // reset only the first 'chunk' TopK buffers
    for (int64_t i = 0; i < Q; ++i) {
        auto &buf = res.topk_buffer_pool[i];
        buf->set_k(K);
        buf->reset();
    }

    auto s2 = std::chrono::high_resolution_clock::now();

    // // init only the first chunk*K slots in scratch
    // float init_val = (metric_ == faiss::METRIC_INNER_PRODUCT)
    //                  ? -std::numeric_limits<float>::infinity()
    //                  :  std::numeric_limits<float>::infinity();
    // std::fill_n(res.batch_distances, Q * K, init_val);
    // std::fill_n(res.batch_ids,       Q * K, -1LL);

    // auto

    // gather queries
    float *qptr = nullptr;
    float *dst = res.batch_queries;
    for (int64_t i = 0; i < Q; ++i) {
        int qid = (*job.query_ids)[i];
        const float *src = nr.local_query_buffer + size_t(qid) * D;
        std::memcpy(dst + i * D, src, D * sizeof(float));
    }
    qptr = dst;

    vector<std::atomic<float> *> pivots;
    pivots.resize(job.num_queries);
    for (int64_t i = 0; i < Q; ++i) {
        int qid = (*job.query_ids)[i];
        pivots[i] = &query_dist_pivots_[qid];
    }

    auto s3 = std::chrono::high_resolution_clock::now();

    // check that things are on the proper NUMA node
    // bool ok = true;
    // ok = ok && verify_numa_locality(qptr, "qptr");
    // ok = ok && verify_numa_locality(codes, "codes");
    // ok = ok && verify_numa_locality(ids, "ids");
    // ok = ok && verify_numa_locality(res.batch_queries, "batch_queries");
    // ok = ok && verify_numa_locality(res.blas_ip_block, "blas_ip_block");
    // ok = ok && verify_numa_locality(res.blas_norms_x, "blas_norms_x");
    // ok = ok && verify_numa_locality(res.blas_norms_y, "blas_norms_y");
    // for (int64_t i = 0; i < Q; ++i) {
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->ord_, "ord_");
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->vals_, "vals_");
    //     ok = ok && verify_numa_locality(res.topk_buffer_pool[i]->ids_, "ids_");
    // }
    // if (!ok) {
    //     std::cerr << "[QueryCoordinator::handle_batched_job] NUMA locality check failed.\n";
    //     // throw std::runtime_error("NUMA locality check failed");
    // }

    // run the scan on this chunk
    batched_scan_list(
            qptr,
            codes, ids,
            Q, part_size, D,
            res.topk_buffer_pool,
            metric_,
            res.blas_ip_block,
            res.blas_norms_x,
            res.blas_norms_y,
            BLAS_DB_BS,
            pivots);

    auto s4 = std::chrono::high_resolution_clock::now();

    auto end = std::chrono::high_resolution_clock::now();
    res.scan_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    // collect results for this chunk
    std::vector<ResultJob> results_batch;
    results_batch.reserve(Q);
    for (int64_t i = 0; i < Q; ++i) {
        int global_q = (*job.query_ids)[i];
        int rank_q   = (*job.ranks)    [i];
        auto tv = res.topk_buffer_pool[i]->get_topk(false);
        auto ti = res.topk_buffer_pool[i]->get_topk_indices(false);
        enqueue_result_job(ResultJob{global_q, rank_q, std::move(tv), std::move(ti)});
    }

    end = std::chrono::high_resolution_clock::now();
    auto s5 = std::chrono::high_resolution_clock::now();
    res.enqueue_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // // print out debug timing info s1, ... s5
    // std::cout << "QueryCoordinator::handle_batched_job: "
    //           << "job_id: " << job.job_id
    //             << ", core_id: " << res.core_id
    //             << ", num_queries: " << job.num_queries
    //             << ", partition_id: " << job.partition_id
    //             << ", k: " << job.k
    //             << ", rank: " << job.rank
    //             << ", numa_node: " << node
    //           << ", preamble: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s2 - s1).count()
    //           << ", query copy: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s3 - s2).count()
    //           << ", scan: " << std::chrono::duration_cast<std::chrono::nanoseconds>(s4 - s3).count()
    //           << ", enqueue: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - s4).count()
    //           << std::endl;
}

// --- replace old enqueue -------------------------------------------------
inline void QueryCoordinator::enqueue_result_job(ResultJob job)
{
    if (job.query_id < 0) {
        // Poison pill to stop the merge worker
        for (auto& mr : merge_res_) {
            mr.queue.enqueue(ResultJob{-1, 0, {}, {}});
        }
        return;
    }
    const size_t mid = static_cast<size_t>(job.query_id) % num_merge_workers_;
    merge_res_[mid].queue.enqueue(std::move(job));
}


void QueryCoordinator::init_global_buffers(int64_t nQ,
                                           int K,
                                           Tensor &partition_ids,
                                           shared_ptr<SearchParams> params) {
    std::lock_guard<std::mutex> lg(global_mutex_);
    // resize or reset

    float * vals = (float *) quake_alloc(nQ * K * sizeof(float), 0);
    int64_t * ids = (int64_t *) quake_alloc(nQ * K * sizeof(int64_t), 0);
    std::fill_n(ids,  nQ * K, -1);

    float max_val = (metric_ == faiss::METRIC_INNER_PRODUCT)
              ? -std::numeric_limits<float>::infinity()
              :  std::numeric_limits<float>::infinity();

    std::fill_n(vals, nQ * K, max_val);

    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        global_max_heaps_ = std::make_shared<
            faiss::HeapBlockResultHandler<
                faiss::CMin<float,int64_t>>>(nQ, vals, ids, K);
    } else {
        global_min_heaps_ = std::make_shared<
            faiss::HeapBlockResultHandler<
                faiss::CMax<float,int64_t>>>(nQ, vals, ids, K);
    }

    for (auto& mr : merge_res_) {
        mr.handlers.clear();
        mr.handlers.resize(nQ, nullptr);

        /* allocate handler objects once per query --------------------------- */
        if (metric_ == faiss::METRIC_INNER_PRODUCT) {
            using H = faiss::HeapBlockResultHandler<
                        faiss::CMin<float,int64_t>>::SingleResultHandler;
            for (int64_t q = 0; q < nQ; ++q) {
                mr.handlers[q] = new H(*global_max_heaps_);
                static_cast<H*>(mr.handlers[q])->begin(q);
            }
        } else {
            using H = faiss::HeapBlockResultHandler<
                        faiss::CMax<float,int64_t>>::SingleResultHandler;
            for (int64_t q = 0; q < nQ; ++q) {
                mr.handlers[q] = new H(*global_min_heaps_);
                static_cast<H*>(mr.handlers[q])->begin(q);
            }
        }
    }

    query_dist_pivots_ = vector<std::atomic<float>>(nQ);
    for (int64_t q = 0; q < nQ; ++q) {
        query_dist_pivots_[q].store(max_val, std::memory_order_relaxed);
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

    vector<int> core_to_numa(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
        core_to_numa[i] = cpu_numa_node(i);
    }

    // std::cout << "[enqueue_scan_jobs] Enqueuing jobs for " << nQ
    //           << " queries, " << partition_ids.size(1)
    //           << " partitions, k = " << params->k
    //           << ", batched scan: " << (params->batched_scan ? "yes" : "no") << std::endl;

    // Reset job state
    next_job_id_ = 0;
    total_left_.store(0, std::memory_order_relaxed);
    job_flags_.clear();
    job_flags_.resize(nQ);
    per_query_total_left_ = vector<std::atomic<int>>(nQ);
    for (int64_t q = 0; q < nQ; ++q) {
        job_flags_[q] = vector<std::atomic<bool>>(partition_ids.size(1));

        int valid_count = 0;
        for (int p = 0; p < partition_ids.size(1); ++p) {
            job_flags_[q][p].store(false);
            if (partition_ids_acc[q][p] < 0) {
                job_flags_[q][p] = true;
            } else {
                valid_count++;
            }
        }
        per_query_total_left_[q].store(valid_count, std::memory_order_relaxed);
    }
    job_buffer_.clear();
    job_buffer_.reserve(nQ * partition_ids.size(1));

    auto pid_acc = partition_ids.accessor<int64_t,2>();
    if (!params->batched_scan) {
        // one job per (q,p)
        for (int64_t q = 0; q < nQ; ++q) {
            const float* qptr = xptr + q*D;
            for (int p = 0; p < partition_ids.size(1); ++p) {
                int64_t pid = pid_acc[q][p];
                if (pid < 0) continue;
                ScanJob job;
                job.job_id        = next_job_id_;
                job.is_batched    = false;
                job.query_id      = (int)q;
                job.partition_id  = pid;
                job.k             = params->k;
                job.rank          = p;
                job_buffer_.push_back(job);
                numa_resources_[core_to_numa[pid % num_workers_]].job_queue.enqueue(next_job_id_);
                next_job_id_++;
                total_left_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    } else {
        /* Build per-partition query lists -------------------------------------- */
        std::unordered_map<int64_t, std::vector<std::pair<int,int>>> qlist;
        for (int64_t q = 0; q < nQ; ++q) {
            for (int p = 0; p < partition_ids.size(1); ++p) {
                int64_t pid = pid_acc[q][p];
                if (pid >= 0) qlist[pid].emplace_back(q, p);
            }
        }

        int nlist = partition_manager_->nlist();
        bool scan_all = false;
        if (nlist == 1) {
            scan_all = true;
        }

        /* Emit ScanJobs, already split into ≤ params->batch_size chunks -------------- */
        for (int64_t pid = 0; pid < (int64_t)qlist.size(); ++pid) {
            auto &pairs = qlist[pid];
            if (pairs.empty()) continue;

            for (size_t off = 0; off < pairs.size(); off += params->batch_size) {
                size_t chunk = std::min<size_t>(params->batch_size, pairs.size() - off);

                // Split query / rank vectors for this chunk.
                auto qids  = std::make_shared<std::vector<int>>();
                auto ranks = std::make_shared<std::vector<int>>();
                qids ->reserve(chunk);
                ranks->reserve(chunk);
                for (size_t i = 0; i < chunk; ++i) {
                    qids ->push_back(pairs[off + i].first);
                    ranks->push_back(pairs[off + i].second);
                }

                if (chunk < MIN_BATCH_SCAN_SIZE) {
                    for (size_t i = 0; i < chunk; ++i) {
                        int qid = qids->at(i);
                        int rank = ranks->at(i);
                        ScanJob job;
                        job.is_batched   = false;
                        job.job_id       = next_job_id_;
                        job.partition_id = pid;
                        job.k           = params->k;
                        job.rank         = rank;
                        job.query_id     = qid;
                        job_buffer_.push_back(job);
                        numa_resources_[core_to_numa[pid % num_workers_]].job_queue.enqueue(next_job_id_);
                        next_job_id_++;
                        total_left_.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    ScanJob job;
                    job.is_batched   = true;
                    job.job_id        = next_job_id_;
                    job.partition_id = pid;
                    job.k            = params->k;
                    job.num_queries  = static_cast<int>(chunk);
                    job.query_ids    = qids;
                    job.ranks        = ranks;
                    job.scan_all     = scan_all;

                    job_buffer_.push_back(job);

                    // Choose NUMA queue by partition-to-core mapping.
                    int core = pid % num_workers_;
                    int node = core_to_numa[core];
                    numa_resources_[node].job_queue.enqueue(next_job_id_);
                    next_job_id_++;
                    total_left_.fetch_add(chunk, std::memory_order_relaxed);
                }
            }
        }
    }
}


void QueryCoordinator::drain_and_apply_aps(Tensor                      x,
                                           Tensor                      partition_ids,
                                           shared_ptr<SearchParams> search_params,
                                           std::shared_ptr<SearchTimingInfo> timing)
{

    int64_t nQ = x.size(0), D = x.size(1);
    // compute boundary distances

    while (total_left_.load(std::memory_order_relaxed) > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(search_params->aps_flush_period_us));



    // mark hits
    if (search_params->track_hits && maintenance_policy_) {
        for (int64_t q = 0; q < nQ; ++q) {
            std::vector<int64_t> scanned_ids;
            scanned_ids.reserve(partition_ids.size(1));
            for (int p = 0; p < partition_ids.size(1); ++p) {
                int64_t pid = partition_ids[q][p].item<int64_t>();
                if (pid < 0) continue;
                scanned_ids.emplace_back(pid);
            }
            maintenance_policy_->record_query_hits(scanned_ids);
        }
        // maintenance_policy_->record_query_hits(std::vector<int64_t>(scanned_ids.begin(), scanned_ids.end()));
    }


        // check if we need to apply APS
}

std::shared_ptr<SearchResult>
QueryCoordinator::aggregate_scan_results(int64_t nQ,
                                         int K,
                                         shared_ptr<SearchTimingInfo> timing,
                                         Tensor out_ids,
                                         Tensor out_dists) {
    auto id_acc = out_ids.accessor<int64_t,2>();
    auto d_acc  = out_dists.accessor<float,2>();

    // copy results from the global heaps to the output tensors
    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        for (int64_t q = 0; q < nQ; ++q) {
            for (int64_t i = 0; i < K; ++i) {
                id_acc[q][i]  = global_max_heaps_->heap_ids_tab[q * K + i];
                d_acc [q][i]  = global_max_heaps_->heap_dis_tab[q * K + i];
            }
        }
    } else {
        for (int64_t q = 0; q < nQ; ++q) {
            for (int64_t i = 0; i < K; ++i) {
                id_acc[q][i]  = global_min_heaps_->heap_ids_tab[q * K + i];
                d_acc [q][i]  = global_min_heaps_->heap_dis_tab[q * K + i];
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

    int64_t nJobsExpected = 0;

    auto timing = std::make_shared<SearchTimingInfo>();
    timing->n_queries  = nQ;
    timing->n_clusters = partition_manager_->nlist();
    timing->search_params = params;

    // get initial values of the per-core resource timers;
    vector<int64_t> core_wait_time_ns(num_workers_, 0);
    vector<int64_t> core_process_time_ns(num_workers_, 0);
    vector<int64_t> core_process_preamble_time_ns(num_workers_, 0);
    vector<int64_t> core_enqueue_time_ns(num_workers_, 0);
    vector<int64_t> core_job_time_ns(num_workers_, 0);
    vector<int64_t> core_scan_setup_time_ns(num_workers_, 0);
    vector<int64_t> core_scan_time_ns(num_workers_, 0);
    vector<int64_t> core_scan_push_time_ns(num_workers_, 0);
    for (int i = 0; i < num_workers_; ++i) {
        core_wait_time_ns[i] = core_resources_[i].wait_time_ns;
        core_process_time_ns[i] = core_resources_[i].process_time_ns;
        core_process_preamble_time_ns[i] = core_resources_[i].process_preamble_time_ns;
        core_enqueue_time_ns[i] = core_resources_[i].enqueue_time_ns;
        core_job_time_ns[i] = core_resources_[i].job_time_ns;
        core_scan_time_ns[i] = core_resources_[i].scan_time_ns;
    }

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

    Tensor out_ids = torch::empty({nQ, K}, torch::kLong);
    Tensor out_dists = torch::empty({nQ, K}, torch::kFloat);

    // 4) drain results + APS
    drain_and_apply_aps(x, partition_ids, params, timing);

    auto s5 = high_resolution_clock::now();

    auto res = aggregate_scan_results(nQ, K, timing, out_ids, out_dists);

    auto s6 = high_resolution_clock::now();

    res->timing_info->buffer_init_time_ns =
            duration_cast<nanoseconds>(s2 - s1).count();
    res->timing_info->copy_query_time_ns =
            duration_cast<nanoseconds>(s3 - s2).count();
    res->timing_info->job_enqueue_time_ns =
            duration_cast<nanoseconds>(s4 - s3).count();
    res->timing_info->job_wait_time_ns =
            duration_cast<nanoseconds>(s5 - s4).count();
    res->timing_info->result_aggregate_time_ns =
            duration_cast<nanoseconds>(s6 - s5).count();
    //
    // // retrieve the final values of the per-core resource timers;
    for (int i = 0; i < num_workers_; ++i) {
        core_wait_time_ns[i] = core_resources_[i].wait_time_ns - core_wait_time_ns[i];
        core_process_time_ns[i] = core_resources_[i].process_time_ns - core_process_time_ns[i];
        core_process_preamble_time_ns[i] = core_resources_[i].process_preamble_time_ns - core_process_preamble_time_ns[i];
        core_enqueue_time_ns[i] = core_resources_[i].enqueue_time_ns - core_enqueue_time_ns[i];
        core_job_time_ns[i] = core_resources_[i].job_time_ns - core_job_time_ns[i];
        core_scan_time_ns[i] = core_resources_[i].scan_time_ns - core_scan_time_ns[i];
    }


    // // // print out the per-core resource timers;
    // for (int i = 0; i < num_workers_; ++i) {
    //     std::cout << "[QueryCoordinator::worker_scan] Core " << i << ": "
    //                 << "job_counter=" << core_resources_[i].job_counter << " "
    //                 << "queries_counter=" << core_resources_[i].queries_counter << " "
    //               << "wait_time_ms=" << (float) core_wait_time_ns[i] / 1e6 << " "
    //                 << "scan_setup_time_ms=" << (float) core_scan_setup_time_ns[i] / 1e6 << " "
    //                 << "scan_time_ms=" << (float) core_scan_time_ns[i] / 1e6 << " "
    //     << "scan_push_time_ms=" << (float) core_scan_push_time_ns[i] / 1e6 << " "
    //               << "process_time_ms=" << (float) core_process_time_ns[i] / 1e6 << " "
    //     << "process_preamble_time_ms=" << (float) core_process_preamble_time_ns[i] / 1e6 << " "
    //               << "enqueue_time_ms=" << (float) core_enqueue_time_ns[i] / 1e6 << " "
    //               << "job_time_ms=" << (float) core_job_time_ns[i] / 1e6 << std::endl;
    // }
    //
    // // print out the main thread timers;
    // std::cout << "[QueryCoordinator::worker_scan] Main thread: "
    //           << "buffer_init_time_ms=" << (float) res->timing_info->buffer_init_time_ns / 1e6 << " "
    //           << "copy_query_time_ms=" << (float) res->timing_info->copy_query_time_ns / 1e6 << " "
    //           << "job_enqueue_time_ms=" << (float) res->timing_info->job_enqueue_time_ns / 1e6 << " "
    //           << "job_wait_time_ms=" << (float) res->timing_info->job_wait_time_ns / 1e6 << " "
    //           << "result_aggregate_time_ms=" << (float) res->timing_info->result_aggregate_time_ns / 1e6
    //           << std::endl;

    return res;
}

// Initialize Worker Threads
void QueryCoordinator::initialize_workers(int num_workers, int num_merge_workers, bool use_numa) {
    if (workers_initialized_) {
        std::cerr << "[QueryCoordinator::initialize_workers] Workers already initialized." << std::endl;
        return;
    }

    std::cout << "[QueryCoordinator::initialize_workers] Initializing " << num_workers << " worker threads with use_numa=" << use_numa <<
            std::endl;

    partition_manager_->distribute_partitions(num_workers, use_numa);
    core_resources_.resize(num_workers);
    worker_threads_.resize(num_workers);
    stop_workers_.store(false);
    for (int i = 0; i < num_workers; i++) {
        if (!set_thread_affinity(i)) {
            std::cout << "[QueryCoordinator::initialize_workers] Failed to set thread affinity on core " << i << std::endl;
        }
        allocate_core_resources(i, 1, 10, partition_manager_->d());
        worker_threads_[i] = std::thread(&QueryCoordinator::partition_scan_worker_fn, this, i);
    }

    merge_threads_.resize(num_merge_workers_);
    merge_res_.resize(num_merge_workers_);
    if (metric_ == faiss::METRIC_INNER_PRODUCT) {
        // CMin: we want largest-inner-product first
        for (int i = 0; i < num_merge_workers_; ++i) {
            merge_threads_[i] = std::thread(
              &QueryCoordinator::merge_worker_fn<faiss::CMin<float,int64_t>>,
              this, i);
        }
    } else {
        // CMax: we want smallest-L2 first
        for (int i = 0; i < num_merge_workers_; ++i) {
            merge_threads_[i] = std::thread(
              &QueryCoordinator::merge_worker_fn<faiss::CMax<float,int64_t>>,
              this, i);
        }
    }

    workers_initialized_ = true;

    // set main thread on separate thread from workers
    int num_cores_on_machine = std::thread::hardware_concurrency();
    set_thread_affinity(num_workers + num_merge_workers);
    // set_thread_affinity(0);
}

// Shutdown Worker Threads
void QueryCoordinator::shutdown_workers() {
    if (!workers_initialized_) {
        return;
    }

    stop_workers_.store(true);
    // Enqueue poison pills to all worker threads.
    for (auto &res : numa_resources_) {
        for (int i = 0; i < num_workers_; ++i)
            res.job_queue.enqueue(-1);
    }

    // Enqueue poison pills to all merge workers.
    for (int m = 0; m < num_merge_workers_; ++m) {
        merge_res_[m].queue.enqueue(ResultJob{-1, 0, {}, {}});
    }

    // Join all worker threads.
    for (auto &thr : worker_threads_) {
        if (thr.joinable())
            thr.join();
    }

    for (auto &t : merge_threads_) {
        if (t.joinable())
            t.join();
    }


    merge_threads_.clear();
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
                      metric_,
                      NULL);
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
//            parent_search_params->recall_target = .99;
            parent_search_params->use_precomputed = search_params->use_precomputed;
            parent_search_params->recompute_threshold = search_params->recompute_threshold;
//            parent_search_params->initial_search_fraction = .5;
            parent_search_params->batched_scan = false;

            if (x.size(0) > 10) {
                parent_search_params->batched_scan = true;
            }
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

    static CoreResources serial_res;                    // one per process
    ensure_blas_buffers(serial_res, x.size(0), BLAS_DB_BS, /*node=*/0);

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
                          metric_,
                          /* BLAS scratch */ serial_res.blas_ip_block,
                                            serial_res.blas_norms_x,
                                            serial_res.blas_norms_y,
                          BLAS_DB_BS);

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
