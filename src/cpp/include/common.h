//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COMMON_H
#define COMMON_H

#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <mutex>
#include <atomic>
#include <utility>
#include <unordered_map>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <faiss/MetricType.h>
#include <filesystem>
#include <unordered_set>
#include <sstream>
#include <thread>
#include <pthread.h>
#include <ctime>

#ifdef QUAKE_USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#ifdef FAISS_ENABLE_GPU
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#endif

#ifdef QUAKE_OMP
#include <omp.h>
#endif

using torch::Tensor;
using std::vector;
using std::shared_ptr;
using std::tuple;
using std::make_shared;
using std::size_t;
using std::string;
using std::chrono::high_resolution_clock;
using faiss::idx_t;
using faiss::MetricType;

// constants
static const uint32_t SerializationMagicNumber = 0x44494E4C;
static const uint32_t SerializationVersion = 3;

// macros
#define DEBUG_PRINT(x) std::cout << #x << " = " << x << std::endl;

struct MaintenancePolicyParams {
    std::string maintenance_policy = "query_cost";
    int window_size = 1000;
    int refinement_radius = 100;
    int refinement_iterations = 3;
    int min_partition_size = 32;
    float alpha = .9;
    bool enable_split_rejection = true;
    bool enable_delete_rejection = true;

    float delete_threshold_ns = 20.0;
    float split_threshold_ns = 20.0;

    // de-drift parameters
    int k_large = 50;
    int k_small = 50;
    bool modify_centroids = true;

    // lire parameters
    int target_partition_size = 1000;
    float max_partition_ratio = 2.0;

    MaintenancePolicyParams() = default;
};

/**
 * @brief Parameters that govern how the DynamicIVF index should be built.
 */
struct IndexBuildParams {
    // Basic configuration
    int dimension = 0;          // e.g. d_
    int nlist = 0;              // number of clusters
    int num_workers = 0;        // concurrency (0 means main thread processes queries)
    int code_size = -1;         // e.g. for product quantization
    int num_codebooks = -1;     // e.g. for product quantization
    string metric = "l2";       // distance metric
    int niter = 5;              // number of kmeans iterations

    bool use_adaptive_nprobe = false;
    bool use_numa = false;
    bool verify_numa = false;
    bool same_core = true;
    bool verbose = false;

    shared_ptr<IndexBuildParams> parent_params = nullptr;

    IndexBuildParams() = default;
};

inline faiss::MetricType str_to_metric_type(string metric) {
    // convert the string to lowercase
    std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);

    if (metric == "l2") {
        return faiss::METRIC_L2;
    } else if (metric == "ip") {
        return faiss::METRIC_INNER_PRODUCT;
    } else {
        throw std::invalid_argument("Invalid metric type: " + metric);
    }
}

inline string metric_type_to_str(faiss::MetricType metric) {
    if (metric == faiss::METRIC_L2) {
        return "l2";
    } else if (metric == faiss::METRIC_INNER_PRODUCT) {
        return "ip";
    } else {
        throw std::invalid_argument("Invalid metric type");
    }
}

/**
* @brief Parameters for the search operation
*/
struct SearchParams {
    int nprobe = 1;
    int k = 1;
    float recall_target = -1.0f;
    float k_factor = 1.0f;
    bool use_precomputed = false;
    bool batched_scan = false;
    float recompute_threshold = 0.01f;
    float initial_search_fraction = 0.2f;
    int aps_flush_period_us = 100;

    SearchParams() = default;
};

/**
 * @brief Structure to hold timing information for building the index.
 */
struct BuildTimingInfo {
    int64_t n_vectors; ///< Number of vectors.
    int64_t n_clusters; ///< Number of clusters.
    int d; ///< Dimensionality of the vectors.
    int num_codebooks; ///< Number of codebooks used in PQ.
    int code_size; ///< Code size for PQ.
    int train_time_us; ///< Training time in microseconds.
    int assign_time_us; ///< Assignment time in microseconds.
    int total_time_us; ///< Total time in microseconds.
};

/**
 * @brief Structure to hold timing information for modify (add/remove) operations.
 */
struct ModifyTimingInfo {
    int64_t n_vectors; ///< Number of vectors.
    int find_partition_time_us; ///< Time spent on finding the partition for each vector in microseconds.
    int modify_time_us; ///< Time spent on modify operations in microseconds.
    int maintenance_time_us; ///< Time spent on maintenance operations in microseconds.
};

/**
 * @brief Structure to hold timing information for search operations.
 */
struct SearchTimingInfo {
    int64_t n_queries; ///< Number of queries.
    int64_t n_clusters; ///< Number of clusters (nlist).
    int partitions_scanned; ///< Number of partitions scanned.
    shared_ptr<SearchParams> search_params = nullptr; ///< Search parameters.
    shared_ptr<SearchTimingInfo> parent_info = nullptr; ///< Timing info for the parent index, if any.

    // main thread counters for worker scan
    int64_t buffer_init_time_ns; ///< Time spent on initializing buffers in nanoseconds.
    int64_t job_enqueue_time_ns; ///< Time spent on creating jobs in nanoseconds.
    int64_t boundary_distance_time_ns; ///< Time spent on computing boundary distances in nanoseconds.
    int64_t job_wait_time_ns; ///< Time spent waiting for jobs to complete in nanoseconds.
    int64_t result_aggregate_time_ns; ///< Time spent on aggregating results in nanoseconds.
    int64_t total_time_ns; ///< Total time spent in nanoseconds.
};

/**
 * @brief Structure to hold timing information for maintenance operations.
 */
struct MaintenanceTimingInfo {
    int64_t n_splits; ///< Number of splits.
    int64_t n_deletes; ///< Number of merges.
    int64_t delete_time_us; ///< Time spent on deletions in microseconds.
    int64_t delete_refine_time_us; ///< Time spent on deletions with refinement in microseconds.
    int64_t split_time_us; ///< Time spent on splits in microseconds.
    int64_t split_refine_time_us; ///< Time spent on splits with refinement in microseconds.
    int64_t total_time_us; ///< Total time spent in microseconds.
};

struct SearchResult {
    Tensor ids;
    Tensor distances;
    shared_ptr<SearchTimingInfo> timing_info;
};

struct Clustering {
    Tensor centroids;
    Tensor partition_ids;
    vector<Tensor> vectors;
    vector<Tensor> vector_ids;

    int64_t ntotal() const {
        int64_t n = 0;
        for (const auto &v : vectors) {
            if (v.defined() && v.numel() > 0) {
                n += v.size(0);
            }
        }
        return n;
    }

    int64_t nlist() const {
        return vectors.size();
    }

    int64_t dim() const {
        return centroids.size(1);
    }

    int64_t cluster_size(int64_t i) const {
        return vectors[i].size(0);
    }
};

#endif //COMMON_H
