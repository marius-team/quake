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

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/chunked_array.h>

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

using torch::Tensor;
using std::vector;
using std::unordered_map;
using std::shared_ptr;
using std::tuple;
using std::make_shared;
using std::size_t;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using faiss::idx_t;
using faiss::MetricType;

// constants
static const uint32_t SerializationMagicNumber = 0x44494E4C;
static const uint32_t SerializationVersion = 3;

// Default constants for index build parameters
constexpr int DEFAULT_NLIST = 0;                   ///< Default number of clusters (lists); if not specified, a flat index is assumed.
constexpr int DEFAULT_NITER = 5;                   ///< Default number of k-means iterations used during clustering.
constexpr const char* DEFAULT_METRIC = "l2";       ///< Default distance metric (either "l2" for Euclidean or "ip" for inner product).
constexpr int DEFAULT_NUM_WORKERS = 0;             ///< Default number of workers (0 means single-threaded).

// Default constants for search parameters
constexpr int DEFAULT_K = 1;                             ///< Default number of neighbors to return.
constexpr int DEFAULT_NPROBE = 1;                        ///< Default number of partitions to probe during search.
constexpr float DEFAULT_RECALL_TARGET = -1.0f;           ///< Default recall target (a negative value means no adaptive search).
constexpr bool DEFAULT_BATCHED_SCAN = false;             ///< Default flag for batched scanning.
constexpr bool DEFAULT_PRECOMPUTED = true;               ///< Default flag to use precomputed incomplete beta fn for APS.
constexpr float DEFAULT_INITIAL_SEARCH_FRACTION = 0.02f; ///< Default initial fraction of partitions to search.
constexpr float DEFAULT_RECOMPUTE_THRESHOLD = 0.001f;    ///< Default threshold to trigger recomputation of search parameters.
constexpr int DEFAULT_APS_FLUSH_PERIOD_US = 100;         ///< Default period (in microseconds) for flushing the APS buffer.
constexpr int DEFAULT_PRICE_THRESHOLD = INT_MAX;

// Default constants for maintenance policy parameters
constexpr const char* DEFAULT_MAINTENANCE_POLICY = "query_cost"; ///< Default maintenance policy type.
constexpr int DEFAULT_WINDOW_SIZE = 1000;              ///< Default window size for measuring hit rates.
constexpr int DEFAULT_REFINEMENT_RADIUS = 25;         ///< Default radius for local partition refinement.
constexpr int DEFAULT_REFINEMENT_ITERATIONS = 3;       ///< Default number of iterations for refinement.
constexpr int DEFAULT_MIN_PARTITION_SIZE = 32;         ///< Default minimum allowed partition size.
constexpr float DEFAULT_ALPHA = 0.9f;                  ///< Default alpha parameter for maintenance.
constexpr bool DEFAULT_ENABLE_SPLIT_REJECTION = true;  ///< Default flag to enable rejection of splits.
constexpr bool DEFAULT_ENABLE_DELETE_REJECTION = true; ///< Default flag to enable rejection of deletions.
constexpr float DEFAULT_DELETE_THRESHOLD_NS = 10.0f;   ///< Default threshold in nanoseconds for deletion decisions.
constexpr float DEFAULT_SPLIT_THRESHOLD_NS = 10.0f;    ///< Default threshold in nanoseconds for split decisions.

const vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_N = {1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536};   ///< Default range of n values for latency estimator.
const vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_K = {1, 4, 16, 64, 256};                                ///< Default range of k values for latency estimator.
constexpr int DEFAULT_LATENCY_ESTIMATOR_NTRIALS = 5;                                                          ///< Default number of trials for latency estimator.

// macros
#define DEBUG_PRINT(x) std::cout << #x << " = " << x << std::endl;

struct MaintenancePolicyParams {
    std::string maintenance_policy = DEFAULT_MAINTENANCE_POLICY;
    int window_size = DEFAULT_WINDOW_SIZE;
    int refinement_radius = DEFAULT_REFINEMENT_RADIUS;
    int refinement_iterations = DEFAULT_REFINEMENT_ITERATIONS;
    int min_partition_size = DEFAULT_MIN_PARTITION_SIZE;
    float alpha = DEFAULT_ALPHA;
    bool enable_split_rejection = DEFAULT_ENABLE_SPLIT_REJECTION;
    bool enable_delete_rejection = DEFAULT_ENABLE_DELETE_REJECTION;

    float delete_threshold_ns = DEFAULT_DELETE_THRESHOLD_NS;
    float split_threshold_ns = DEFAULT_SPLIT_THRESHOLD_NS;

    MaintenancePolicyParams() = default;
};

/**
 * @brief Parameters that govern how the DynamicIVF index should be built.
 */
struct IndexBuildParams {
    // Basic configuration
    int dimension = 0;
    int nlist = DEFAULT_NLIST;
    int num_workers = DEFAULT_NUM_WORKERS;
    int code_size = -1;         // for PQ
    int num_codebooks = -1;     // for PQ
    string metric = DEFAULT_METRIC;
    int niter = DEFAULT_NITER;

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

enum class FilteringType {
    PRE_FILTERING,
    POST_FILTERING,
    IN_FILTERING
};

/**
* @brief Parameters for the search operation
*/
struct SearchParams {
    int nprobe = DEFAULT_NPROBE;
    int k = DEFAULT_K;
    float recall_target = DEFAULT_RECALL_TARGET;
    int num_threads = 1; // number of threads to use for search within a single worker
    float k_factor = 1.0f;
    bool use_precomputed = DEFAULT_PRECOMPUTED;
    bool batched_scan = DEFAULT_BATCHED_SCAN;
    float recompute_threshold = DEFAULT_RECOMPUTE_THRESHOLD;
    float initial_search_fraction = DEFAULT_INITIAL_SEARCH_FRACTION;
    int aps_flush_period_us = DEFAULT_APS_FLUSH_PERIOD_US;
    int price_threshold = DEFAULT_PRICE_THRESHOLD;
    FilteringType filteringType = FilteringType::IN_FILTERING;

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
    int input_validation_time_us; ///< Time spent on input validation in microseconds.
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
    vector<shared_ptr<arrow::Table>> attributes_tables;

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
