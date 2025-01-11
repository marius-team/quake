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
    faiss::MetricType metric = faiss::METRIC_L2; // distance metric
    int niter = 5;              // number of kmeans iterations

    bool use_adaptive_nprobe = false;
    bool use_numa = false;
    bool verify_numa = false;
    bool same_core = true;
    bool verbose = false;

    shared_ptr<IndexBuildParams> parent_params = nullptr;

    IndexBuildParams() = default;
};

/**
* @brief Paramters for the search operation
*/
struct SearchParams {
    int nprobe = 1;
    int k = 1;
    float recall_target = 1.0f;
    float k_factor = 1.0f;
    bool use_precomputed = false;
    bool batched_scan = false;
    float recompute_threshold = 0.1f;

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

    /**
     * @brief Prints the timing information.
     */
    void print() const {
        std::cout << "#### Build Timing Information ####" << std::endl;
        std::cout << "N = " << n_vectors << ", D = " << d << ", K = " << n_clusters
                << ", M = " << num_codebooks << ", code_size = " << code_size << std::endl;
        std::cout << "Train time (us): " << train_time_us << std::endl;
        std::cout << "Assign time (us): " << assign_time_us << std::endl;
        std::cout << "Total time (us): " << total_time_us << std::endl;
    }
};

/**
 * @brief Structure to hold timing information for modify (add/remove) operations.
 */
struct ModifyTimingInfo {
    int64_t n_vectors; ///< Number of vectors.
    int find_partition_time_us; ///< Time spent on finding the partition for each vector in microseconds.
    int modify_time_us; ///< Time spent on modify operations in microseconds.
    int maintenance_time_us; ///< Time spent on maintenance operations in microseconds.

    /**
    * @brief Prints the timing information.
    */
    void print() const {
        std::cout << "#### Modify Timing Information ####" << std::endl;
        std::cout << "N = " << n_vectors << std::endl;
        std::cout << "Find partition time (us): " << find_partition_time_us << std::endl;
        std::cout << "Modify time (us): " << modify_time_us << std::endl;
        std::cout << "Maintenance time (us): " << maintenance_time_us << std::endl;
    }
};

/**
 * @brief Structure to hold timing information for search operations.
 */
struct SearchTimingInfo {
    int64_t n_queries; ///< Number of queries.
    int64_t n_vectors; ///< Total number of vectors in the index.
    int64_t n_clusters; ///< Number of clusters (nlist).
    int d; ///< Dimensionality of the vectors.
    int num_codebooks; ///< Number of codebooks used in PQ.
    int code_size; ///< Code size for PQ.
    int k; ///< Number of nearest neighbors to search for.
    int nprobe; ///< Number of clusters to probe.
    float k_factor; ///< Multiplicative factor for k in searches.
    float recall_target; ///< Target recall value.

    int metadata_update_time_us; ///< Time spent updating the metadata in microseconds.
    int quantizer_search_time_us; ///< Time spent searching the quantizer in microseconds.
    int scan_pq_time_us; ///< Time spent scanning PQ codes in microseconds.
    int refine_time_us; ///< Time spent refining search results in microseconds.
    int partition_scan_time_us; ///< Time spent scanning partitions in microseconds.
    int total_time_us; ///< Total search time in microseconds.
    int partition_scan_setup_time_us; ///< Time spent in setting up the values we need to perform the partition scan
    int partition_scan_search_time_us; ///< Time spent actually perform the partition scans
    int partition_scan_post_process_time_us; ///< Time spent performing any post processing
    int average_worker_job_time_us; ///< Total time spent by the workers in performing the job
    int average_worker_scan_time_us; ///< Total time spent by the workers in performing the scans
    int target_vectors_scanned; ///< The number of vectors we expect to scan
    int total_vectors_scanned; ///< Total number of vectors scanned by the workers in performing the scan
    float average_worker_throughput; ///< Average throughput measured by the worker
    float recall_profile_us; ///< Time spent profiling the recall
    float boundary_time_us; ///< Time spent computing boundaries
    shared_ptr<SearchTimingInfo> parent_info = nullptr; ///< Timing info for the parent index, if any.

    int total_numa_preprocessing_time_us; ///< Total preprocessing time to setup the numa workers
    int total_numa_adaptive_preprocess_time_us; ///< Total time spent setting up adaptive probing
    int total_job_distribute_time_us; ///< Total time spent distributing the jobs to the numa workers
    int total_result_wait_time_us; ///< Total time spent by the main thread waiting for the result
    int total_adaptive_calculate_time_us; ///< Total time spent by the main thread in the adaptive nprobe calculation
    int total_shared_flush_time; ///< Total time spent by the main thread in flushing the shared buffer
    int total_numa_postprocessing_time_us; ///< Total postprocessing time to combine the results from the numa workers
    bool using_faiss_index; ///<Whether we used the faiss index or the custom workers to perfrom the scan
    bool using_numa; ///<Whether we performed the search using numa or not
    int num_workers_; ///< The number of workers we use to perform the scan
    bool using_adpative_nprobe; ///< Whether we used adaptive nprobe to scan the partitions

    /**
     * @brief Prints all the timing information
     * @param indent Number of spaces to indent for nested outputs.
     */
    void print(int indent = 0) const {
        if (indent == 0) {
            std::cout << "#### Search Timing Information ####" << std::endl;
        }
        std::string indent_str(indent * 2, ' ');

        std::cout << indent_str << "N = " << n_vectors << ", D = " << d << ", K = " << n_clusters
                << ", M = " << num_codebooks << ", code_size = " << code_size << ", using_numa = " << using_numa << ", num_workers = " << num_workers_ << std::endl;
        std::cout << indent_str << "NQ = " << n_queries << ", K = " << k << ", nprobe = " << nprobe
                << ", recall_target = " << recall_target << ", k_factor = " << k_factor << ", using_faiss_index = " << using_faiss_index << std::endl;

        if (parent_info != nullptr) {
            std::cout << indent_str << "## Parent Info ##" << std::endl;
            parent_info->print(indent + 1);
        }

        std::cout << indent_str << "Quantizer search time (us): " << quantizer_search_time_us << std::endl;
        std::cout << indent_str << "Scan PQ time (us): " << scan_pq_time_us << std::endl;
        std::cout << indent_str << "Refine time (us): " << refine_time_us << std::endl;
        std::cout << indent_str << "Partition scan time (us): " << partition_scan_time_us << std::endl;
        std::cout << indent_str << "Partition scan setup time (us): " << partition_scan_setup_time_us << std::endl;
        std::cout << indent_str << "Partition scan search time (us): " << partition_scan_search_time_us << std::endl;
        std::cout << indent_str << "Partition scan post process time (us): " << partition_scan_post_process_time_us << std::endl;

        if(using_numa) {
            std::cout << indent_str << "Total numa preprocessing time (us): " << total_numa_preprocessing_time_us << std::endl;
            std::cout << indent_str << "Total numa adaptive preprocessing time (us): " << total_numa_adaptive_preprocess_time_us << std::endl;
            std::cout << indent_str << "Total numa job distribution time (us): " << total_job_distribute_time_us << std::endl;
            std::cout << indent_str << "Total numa job wait time (us): " << total_result_wait_time_us << std::endl;
            std::cout << indent_str << "Total numa adaptive calculate time (us): " << total_adaptive_calculate_time_us << std::endl;
            std::cout << indent_str << "Total numa shared buffer flush time (us): " << total_shared_flush_time << std::endl;
            std::cout << indent_str << "Total numa postprocessing time (us): " << total_numa_postprocessing_time_us << std::endl;
        }

        std::cout << indent_str << "Average worker job time (us): " << average_worker_job_time_us << std::endl;
        std::cout << indent_str << "Average worker scan time (us): " << average_worker_scan_time_us << std::endl;
        std::cout << indent_str << "Average worker throughput (GB/s): " << average_worker_throughput << std::endl;

        std::cout << indent_str << "Total time (us): " << total_time_us << std::endl;

        // Log the throughput
        if(total_vectors_scanned > 0) {
            float single_vector_size = 1.0 * d * sizeof(float) + sizeof(idx_t);
            float vector_memory_gb = (total_vectors_scanned * single_vector_size)/(1.0 * pow(10, 9));
            float scan_time_sec = get_scan_secs();
            float scan_throughput = vector_memory_gb/scan_time_sec;
            std::cout << indent_str << "Scanning " << total_vectors_scanned << "/" << target_vectors_scanned << " vectors of size " << vector_memory_gb;
            std::cout << " GB workers took " << scan_time_sec << " seconds resulting in throughput of " << scan_throughput << " GB/s" << std::endl;
        }
    }

    float get_scan_secs() const {
        return (1.0 * partition_scan_time_us)/(1.0 * pow(10, 6));
    }

    int64_t get_scan_bytes() const {
        int num_vectors = total_vectors_scanned > 0 ? total_vectors_scanned : target_vectors_scanned;
        int64_t single_vector_size = 1.0 * d * sizeof(float) + sizeof(idx_t);
        return num_vectors * single_vector_size;
    }

    int64_t get_overall_bytes() const {
        int64_t curr_bytes = get_scan_bytes();
        if(parent_info != nullptr) {
            curr_bytes += parent_info->get_overall_bytes();
        }
        return curr_bytes;
    }

    float get_scan_throughput() const {
        float vector_memory_gb = get_overall_bytes()/(1.0 * pow(10, 9));
        float scan_time_sec = (1.0 * total_time_us)/pow(10.0, 6);
        float scan_throughput = vector_memory_gb/scan_time_sec;
        return scan_throughput;
    }

    /**
    * @brief Prints a summary of the timing information
    */
    void print_summary() const {
        std::cout << "Partition scan search time (us): " << partition_scan_search_time_us << std::endl;
        if(using_numa) {
            std::cout << "Total numa job wait time (us): " << total_result_wait_time_us << std::endl;
        }

        std::cout << "Average worker job time (us): " << average_worker_job_time_us << std::endl;
        std::cout << "Average worker scan time (us): " << average_worker_scan_time_us << std::endl;
        std::cout << "Total time (us): " << total_time_us << std::endl;
    }

    /**
    * @brief Copies the partition scan timings to the other SearchTimingInfo
    * @param other The instance of the SearchTimingInfo we want to copy the search timings over to
    */
    void copy_partition_scan_info(shared_ptr<SearchTimingInfo> other) {
        other->scan_pq_time_us = scan_pq_time_us;
        other->refine_time_us = refine_time_us;
        other->partition_scan_time_us = partition_scan_time_us;
        other->total_numa_preprocessing_time_us = total_numa_preprocessing_time_us;
        other->total_numa_adaptive_preprocess_time_us = total_numa_adaptive_preprocess_time_us;
        other->total_job_distribute_time_us = total_job_distribute_time_us;
        other->total_result_wait_time_us = total_result_wait_time_us;
        other->total_numa_postprocessing_time_us = total_numa_postprocessing_time_us;
        other->partition_scan_setup_time_us = partition_scan_setup_time_us;
        other->partition_scan_search_time_us = partition_scan_search_time_us;
        other->average_worker_job_time_us = average_worker_job_time_us;
        other->average_worker_scan_time_us = average_worker_scan_time_us;
        other->using_numa = using_numa;
        other->total_vectors_scanned = total_vectors_scanned;
        other->target_vectors_scanned = target_vectors_scanned;
        other->partition_scan_post_process_time_us = partition_scan_post_process_time_us;
        other->average_worker_throughput = average_worker_throughput;
        other->total_adaptive_calculate_time_us = total_adaptive_calculate_time_us;
        other->total_shared_flush_time = total_shared_flush_time;
    }
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
    /**
     * @brief Prints the timing information.
     */
    void print() const {
        std::cout << "#### Maintenance Timing Information ####" << std::endl;
        std::cout << "Splits: " << n_splits << ", Deletes: " << n_deletes << std::endl;
        std::cout << "Delete time (us): " << delete_time_us << std::endl;
        std::cout << "Delete refine time (us): " << delete_refine_time_us << std::endl;
        std::cout << "Split time (us): " << split_time_us << std::endl;
        std::cout << "Split refine time (us): " << split_refine_time_us << std::endl;
        std::cout << "Total time (us): " << total_time_us << std::endl;
    }
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
            n += v.size(0);
        }
        return n;
    }

    int64_t nlist() const {
        return vectors.size();
    }

    int64_t dim() const {
        return vectors[0].size(1);
    }

    int64_t cluster_size(int64_t i) const {
        return vectors[i].size(0);
    }
};

#endif //COMMON_H
