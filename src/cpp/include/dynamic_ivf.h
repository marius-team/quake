//
// Created by Jason on 8/28/24.
// Prompt for GitHub Copilot:
// - Conform to the Google style guide
// - Use descriptive variable names

#ifndef CPP_UTILS_DYNAMIC_IVF_H
#define CPP_UTILS_DYNAMIC_IVF_H

#include <torch/torch.h>
#include "faiss/index_io.h"
#include <faiss/IVFlib.h>
#include <faiss/IndexRefine.h>
#include <maintenance_policies.h>
#include <dynamic_inverted_list.h>
#include <chrono>

#ifdef QUAKE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#include <faiss/IndexIDMap.h>

#include "dynamic_centroid_store.h"
#include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"
#include "list_scanning.h"

using torch::Tensor;
using std::vector;
using faiss::idx_t;
using std::shared_ptr;
using std::chrono::high_resolution_clock;
using TopKVectors = TypedTopKBuffer<float, int64_t>;

class MaintenancePolicy;
struct MaintenancePolicyParams;
struct MaintenanceTimingInfo;

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

#ifdef QUAKE_NUMA
        if(using_numa) {
            std::cout << indent_str << "Total numa preprocessing time (us): " << total_numa_preprocessing_time_us << std::endl;
            std::cout << indent_str << "Total numa adaptive preprocessing time (us): " << total_numa_adaptive_preprocess_time_us << std::endl;
            std::cout << indent_str << "Total numa job distribution time (us): " << total_job_distribute_time_us << std::endl;
            std::cout << indent_str << "Total numa job wait time (us): " << total_result_wait_time_us << std::endl;
            std::cout << indent_str << "Total numa adaptive calculate time (us): " << total_adaptive_calculate_time_us << std::endl;
            std::cout << indent_str << "Total numa shared buffer flush time (us): " << total_shared_flush_time << std::endl;
            std::cout << indent_str << "Total numa postprocessing time (us): " << total_numa_postprocessing_time_us << std::endl;
        }
#endif

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
#ifdef QUAKE_NUMA
        if(using_numa) {
            std::cout << "Total numa job wait time (us): " << total_result_wait_time_us << std::endl;
        }
#endif

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
 * @brief Class representing a dynamic Inverted File (IVF) index.
 *
 * This class extends the functionalities of FAISS IVF indexes to support dynamic operations such as
 * adding/removing clusters, refining clusters, and hierarchical indexing.
 */
class DynamicIVF_C : public std::enable_shared_from_this<DynamicIVF_C> {
private:
    static constexpr int DEQUEUE_SLEEP_TIME_US = 3;
    static constexpr int CENTROID_SCAN_SLEEP_TIME_US = 10;
    static constexpr int QUERY_BUFFER_REUSE_THRESHOLD = 25;

    // Vector of worker threads
    std::vector<std::thread> scan_workers_;
    int num_scan_workers_;
    bool clusters_distributed_;
    bool workers_initialized_;
    bool log_mode_;

    BuildTimingInfo build_index_on_cpu(Tensor x, Tensor ids);

    void rebuild_index_on_cpu(Tensor x, Tensor ids, int new_nlist);

#ifdef FAISS_ENABLE_GPU
    BuildTimingInfo build_index_on_gpu(Tensor x, Tensor ids);
    void rebuild_index_on_gpu(Tensor x, Tensor ids, int new_nlist);
#endif

    // Cluster Scan Worker and all of the fields it needs to read/write the requests/responses
    std::vector<moodycamel::BlockingConcurrentQueue<int>> jobs_queue_;
    std::vector<float*> curr_queries_per_node_;
    std::vector<TopKVectors> intermediate_results_buffer_;
    std::mutex jobs_completed_mutex_;
    std::condition_variable jobs_completed_condition_;
    std::atomic<int> jobs_remaining_;
    std::atomic<int> workers_ready_;
    std::vector<int> workers_numa_nodes_;

    // Buffer for worker to write the results
    float* all_distances_ptr_;
    int64_t* all_ids_ptr_;
    int* all_counts_ptr_;
    int* all_job_times_ptr_;
    int* all_scan_times_ptr_;
    int* all_vectors_scanned_ptr_;
    float* all_throughputs_ptr_;

    // Buffer storing the job details
    int num_search_vectors_;
    int max_vectors_per_cluster_;
    int num_partitions_to_scan_;
    int64_t* job_search_cluster_id_;
    int* job_query_id_;
    int* job_write_offset_;
    std::vector<TopKVectors> final_result_mergers_;
    int max_jobs_;
    int total_outputs_per_query_;

    // Fields for the centroids scanner
    int* centroids_write_offset_;
    std::condition_variable job_avaliable_condition_;
    std::atomic<int> centroid_query_id_;
    faiss::DynamicCentroidStore* centroid_store_;

    // Fields for the adaptive query execution
    int partition_search_flush_gap_us_;
    int query_latency_target_time_us_;

public:
    shared_ptr<DynamicIVF_C> parent_; ///< Pointer to a higher-level parent index.
    faiss::Index* index_; ///< Pointer to the FAISS index.
    faiss::IndexIVF* ivf_index_; ///< Pointer to the FAISS IVF view of the index.
    shared_ptr<MaintenancePolicy> maintenance_policy_; ///< Maintenance policy for the index.
    int curr_depth_;
    Tensor centroids_; ///< Tensor holding the centroids of the clusters.

    int d_; ///< Dimensionality of the data vectors.
    int num_codebooks_; ///< Number of codebooks used in product quantization.
    int code_size_; ///< Code size for product quantization.
    bool use_refine_; ///< Flag indicating whether the refinement index is used.
    faiss::IndexRefineFlat *refine_index_; ///< Optional refinement index for re-ranking search results. Needed for PQ.
    bool using_numa_optimizations_; ///< Whether numa optimizations are enabled or not
    bool same_core_; ///< Whether different types of workers run on the same core
    bool verify_numa_; ///< Whether we should verify numa or not
    bool use_centroid_workers_; ///< Whether we should our workers to scan centroids
    bool use_adpative_nprobe_; ///< Whether we should use adaptive nprobe when scanning partitions

    /**
     * @brief Constructs a DynamicIVF_C object.
     * @param d Dimensionality of the data vectors.
     * @param nlist Number of clusters (partitions) in the index.
     * @param metric Distance metric to use (e.g., faiss::METRIC_L2).
     * @param num_workers The number of workers we should use for the scan
     * @param m Number of codebooks for PQ (optional).
     * @param code_size Code size for PQ (optional).
     * @param use_numa If numa enabled optimizations should be used (optional)
     * @param verbose If the index should be created in verbose mode (optional)
     * @param verify_numa Whether the worker threads should verify the numa properties (optional)
     * @param same_core If different types of workers should run on the same core (optional)
     * @param use_centroid_workers If we should utilize our workers to scan centroids (optional)
     * @param use_adaptive_nprobe Whether we should adaptive nprobe when scanning partitions (optional)
     */
    DynamicIVF_C(int d, int nlist, int metric, int num_workers=1, int m = -1, int code_size = -1, bool use_numa = false,
        bool verbose = false, bool verify_numa = false, bool same_core = true, bool use_centroid_workers = true, bool use_adaptive_nprobe = false);

    /**
     * @brief Destructor to clean up resources.
     */
    ~DynamicIVF_C();

    /**
    * @brief Used to check if the workers are ready to serve requests
    * @return If this index is ready to serve requests
    */
    bool index_ready();

    /**
    * @brief Set the timeout for the query searches
    * @param max_query_latency_us The end to end query time (in uS)
    * @param flush_gap_time_us The number of intermediate flushes per query (in uS)
    */
    void set_timeout_values(int max_query_latency_us, int flush_gap_time_us) {
        query_latency_target_time_us_ = max_query_latency_us;
        partition_search_flush_gap_us_ = flush_gap_time_us;

        if(parent_ != nullptr) {
            parent_->set_timeout_values(max_query_latency_us, flush_gap_time_us);
        }
    }

    /**
     * @brief Enable numa based optimizations to be used. Note that this only applies to future function calls.
     * @param use_numa_optimization Whether or not we should use numa optimization
     */
    void set_numa_optimization(bool use_numa_optimization) {
        using_numa_optimizations_ = use_numa_optimization;
        if(parent_ != nullptr) {
            parent_->set_numa_optimization(use_numa_optimization);
        }
    }


    /**
     * @brief Resets the workers to the specified number and configuration.
     * @param num_workers Number of workers to reset to.
     * @param same_core If different types of workers should run on the same core (optional)
     * @param use_numa_optimizations If numa optimizations should be used (optional)
     */
    void reset_workers(int num_workers, bool same_core, bool use_numa_optimizations);

    /**
     * @brief Returns the number of clusters (nlist) in the index.
     * @return Number of clusters.
     */
    int nlist() const;

    /**
     * @brief Returns the scan fraction for the index.
     * @return Scan fraction.
     */
    float get_scan_fraction() const;

    /**
     * @brief Returns the total number of vectors indexed.
     * @return Total number of vectors.
     */
    int64_t ntotal() const;

    /**
     * @brief Gets all vectors and their IDs per partition
     * @return A tuple containing the vectors and ids per partition
     */
    std::tuple<std::vector<std::vector<uint8_t>>, std::vector<std::vector<idx_t>>> get_partitions();

    /**
     * @brief Gets all vectors and their IDs from the index.
     * @param use_centroid_store Whether we should use the centroid store to get the centroid vectors and ids (optional)
     * @return A tuple containing vectors and IDs.
     */
    std::tuple<Tensor, Tensor> get_vectors_and_ids(bool use_centroid_store = true);

    /**
     * @brief Gets the ids of the vectors in the index.
     * @return Tensor containing the IDs.
     */
    Tensor get_ids();

    /**
     * @brief Gets the inverted lists from the index.
     * @return Pointer to the inverted lists.
     */
    faiss::DynamicInvertedLists *get_invlists();

    /**
     * @brief Builds the index using the provided centroids and data.
     * @param centroids Tensor containing centroid vectors.
     * @param x Data vectors to index.
     * @param ids IDs corresponding to the data vectors.
     */
    void build_given_centroids(Tensor centroids, Tensor x, Tensor ids);

    /**
     * @brief Retrieves the centroids of the clusters.
     * @return Tensor containing the centroids.
     */
    Tensor centroids();

    /**
     * @brief Adds a new level to the hierarchical index.
     * @param nlist Number of clusters at the new level.
     */
    void add_level(int nlist);

    /**
     * @brief Removes the top level from the hierarchical index.
     */
    void remove_level();

    /**
     * @brief Splits specified partitions into smaller partitions.
     * @param partition_ids Tensor containing IDs of partitions to split.
     * @return A tuple containing the new partition ID, new centroids, and new cluster vectors.
     */
    std::tuple<Tensor, vector<Tensor>, vector<Tensor> > split_partitions(Tensor partition_ids);

    /**
     * @brief Deletes specified partitions from the index.
     * @param partition_ids Tensor containing IDs of partitions to delete.
     * @param reassign If true, reassigns vectors to other partitions.
     */
    void delete_partitions(Tensor partition_ids, bool reassign = false, Tensor reassignments = Tensor());

    /**
     * @brief Merges specified partitions into a single partition.
     * @param partition_ids Tensor containing IDs of partitions to merge.
     * @return Tensor containing the ID of the merged partition.
     */
    Tensor merge_partitions(Tensor partition_ids);

    /**
     * @brief Adds new partitions to the index.
     * @param new_centroids Tensor containing new centroid vectors.
     * @param new_cluster_vectors Vectors to add to the new partitions.
     * @param new_cluster_ids IDs corresponding to the new vectors.
     * @return Tensor containing IDs of the new partitions.
     */
    Tensor add_partitions(Tensor new_centroids, vector<Tensor> new_cluster_vectors, vector<Tensor> new_cluster_ids);

    Tensor get_partition_ids() const;

    /**
     * @brief Refines specified partitions by reassigning vectors and updating centroids.
     * @param partition_ids Tensor containing IDs of partitions to refine.
     * @param refine_nprobe Number of clusters to probe during refinement.
     */
    void refine_partitions(Tensor partition_ids, int refine_nprobe);

    /**
     * @brief Adds new centroids to the parent.
     * @param centroids Tensor containing new centroid vectors.
     * @return Tensor containing new partition IDs.
     */
    Tensor add_centroids(Tensor centroids);

    /**
     * @brief Builds the index using the provided data and IDs.
     * @param x Data vectors to index.
     * @param ids IDs corresponding to the data vectors.
     * @param build_parent A boolean to override the logic on when to build the parent (optional)
     * @param launch_workers A boolean specifying if we should launch workers (optional)
     * @return Timing information for the build process.
     */
    BuildTimingInfo build(Tensor x, Tensor ids, bool build_parent = true, bool launch_workers = true);

    /**
     * @brief Rebuilds the index with a new number of clusters.
     * @param new_nlist New number of clusters (nlist).
     */
    void rebuild(int new_nlist);

    /**
     * @brief Saves the index to a file.
     * @param path File path to save the index.
     */
    void save(const std::string &path);

    /**
     * @brief Loads the index from a file.
     * @param path File path to load the index from.
     * @param launch_workers A boolean indicating whether we should launch workers after loading the index (optional)
     */
    void load(const std::string &path, bool launch_workers = true);

    /**
     * @brief Adds data vectors to the index.
     * @param x Data vectors to add.
     * @param ids IDs corresponding to the data vectors.
     */
    ModifyTimingInfo add(Tensor x, Tensor ids, bool call_maintenance = false);

    /**
     * @brief Removes data vectors from the index based on their IDs.
     * @param ids IDs of the data vectors to remove.
     */
    ModifyTimingInfo remove(Tensor ids, bool call_maintenance = false);

    /**
     * @brief Modify existing vectors in the index.
     * @param x Data vectors to modify.
     * @param ids IDs corresponding to the data vectors.
     */
    ModifyTimingInfo modify(Tensor x, Tensor ids, bool call_maintenance = false);

    /**
     * @brief Conducts maintenance operations on the index if policy decides.
     */
    MaintenanceTimingInfo maintenance();

    /**
     * @brief Retrieves the sizes (number of vectors) of each cluster.
     * @return Tensor containing cluster sizes.
     */
    Tensor get_cluster_sizes() const;

    /**
     * @brief Selects specified clusters and retrieves their data.
     * @param select_ids Tensor containing IDs of clusters to select.
     * @param copy If true, copies the data; otherwise, uses references.
     * @return A tuple containing centroids, cluster vectors, and cluster IDs.
     */
    std::tuple<Tensor, vector<Tensor>, vector<Tensor> > select_clusters(Tensor select_ids, bool copy = true);

    /**
     * @brief Select vectors based on id.
     * @param ids Tensor containing IDs of vectors to select.
     * @return The vectors
     */
    Tensor select_vectors(Tensor ids);

    /**
     * @brief Recomputes centroids for specified clusters.
     * @param ids Tensor containing IDs of clusters to recompute. If empty, recomputes all.
     */
    void recompute_centroids(Tensor ids = Tensor());

    /**
     * @brief Retrieves the cluster IDs for all vectors in the index.
     * @return Tensor containing cluster IDs.
     */
    Tensor get_cluster_ids() const;

    /**
     * @brief Determines the minimal nprobe that achieves the target recall.
     * @param x Query vectors.
     * @param k Number of nearest neighbors.
     * @param recall_target Target recall value.
     * @return Optimal nprobe value.
     */
    int get_nprobe_for_recall_target(Tensor x, int k, float recall_target);

    // /**
    //  * @brief Performs a batched scan over the centroids
    //  * @param x Query vectors.
    //  * @param k Number of nearest neighbors.
    //  * @param ret_ids A pointer to write the return ids
    //  * @param ret_dis A pointer to write the retrun distances
    //  * @param timing_info Pointer to struct to record all of the timings
    //  */
    void search_all_centroids(Tensor x, int k, int64_t* ret_ids, float* ret_dis, shared_ptr<SearchTimingInfo> timing_info);

    /**
     * @brief Scans specified partitions to perform search.
     * @param x Query vectors.
     * @param cluster_ids Cluster IDs to scan.
     * @param cluster_dists Distances to the clusters.
     * @param k Number of nearest neighbors.
     * @param target_latency_us The latency we are allowing for partition scanning (in uS) 
     * @param recall_target Target recall value.
     * @param k_factor Multiplicative factor for k in searches.
     * @param distance_budget The distance budget we have for the search
     * @param use_precomputed Whether to use precomputed in computing recall profile (optional)
     * @return A tuple containing IDs, distances, and timing information.
     */
    std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > scan_partitions(
        Tensor x, Tensor cluster_ids, Tensor cluster_dists, int k, int target_latency_us = -1,
        float recall_target = 0.9f, float k_factor = 4.0f, int distance_budget = -1, bool use_precomputed = true);

    /**
     * @brief Performs a search operation on the index.
     * @param x Query vectors.
     * @param nprobe Number of clusters to probe.
     * @param k Number of nearest neighbors.
     * @param recall_target Target recall value.
     * @param k_factor Multiplicative factor for k in searches.
     * @param use_precomputed Whether to use precomputed in computing recall profile (optional)
     * @return A tuple containing IDs, distances, and timing information.
     */
    std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > search(
        Tensor x, int nprobe, int k, float recall_target = 0.9f, float k_factor = 4.0f, bool use_precomputed = true);

    /**
     * @brief Conducts optimized search for a single query vector.
     * @param x Query vector.
     * @param k Number of nearest neighbors.
     * @param recall_target Target recall value.
     * @return A tuple containing IDs, distances, and timing information.
     */
    std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > search_one(
        Tensor x, int k, float recall_target = 0.9f, int nprobe = -1, float recompute_threshold = .05, bool use_precomputed = true);

    /**
     * @brief Searches the quantizer to find nearest clusters.
     * @param x Query vectors.
     * @param k Number of clusters to find.
     * @param nprobe Number of clusters to probe.
     * @param recall_target Target recall value.
     * @param k_factor Multiplicative factor for k in searches.
     * @param use_gt_to_meet_target If true, uses ground truth to meet recall target.
     * @param use_precomputed Whether to use precomputed in computing recall profile (optional)
     * @return A tuple containing IDs, distances, and timing information.
     */
    std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > search_quantizer(
        Tensor x, int k, int nprobe = 10, float recall_target = 0.9f,
        float k_factor = 4.0f, bool use_gt_to_meet_target = true, bool use_precomputed = true);

    /**
     * @brief Merges selected clusters from another index into the current index.
     * @param selected_clusters IDs of clusters to keep from the current index.
     * @param other The other index to merge from.
     * @param other_selected_clusters IDs of clusters to merge from the other index.
     */
    void selective_merge(Tensor selected_clusters, DynamicIVF_C &other, Tensor other_selected_clusters);

    /**
     * @brief Adds new centroids and reassigns existing vectors accordingly.
     * @param new_centroids Tensor containing new centroid vectors.
     * @param reassign_cluster_ids IDs of clusters whose vectors need reassignment.
     * @param kept_cluster_ids IDs of clusters to keep unchanged.
     * @return Tensor containing modified cluster IDs.
     */
    Tensor add_centroids_and_reassign_existing(Tensor new_centroids, Tensor reassign_cluster_ids,
                                               Tensor kept_cluster_ids);

    /**
     * @brief Refines specified clusters by performing k-means until convergence.
     * @param cluster_ids IDs of clusters to refine.
     */
    void refine_clusters(Tensor cluster_ids, int refinement_iter = 3);

    /**
     * @brief Computes the average quantization error for each cluster.
     * @return Tensor containing quantization errors.
     */
    Tensor compute_quantization_error();

    /**
     * @brief Computes the sum or squared sum of vectors in each cluster.
     * @param squared If true, computes squared sums; otherwise, computes sums.
     * @return Tensor containing sums for each cluster.
     */
    Tensor compute_cluster_sums(bool squared = false) const;

    /**
     * @brief Computes the covariance matrix of a specified cluster.
     * @param cluster_id ID of the cluster.
     * @return Tensor representing the covariance matrix.
     */
    Tensor compute_cluster_covariance(int cluster_id) const;

    /**
     * @brief Computes covariance matrices for all clusters.
     * @return Vector of tensors, each representing a cluster's covariance matrix.
     */
    vector<Tensor> compute_partition_covariances();

    // utility functions for measure inputs to adaptive nprobe model
    Tensor compute_partition_boundary_distances(Tensor query, Tensor partition_ids = Tensor());

    Tensor get_partition_ids_for_vector_ids(Tensor vector_ids);

    float compute_kth_nearest_neighbor_distance(Tensor query, int k);

    Tensor compute_partition_probabilities(Tensor query, int k, Tensor partition_ids = Tensor(), bool use_size = false);

    Tensor compute_partition_intersection_volumes(Tensor query, Tensor partition_ids = Tensor(), int k = 1);

    Tensor compute_partition_distances(Tensor query, Tensor partition_ids = Tensor());

    Tensor compute_partition_density(Tensor partition_ids = Tensor());

    Tensor compute_partition_volume(Tensor partition_ids = Tensor());

    vector<Tensor> compute_partition_variances(Tensor partition_ids = Tensor());

    Tensor get_partition_sizes(Tensor partition_ids = Tensor());

    void set_maintenance_policy_params(MaintenancePolicyParams params);

    vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> > get_split_history();

    vector<Tensor> compute_cluster_covariances() const;

    /**
    * @brief Launches the specified number of cluster scan workers
    * @param only_current_level A boolean indicating we should only launch for this level (optional)
    */
    void launch_cluster_scan_workers(bool only_current_level = false);

    /**
    * @brief Distributes the clusters in the current index across numa nodes
    * @param only_current_level A boolean indicating we should only launch for this level (optional)
    */
    void distribute_clusters(bool only_current_level = false);

#ifdef QUAKE_NUMA
    /**
    * @brief Determines the number of numa nodes to use
    * @returns Returns the number of numa nodes we should use
    */
    int get_num_numa_nodes() { return std::min(numa_max_node() + 1, num_scan_workers_); }

    /** 
    * @brief Worker function to perform scan of the centroids running on a seperate thread
    * @param thread_id The id of the thread
    */
    void centroids_scan_worker_function(int thread_id);

    /** 
    * @brief Worker function to perform scan of a cluster, running on a seperate thread
    * @param thread_id The id of the thread
    */
    void partition_scan_worker_function(int thread_id);
#endif

};

/**
 * @brief Merges two inverted lists from different indexes.
 * @param array_lists1 Inverted lists from the first index.
 * @param selected_clusters1 Selected clusters from the first index.
 * @param array_lists2 Inverted lists from the second index.
 * @param selected_clusters2 Selected clusters from the second index.
 * @return Merged inverted lists.
 */
faiss::InvertedLists *merge_invlists(faiss::ArrayInvertedLists *array_lists1, Tensor selected_clusters1,
                                     faiss::ArrayInvertedLists *array_lists2, Tensor selected_clusters2);

/**
 * @brief Merges two FAISS IVF indexes by combining their inverted lists and centroids.
 * @param index1 The first index.
 * @param selected_clusters1 Selected clusters from the first index.
 * @param index2 The second index.
 * @param selected_clusters2 Selected clusters from the second index.
 * @return New merged index.
 */
faiss::IndexIVF *merge_faiss_ivf(faiss::IndexIVF *index1, Tensor selected_clusters1,
                                 faiss::IndexIVF *index2, Tensor selected_clusters2);

#endif // CPP_UTILS_DYNAMIC_IVF_H