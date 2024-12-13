#include <iostream>
#include <sched.h>
#include <torch/torch.h>
#include <cassert>
#include "simsimd/simsimd.h"
#include <chrono>
#include <vector>
#include <thread>
#include <string>
#include <stdint.h>
#include <stdexcept>
#include <math.h>
#include <mutex>
#include <sys/mman.h>
#include <atomic>
#include <thread>
#include "list_scanning.h"
#include <random>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <random>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <math.h>
#include "faiss/impl/platform_macros.h"
#include "faiss/IVFlib.h"
#include "faiss/index_io.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexFlat.h"
#include "dynamic_ivf.h"
#include "concurrentqueue.h"

#ifdef __linux__
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#endif

typedef unsigned long long longType;
typedef faiss::idx_t idx_t;

using torch::Tensor;
using std::vector;
using BenchTopKClusters = TypedTopKBuffer<double, int>;
using BenchTopKVectors = TypedTopKBuffer<double, idx_t>;
static constexpr bool LOG_MODE = true;

// Global CentroidScanWorker declaration
class CentroidScanWorker;
CentroidScanWorker *global_centroid_worker = nullptr;

// CentroidScanWorker class definition
class CentroidScanWorker {
public:
    CentroidScanWorker(size_t num_threads, int num_centroids, int vector_dimension);

    ~CentroidScanWorker();

    // Initialize the worker threads
    void init(float *centroids, int num_centroids, int vector_dimension);

    // Submit a query to be processed
    void process_query(float *query_vector, double *distances_array);

    // Stop the worker threads
    void shutdown();

    // Get the number of threads
    size_t num_threads() const { return workers.size(); }

private:
    // Worker function
    void worker_thread(int thread_id, int start_idx, int end_idx);

    // Vector of worker threads
    std::vector<std::thread> workers;
    int n_threads;

    // Synchronization
    std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable main_condition;
    bool stop;

    // Query management
    float *current_query_vector;
    float *centroids;
    double *distances_array;
    int num_centroids;
    int vector_dimension;
    std::atomic<int> threads_remaining;

    // Control flags
    std::atomic<int> query_id;
};

CentroidScanWorker::CentroidScanWorker(size_t num_threads, int num_centroids, int vector_dimension)
    : stop(false),
      centroids(nullptr),
      distances_array(nullptr),
      num_centroids(num_centroids),
      vector_dimension(vector_dimension),
      threads_remaining(0),
      query_id(-1),
      n_threads(num_threads) {
}

CentroidScanWorker::~CentroidScanWorker() {
    shutdown();
}

void CentroidScanWorker::init(float *centroids, int num_centroids, int vector_dimension) {
    std::unique_lock<std::mutex> lock(mutex);
    this->centroids = centroids;
    this->num_centroids = num_centroids;
    this->vector_dimension = vector_dimension;

    int chunk_size = (num_centroids + n_threads - 1) / n_threads;

    for (int t = 0; t < n_threads; ++t) {
        int start_idx = std::min(t * chunk_size, num_centroids);
        if(start_idx == num_centroids) {
            break;
        }
        int end_idx = std::min(start_idx + chunk_size, num_centroids);
        workers.emplace_back(&CentroidScanWorker::worker_thread, this, t, start_idx, end_idx);
    }
}

void CentroidScanWorker::process_query(float *query_vector, double *distances_array) { 
    {
        std::unique_lock<std::mutex> lock(mutex);
        // Set the current query and associated data
        this->current_query_vector = query_vector;
        this->distances_array = distances_array;
        threads_remaining.store(workers.size());
        query_id.fetch_add(1);
    }

    // Notify all worker threads that a new query is available
    condition.notify_all();

    // Wait until all threads have processed the query
    {
        std::unique_lock<std::mutex> lock(mutex);
        main_condition.wait(lock, [this]() { return threads_remaining.load() == 0; });
    }
}

void CentroidScanWorker::shutdown() { {
        std::unique_lock<std::mutex> lock(mutex);
        stop = true;
    }

    // Notify all workers to exit
    condition.notify_all();

    // Join all worker threads
    for (std::thread &worker: workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void CentroidScanWorker::worker_thread(int thread_id, int start_idx, int end_idx) {
#ifdef __linux__
    // Determine the NUMA node based on thread_id
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu = thread_id % num_cpus;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }
    
    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }

    // Set the preferred NUMA node for memory allocations
    int numa_node_id = numa_node_of_cpu(sched_getcpu());
    numa_set_preferred(numa_node_id);
    numa_set_strict(1);
#endif

    int last_processed_query_id = -1;

    // Allocate local centroids on the NUMA node
    int64_t n_local_centroids = end_idx - start_idx;
    int64_t local_centroids_size = n_local_centroids * vector_dimension * sizeof(float);
    int64_t local_vector_size = vector_dimension * sizeof(float);
    int64_t local_distances_size = n_local_centroids * sizeof(double);

    // Log the range of the current thread
    if constexpr(LOG_MODE) {
        std::stringstream centroid_range_stream;
        centroid_range_stream << "Thread " << thread_id << " is responsible for centroids " << start_idx << " to " << end_idx << std::endl;
        std::cout << centroid_range_stream.str();
    }

    // Allocate the buffers
#ifdef __linux__
    float* local_query_vector = (float*) numa_alloc_onnode(local_vector_size, numa_node_id);
    if(local_query_vector == NULL) {
        std::string err_msg = "Thread " + std::to_string(thread_id) +  " unable to allocate " + std::to_string(local_vector_size) + " bytes for query vector on " + std::to_string(numa_node_id);
        throw std::runtime_error(err_msg);
    }

    double* local_distances = (double*) numa_alloc_onnode(local_distances_size, numa_node_id);
    if(local_distances == NULL) {
        std::string err_msg = "Thread " + std::to_string(thread_id) +  " unable to allocate " + std::to_string(local_distances_size) + " bytes for distances on " + std::to_string(numa_node_id);
        throw std::runtime_error(err_msg);
    }

    float* local_centroids = (float*) numa_alloc_onnode(local_centroids_size, numa_node_id);
    if(local_centroids == NULL) {
        std::string err_msg = "Thread " + std::to_string(thread_id) +  " unable to allocate " + std::to_string(local_centroids_size) + " bytes for centroids on " + std::to_string(numa_node_id);
        throw std::runtime_error(err_msg);
    }
    
#else
    float* local_query_vector = (float*) aligned_alloc(64, local_vector_size);
    float* local_centroids = (float*) aligned_alloc(64, local_centroids_size);
    double* local_distances = (double*) aligned_alloc(64, local_distances_size);
#endif

    // Mem lock the buffers
    mlock(local_centroids, local_centroids_size);
    mlock(local_query_vector, local_vector_size);
    mlock(local_distances, local_distances_size);

    // Copy centroids data to local centroids
    std::memcpy(local_centroids, centroids + start_idx * vector_dimension, local_centroids_size);
    std::memset(local_distances, 0, local_distances_size);
    std::memset(local_query_vector, 0, local_vector_size);

    while (true) {
        int current_query_id;

        // Wait for a new query
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this, &last_processed_query_id]() {
                return query_id.load() != last_processed_query_id || stop;
            });

            if (stop) {
#ifdef __linux__
                numa_free(local_centroids, local_centroids_size);
                numa_free(local_query_vector, local_vector_size);
                numa_free(local_distances, local_distances_size);
#else
                free(local_centroids);
                free(local_query_vector);
                free(local_distances);
#endif
                return;
            }

            // Copy the pointers to local variables while holding the lock
            std::memcpy(local_query_vector, current_query_vector, local_vector_size);
            current_query_id = query_id.load();
        }

        // Process the assigned chunk of centroids for the current query
#pragma unroll
        for (int i = 0; i < n_local_centroids; ++i) {
            simsimd_dot_f32(local_centroids + i * vector_dimension,
                            local_query_vector,
                            vector_dimension,
                            local_distances + i);
            
            if constexpr(LOG_MODE) {
                std::stringstream centroid_distance_log;
                centroid_distance_log << "Thread " << thread_id << " calculated distance of " << local_distances[i] << " to local centroid " << i << std::endl;
                std::cout << centroid_distance_log.str();
            }
        }

        // copy the local distances to the global distances
        std::memcpy(distances_array + start_idx, local_distances, n_local_centroids * sizeof(double));

        // Decrement threads_remaining
        last_processed_query_id = current_query_id;
        int remaining = threads_remaining.fetch_sub(1);
        if (remaining == 1) {
            // This was the last thread to finish processing
            std::unique_lock<std::mutex> lock(mutex);
            main_condition.notify_one();
        }
    }
}

// Global CentroidScanWorker declaration
class ClusterScanWorker;
ClusterScanWorker* global_cluster_worker = nullptr;

// CentroidScanWorker class definition
class ClusterScanWorker {
public:
    ClusterScanWorker(int num_threads, int partitions_to_process, int vectors_to_return, int vector_dimension);

    ~ClusterScanWorker();

    // Perform a shutdown
    void shutdown();

    // Initialize the worker threads
    void init(std::vector<std::vector<uint8_t>> all_codes, std::vector<std::vector<idx_t>> all_ids);

    // Submit a query to be processed
    void process_query(float* query_vector, int* clusters_to_process, idx_t* ids_array, double* dists_array);

    // Get the number of threads
    size_t num_threads() const { return workers.size(); }

private:
    // Worker function
    void debug_worker_thread(int thread_id);
    void worker_thread(int thread_id);

    // Vector of worker threads
    std::vector<std::thread> workers;
    int n_threads;
    int get_num_numa_nodes() { return std::min(numa_max_node() + 1, n_threads); }
    
    // Buffers holding various shared objects
    std::vector<moodycamel::ConcurrentQueue<int>> jobs_queue_;
    std::vector<float*> curr_queries_per_node_;
    BenchTopKVectors intermediate_results_merger_;

    // All of the intermediates we need
    double* all_distances_ptr_;
    idx_t* all_ids_ptr_;
    int* all_counts_ptr_;
    int* job_search_cluster_id_;

    // Cluster Management
    std::unordered_map<int, int> cluster_numa_node_;
    std::unordered_map<int, int> cluster_num_vectors_;
    std::unordered_map<int, float*> cluster_vectors_;
    std::unordered_map<int, idx_t*> cluster_ids_;

    // Query Management
    int num_partitions_to_scan_;
    int total_vectors_to_return_;
    int vector_dimension_;

    // Synchronization
    std::atomic<int> jobs_remaining_;
    std::mutex jobs_completed_mutex_;
    std::condition_variable jobs_completed_condition_;
};

ClusterScanWorker::ClusterScanWorker(int num_threads, int partitions_to_process, int vectors_to_return, int vector_dimension) 
    : n_threads(num_threads), 
    num_partitions_to_scan_(partitions_to_process), 
    total_vectors_to_return_(vectors_to_return),
    vector_dimension_(vector_dimension),
    jobs_remaining_(0),
    intermediate_results_merger_(vectors_to_return, false) {

}


ClusterScanWorker::~ClusterScanWorker() {
    shutdown();
}

void ClusterScanWorker::shutdown() {
    int num_clusters = cluster_numa_node_.size();
    for(int i = 0; i < num_clusters; i++) {
        int num_vectors = cluster_num_vectors_[i];
        numa_free(cluster_vectors_[i], num_vectors * vector_dimension_ * sizeof(float));
        numa_free(cluster_ids_[i], num_vectors * sizeof(idx_t));
    }

    int num_numa_nodes = this->get_num_numa_nodes();
    for(int i = 0; i < num_numa_nodes; i++) {
        numa_free(curr_queries_per_node_[i], vector_dimension_ * sizeof(float));
    }

    for(int i = 0; i < num_numa_nodes; i++) {
        for(int j = 0; j < n_threads; j++) {
            jobs_queue_[i].enqueue(-1);
        }
    }

    for(std::thread& curr_thread : workers) {
        if(curr_thread.joinable()) {
            curr_thread.join();
        }
    }
}

void ClusterScanWorker::init(std::vector<std::vector<uint8_t>> all_codes, std::vector<std::vector<idx_t>> all_ids) {
    // Create a queue per numa node
    int num_numa_nodes = this->get_num_numa_nodes();
    for(int i = 0; i < num_numa_nodes; i++) {
        jobs_queue_.emplace_back();
    }

    int num_clusters = all_ids.size();
    for(int i = 0; i < num_clusters; i++) {
        int curr_numa_node = i % num_numa_nodes;
        cluster_numa_node_[i] = curr_numa_node;

        // Load the vectors for this cluster
        std::vector<uint8_t>& curr_codes = all_codes[i];
        std::vector<idx_t>& curr_ids = all_ids[i];
        int num_vectors = curr_ids.size();
        cluster_num_vectors_[i] = num_vectors;

        // If there are zero vectors then skip allocating buffer for the cluster
        if(num_vectors == 0) {
            if(LOG_MODE) {
                std::cout << "Skipping processing of cluster " << i << " with " << num_vectors << " vectors" << std::endl;
            }
            continue;
        }

        // Alloc the codes buffer
        int codes_size = curr_codes.size() * sizeof(uint8_t);
        float* codes_buffer = reinterpret_cast<float*>(numa_alloc_onnode(codes_size, curr_numa_node));
        if(codes_buffer == NULL) {
            throw std::runtime_error("Unable to alloc the codes on a numa node");
        }
        std::memcpy(codes_buffer, curr_codes.data(), codes_size);
        cluster_vectors_[i] = codes_buffer;

        // Alloc the ids buffer
        int ids_size = curr_ids.size() * sizeof(idx_t);
        idx_t* ids_buffer = reinterpret_cast<idx_t*>(numa_alloc_onnode(ids_size, curr_numa_node));
        if(ids_buffer == NULL) {
            throw std::runtime_error("Unable to alloc the ids on a numa node");
        }
        std::memcpy(ids_buffer, curr_ids.data(), ids_size);
        cluster_ids_[i] = ids_buffer;

        // Log the mapping
        if(LOG_MODE) {
            std::cout << "Mapped cluster " << i << " with " << num_vectors << " vectors mapped to numa node " << curr_numa_node << std::endl;
        }
    }

    // Create the intermediate buffers
    curr_queries_per_node_.reserve(num_numa_nodes);
    int vector_buffer_size = vector_dimension_ * sizeof(float);
    for(int i = 0; i < num_numa_nodes; i++) {
        curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(vector_buffer_size, i));
        if(curr_queries_per_node_[i] == NULL) {
            throw std::runtime_error("Unable to create query vector buffer");
        }
        std::memset(curr_queries_per_node_[i], 0, vector_buffer_size);
    }

    // Create the intermediate results
    int num_intermediate_results = num_partitions_to_scan_ * total_vectors_to_return_;
    all_distances_ptr_ = new double[num_intermediate_results];
    all_ids_ptr_ = new idx_t[num_intermediate_results];
    all_counts_ptr_ = new int[num_partitions_to_scan_];
    job_search_cluster_id_ = new int[num_partitions_to_scan_];

    // Start up the threads
    for(int i = 0; i < n_threads; i++) {
        if(LOG_MODE) {
            workers.emplace_back(&ClusterScanWorker::debug_worker_thread, this, i);
        } else {
            workers.emplace_back(&ClusterScanWorker::worker_thread, this, i);
        }
    }
}

void ClusterScanWorker::process_query(float* query_vector, int* clusters_to_process, idx_t* ids_array, double* dists_array) {
    // Copy the vector to the entire numa node
    int num_numa_nodes = this->get_num_numa_nodes();
    for(int i = 0; i < num_numa_nodes; i++) {
        std::memcpy(curr_queries_per_node_[i], query_vector, vector_dimension_ * sizeof(float));
    }

    // Submit the job to the queues
    jobs_remaining_.store(num_partitions_to_scan_);
    int num_skipped_jobs = 0;
    for(int i = 0; i < num_partitions_to_scan_; i++) {
        int curr_cluster = clusters_to_process[i];
        if(cluster_num_vectors_[curr_cluster] > 0) {
            int cluster_numa_node = cluster_numa_node_[curr_cluster];
            job_search_cluster_id_[i] = curr_cluster;
            jobs_queue_[cluster_numa_node].enqueue(i);
        } else {
            all_counts_ptr_[i] = 0;
            num_skipped_jobs += 1;
        }
    }
    jobs_remaining_.fetch_sub(num_skipped_jobs);

    // Wait for the job to completed
    {
        std::unique_lock<std::mutex> lock(jobs_completed_mutex_);
        jobs_completed_condition_.wait(lock, [this]() { return jobs_remaining_.load() == 0; });
    }

    // Merge the results from the workers
    intermediate_results_merger_.reset();
    for(int i = 0; i < num_partitions_to_scan_; i++) {
        // Get the vectors from the current cluster
        int num_vectors = all_counts_ptr_[i];
        idx_t* cluster_ids = all_ids_ptr_ + i * total_vectors_to_return_;
        double* cluster_dists = all_distances_ptr_ + i * total_vectors_to_return_;
        for(int j = 0; j < num_vectors; j++) {
            intermediate_results_merger_.add(cluster_dists[j], cluster_ids[j]);
        }
    }

    // Write out the combined results
    std::vector<idx_t> query_top_ids = intermediate_results_merger_.get_topk_indices();
    std::vector<double> query_top_dists = intermediate_results_merger_.get_topk();
    int num_final_results = std::min((int) query_top_ids.size(), total_vectors_to_return_);
    std::memcpy(dists_array, query_top_dists.data(), num_final_results * sizeof(double));
    std::memcpy(ids_array, query_top_ids.data(), num_final_results * sizeof(idx_t));
}

void ClusterScanWorker::worker_thread(int thread_id) {
    // Determine the NUMA node based on thread_id
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu = thread_id % num_cpus;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }
    
    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }

    // Set the preferred NUMA node for memory allocations
    int numa_node_id = numa_node_of_cpu(sched_getcpu());
    numa_set_preferred(numa_node_id);
    numa_set_strict(1);

    // Load the buffers
    moodycamel::ConcurrentQueue<int>& requests_queue = jobs_queue_[numa_node_id];
    BenchTopKVectors curr_worker_cluster(total_vectors_to_return_, false);
    int job_id;

    while(true) {
        // Block till we have a request to process
        if(!requests_queue.try_dequeue(job_id)) {
            continue;
        }

        // See if this is a stop worker request
        if(job_id == -1) {
            break;
        }

        // Load the job details
        float* query_vector = curr_queries_per_node_[numa_node_id];
        int curr_cluster_id = job_search_cluster_id_[job_id];

        // Load the cluster details
        int num_vectors_in_cluster = cluster_num_vectors_[curr_cluster_id];
        float* cluster_vector = cluster_vectors_[curr_cluster_id];
        idx_t* cluster_ids = cluster_ids_[curr_cluster_id];

        // Perform the actual scan
        curr_worker_cluster.reset();
        double distance;

        #pragma unroll
        for (int i = 0; i < num_vectors_in_cluster; i++) {
            simsimd_dot_f32(cluster_vector + i * vector_dimension_, query_vector, vector_dimension_, &distance);
            curr_worker_cluster.add(distance, cluster_ids[i]);
        }

        // Write the result
        std::vector<idx_t> overall_top_ids = curr_worker_cluster.get_topk_indices();
        std::vector<double> overall_top_dists = curr_worker_cluster.get_topk();
        int num_results = std::min((int) overall_top_ids.size(), total_vectors_to_return_);
        all_counts_ptr_[job_id] = num_results;
        int write_offset = job_id * total_vectors_to_return_;
        std::memcpy(all_distances_ptr_ + write_offset, overall_top_dists.data(), num_results * sizeof(double));
        std::memcpy(all_ids_ptr_ + write_offset, overall_top_ids.data(), num_results * sizeof(idx_t));

        // Mark the job as finished
        int jobs_left = jobs_remaining_.fetch_sub(1);
        if (jobs_left <= 1) {
            // This was the last thread to finish processing
            {
                std::unique_lock<std::mutex> lock(jobs_completed_mutex_);
                jobs_completed_condition_.notify_one();
            }
        }
    }
}

void ClusterScanWorker::debug_worker_thread(int thread_id) {
    // Determine the NUMA node based on thread_id
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu = thread_id % num_cpus;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }
    
    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }

    // Set the preferred NUMA node for memory allocations
    int numa_node_id = numa_node_of_cpu(sched_getcpu());
    numa_set_preferred(numa_node_id);
    numa_set_strict(1);
    std::cout << "Worker " << thread_id << " mapped to cpu " << worker_cpu << "/" << num_cpus << " bind to numa node " << numa_node_id << std::endl;

    // Load the buffers
    moodycamel::ConcurrentQueue<int>& requests_queue = jobs_queue_[numa_node_id];
    BenchTopKVectors curr_worker_cluster(total_vectors_to_return_, false);
    int job_id;

    while(true) {
        // Block till we have a request to process
        if(!requests_queue.try_dequeue(job_id)) {
            continue;
        }

        // See if this is a stop worker request
        if(job_id == -1) {
            break;
        }

        // Load the job details
        float* query_vector = curr_queries_per_node_[numa_node_id];
        int curr_cluster_id = job_search_cluster_id_[job_id];

        // Load the cluster details
        int num_vectors_in_cluster = cluster_num_vectors_[curr_cluster_id];
        float* cluster_vector = cluster_vectors_[curr_cluster_id];
        idx_t* cluster_ids = cluster_ids_[curr_cluster_id];
        std::cout << "Job " << job_id << " processing cluster " << curr_cluster_id << " with " << num_vectors_in_cluster << " vectors" << std::endl;

        // Verify the job is being done on the valid numa node
        int curr_worker_numa_node = numa_node_of_cpu(sched_getcpu());
        int cluster_expected_node = cluster_numa_node_[curr_cluster_id];
        if(curr_worker_numa_node != cluster_expected_node) {
            throw std::runtime_error("Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but cluster on node " + std::to_string(cluster_expected_node));
        }

        int codes_numa_node = -1;
        get_mempolicy(&codes_numa_node, NULL, 0, (void*) cluster_vector, MPOL_F_NODE | MPOL_F_ADDR);
        if(curr_worker_numa_node != codes_numa_node) {
            throw std::runtime_error("Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but codes on node " + std::to_string(codes_numa_node));
        }

        int ids_numa_node = -1;
        get_mempolicy(&ids_numa_node, NULL, 0, (void*) cluster_ids, MPOL_F_NODE | MPOL_F_ADDR);
        if(curr_worker_numa_node != ids_numa_node) {
            throw std::runtime_error("Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but ids on node " + std::to_string(ids_numa_node));
        }

        // Perform the actual scan
        curr_worker_cluster.reset();
        double distance;

        auto worker_scan_start_time = std::chrono::high_resolution_clock::now();
        #pragma unroll
        for (int i = 0; i < num_vectors_in_cluster; i++) {
            simsimd_dot_f32(cluster_vector + i * vector_dimension_, query_vector, vector_dimension_, &distance);
            curr_worker_cluster.add(distance, cluster_ids[i]);
        }
        auto worker_scan_end_time = std::chrono::high_resolution_clock::now();


        // Write the result
        std::vector<idx_t> overall_top_ids = curr_worker_cluster.get_topk_indices();
        std::vector<double> overall_top_dists = curr_worker_cluster.get_topk();
        int num_results = std::min((int) overall_top_ids.size(), total_vectors_to_return_);
        all_counts_ptr_[job_id] = num_results;
        int write_offset = job_id * total_vectors_to_return_;
        std::memcpy(all_distances_ptr_ + write_offset, overall_top_dists.data(), num_results * sizeof(double));
        std::memcpy(all_ids_ptr_ + write_offset, overall_top_ids.data(), num_results * sizeof(idx_t));
        
        std::cout << "Worker " << thread_id << " got " << num_results << " vectors for cluster " << curr_cluster_id << ": ";
        for(int i = 0; i < num_results; i++) {
            std::cout << "(" << overall_top_ids[i] << "," << overall_top_dists[i] << ") ";
        }
        std::cout << std::endl;

        int scan_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_scan_end_time - worker_scan_start_time).count();
        float memory_size_gb = (1.0 * num_vectors_in_cluster * (vector_dimension_ * sizeof(float) + sizeof(idx_t)))/(1.0 * pow(10, 9));
        float scan_time_sec = scan_time/(1.0 * pow(10, 6));
        std::cout << "Scanning cluster with " << num_vectors_in_cluster << " vectors of size " << memory_size_gb << " took " << scan_time << " uS resulting in throughput of " << memory_size_gb/scan_time_sec << " GB/s" << std::endl;

        // Mark the job as finished
        int jobs_left = jobs_remaining_.fetch_sub(1);
        if (jobs_left <= 1) {
            // This was the last thread to finish processing
            {
                std::unique_lock<std::mutex> lock(jobs_completed_mutex_);
                jobs_completed_condition_.notify_one();
            }
        }
    }
}

float measure_openmp_time(float *centroids, int num_centroids, int vector_dimension, float *search_vectors, int num_search_vectors, 
double *distances_array, std::vector<std::vector<uint8_t>> all_codes, std::vector<std::vector<idx_t>> all_ids, int num_threads, int clusters_to_process,  
int total_return_vectors, idx_t* ids_array, double* top_k_dist_array) {
    // Initialize necessary variables
    longType ids_per_query = clusters_to_process * total_return_vectors;
    longType num_output_ids = ids_per_query * num_search_vectors;
    idx_t* all_output_ids = new idx_t[num_output_ids];
    std::memset(all_output_ids, 0, num_output_ids * sizeof(idx_t));

    double* all_output_distances = new double[num_output_ids];
    std::memset(all_output_distances, 0, num_output_ids * sizeof(double));

    int* all_output_counts = new int[clusters_to_process]; // The number of vectors written to all_outputs ids and distances by this cluster
    std::memset(all_output_counts, 0, clusters_to_process * sizeof(int));

    // Create the top k buffers
    BenchTopKClusters centroids_top_k(clusters_to_process, false);
    BenchTopKVectors all_clusters_top_k(total_return_vectors, false);
    BenchTopKVectors* buffer_per_worker[num_threads];
#pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < num_threads; i++) {
        buffer_per_worker[i] = new BenchTopKVectors(total_return_vectors, false);
    }

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_search_vectors; ++i) {
        centroids_top_k.reset();
        float* curr_query_vector = search_vectors + i * vector_dimension;
        double* curr_distances_array = distances_array + i * num_centroids;

        // Get the distances 
#pragma omp parallel for num_threads(num_threads)
        for (int j = 0; j < num_centroids; ++j) {
            double dist_result;
            simsimd_dot_f32(centroids + j * vector_dimension, curr_query_vector, vector_dimension, &dist_result);
            curr_distances_array[j] = dist_result;
        }

        // Get the top k clusters
        centroids_top_k.reset(); 
        for(int j = 0; j < num_centroids; j++) {
            if constexpr(LOG_MODE) {
                std::cout << " Openmp Query " << i << " read distance of " << curr_distances_array[j] << " for centroid " << j << std::endl;
            }    
            centroids_top_k.add(curr_distances_array[j], j);
        }

        std::vector<int> clusters_to_search = centroids_top_k.get_topk_indices();
        int num_clusters_for_query = std::min((int) clusters_to_search.size(), clusters_to_process);
        int* search_clusters_ptr = clusters_to_search.data();
        if constexpr(LOG_MODE) {
            std::cout << "Opemp Query " << i << " got search clusters of size " << num_clusters_for_query << " with values: " << clusters_to_search << std::endl;
        }

        // Set the count for the remaining clusters to zero
        for(int j = num_clusters_for_query; j < clusters_to_process; j++) {
            all_output_counts[j] = 0;
        }

        // Perform search with a sepearte thread processing each cluster
        idx_t* all_id_writer = all_output_ids + i * ids_per_query;
        double* all_dist_writer = all_output_distances + i * ids_per_query;

#pragma omp parallel for num_threads(num_threads)
        for(int j = 0; j < num_clusters_for_query; j++) {
            // Get the vectors and ids for this cluster
            int cluster_id = search_clusters_ptr[j];
            float* cluster_vectors = reinterpret_cast<float*>(all_codes[cluster_id].data());
            std::vector<idx_t> cluster_vectors_ids = all_ids[cluster_id];
            int num_vectors_in_cluster = cluster_vectors_ids.size();
            idx_t* vector_ids_arr = cluster_vectors_ids.data();

            // Record the top k for this cluster
            BenchTopKVectors* cluster_top_k = buffer_per_worker[omp_get_thread_num()];
            cluster_top_k->reset();

            double curr_dist;
            for(int k = 0; k < num_vectors_in_cluster; k++) {
                simsimd_dot_f32(cluster_vectors + k * vector_dimension, curr_query_vector, vector_dimension, &curr_dist);
                cluster_top_k->add(curr_dist, vector_ids_arr[k]);
            }

            // Write the top ids for this cluster to the buffer
            std::vector<idx_t> cluster_top_ids = cluster_top_k->get_topk_indices();
            int ids_to_write = std::min((int) cluster_top_ids.size(), total_return_vectors);
            all_output_counts[j] = ids_to_write;
            std::memcpy(all_id_writer + j * total_return_vectors, cluster_top_ids.data(), ids_to_write * sizeof(idx_t));

            // Also write the distances
            std::vector<double> cluster_top_dists = cluster_top_k->get_topk();
            std::memcpy(all_dist_writer + j * total_return_vectors, cluster_top_dists.data(), ids_to_write * sizeof(double));

            if constexpr(LOG_MODE) {
                #pragma critical
                {
                    std::cout << "Opemp Query " << i << ", Cluster Id " << cluster_id << " has top of: ";
                    for(int k = 0; k < ids_to_write; k++) {
                        std::cout << "(" << cluster_top_ids[k] << "," << cluster_top_dists[k] << ") ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        // Merge the results from the clusters
        all_clusters_top_k.reset();
        for(int j = 0; j < clusters_to_process; j++) {
            // Get the vectors from the current cluster
            idx_t* cluster_ids = all_id_writer + j * total_return_vectors;
            double* cluster_dists = all_dist_writer + j * total_return_vectors;
            int num_vectors = all_output_counts[j];
            if constexpr(LOG_MODE) {
                std::cout << "Openmp Query " << i << " got " << num_vectors << " vectors for cluster " << search_clusters_ptr[j] << std::endl;
            }

            for(int k = 0; k < num_vectors; k++) {
                all_clusters_top_k.add(cluster_dists[k], cluster_ids[k]);
            }
        }

        // Write the top ids for this cluster to the output
        std::vector<idx_t> overall_top_ids = all_clusters_top_k.get_topk_indices();
        int overall_ids_to_write = std::min((int) overall_top_ids.size(), total_return_vectors);
        std::memcpy(ids_array + i * total_return_vectors, overall_top_ids.data(), overall_ids_to_write * sizeof(idx_t));

        // Also write the distances
        std::vector<double> overall_top_dists = all_clusters_top_k.get_topk();
        std::memcpy(top_k_dist_array + i * total_return_vectors, overall_top_dists.data(), overall_ids_to_write * sizeof(double));

        // Log the final output
        if constexpr(LOG_MODE) {
            std::cout << "Opemp Query " << i << " has final top of: ";
            for(int k = 0; k < overall_ids_to_write; k++) {
                std::cout << "(" << overall_top_ids[k] << "," << overall_top_dists[k] << ") ";
            }
            std::cout << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    delete[] all_output_ids;
    delete[] all_output_distances;
    delete[] all_output_counts;

    // Also free the per thread buffer
    for(int i = 0; i < num_threads; i++) {
        delete buffer_per_worker[i];
    }

    return duration / 1000.0;
}

float measure_worker_time(int num_centroids, int vector_dimension, float *search_vectors, int num_search_vectors, double *distances_array, 
int clusters_to_process, int total_return_vectors, idx_t* ids_array, double* top_k_dist_array) {

    // Initialize necessary variables
    BenchTopKClusters centroids_top_k(clusters_to_process, false);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_search_vectors; ++i) {
        if constexpr(LOG_MODE) {
            std::cout << "Worker starting processing for query " << i << std::endl;
        }

        float* curr_query_vector = search_vectors + i * vector_dimension;
        double* curr_distances_array = distances_array + i * num_centroids;

        // Process the query using the worker
        global_centroid_worker->process_query(curr_query_vector, curr_distances_array);

        // Get the top k clusters
        centroids_top_k.reset(); 
        for(int j = 0; j < num_centroids; j++) {
            if constexpr(LOG_MODE) {
                std::cout << " Worker Query " << i << " read distance of " << curr_distances_array[j] << " for centroid " << j << std::endl;
            }    
            centroids_top_k.add(curr_distances_array[j], j);
        }

        std::vector<int> clusters_to_search = centroids_top_k.get_topk_indices();
        int num_clusters = std::min((int) clusters_to_search.size(), clusters_to_process);
        if constexpr(LOG_MODE) {
            std::cout << "Worker Query " << i << " got search clusters of size " << num_clusters << " with values: " << clusters_to_search << std::endl;
        }

        // Probe the determined clusters using the other worker
        idx_t* curr_query_ids = ids_array + i * total_return_vectors;
        double* curr_query_dists = top_k_dist_array + i * total_return_vectors;
        global_cluster_worker->process_query(curr_query_vector, clusters_to_search.data(), curr_query_ids, curr_query_dists);

        if constexpr(LOG_MODE) {
            std::cout << "Worker Query " << i << " has final top of: ";
            for(int j = 0; j < total_return_vectors; j++) {
                std::cout << "(" << curr_query_ids[j] << "," << curr_query_dists[j] << ") ";
            }
            std::cout << std::endl;
        }
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    return duration/1000.0;
}

void run_experiment(float* centroids_array, std::vector<std::vector<uint8_t>> cluster_vectors, std::vector<std::vector<idx_t>> cluster_ids, 
int num_centroids, int vector_dimension, int num_search_vectors, int clusters_to_process,  int total_return_vectors, int num_threads, 
std::ofstream& output_file, int iterations = 5, int iterations_to_ignore = 1) {

    // Setup the global worker
    auto start_time = std::chrono::high_resolution_clock::now();
    global_centroid_worker = new CentroidScanWorker(num_threads, num_centroids, vector_dimension);
    global_centroid_worker->init(centroids_array, num_centroids, vector_dimension);
    global_cluster_worker = new ClusterScanWorker(num_threads, clusters_to_process, total_return_vectors, vector_dimension);
    global_cluster_worker->init(cluster_vectors, cluster_ids);
    auto end_time = std::chrono::high_resolution_clock::now();
    double worker_init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Initialized the global workers " << std::endl;

    // Also create the search vectors
    Tensor search_vectors_tensor = torch::randn({num_search_vectors, vector_dimension}).contiguous();
    longType num_search_elems = num_search_vectors * vector_dimension;
    float* search_vectors_array = (float*) aligned_alloc(64, num_search_elems * sizeof(float));
    std::memcpy(search_vectors_array, search_vectors_tensor.data_ptr<float>(), num_search_elems * sizeof(float));
    std::cout << "Finished creating the search vectors" << std::endl;

    // Allocate distances_array
    longType num_distances = num_search_vectors * num_centroids;
    double* distances_array = new double[num_distances];
    double* distances_array_omp = new double[num_distances];

    // Allocate the result vectors array
    longType total_ids = num_search_vectors * total_return_vectors;
    idx_t* ids_array_omp = new idx_t[total_ids];
    double* top_k_dist_array_omp = new double[total_ids];
    idx_t* ids_array = new idx_t[total_ids];
    double* top_k_dist_array = new double[total_ids];

    // Zero initialize all of the result buffers
    std::memset(distances_array, 0, sizeof(double) * num_distances);
    std::memset(distances_array_omp, 0, sizeof(double) * num_distances);
    std::memset(ids_array_omp, 0, sizeof(idx_t) * total_ids);
    std::memset(top_k_dist_array_omp, 0, sizeof(double) * total_ids);
    std::memset(ids_array, 0, sizeof(idx_t) * total_ids);
    std::memset(top_k_dist_array, 0, sizeof(double) * total_ids);

    // Get the total cluster size
    longType num_centroids_bytes = num_centroids * vector_dimension * sizeof(float);
    longType total_cluster_bytes = 0;
    longType total_vectors = 0;
    for(int i = 0; i < num_centroids; i++) {
        int num_vectors = cluster_ids[i].size();
        longType vectors_size = sizeof(uint8_t) * cluster_vectors[i].size();
        longType ids_size = sizeof(idx_t) * cluster_ids[i].size();

        total_cluster_bytes += vectors_size + ids_size;
        total_vectors += num_vectors;
    }

    longType expected_bytes_scan = num_search_vectors * (num_centroids_bytes + total_cluster_bytes);
    float bytes_scanned_gb = (1.0 * expected_bytes_scan)/(pow(10, 9));

    float centroid_throughput_avg = 0.0; float openmp_throughput_avg = 0.0; float counts = 0.0;
    for (int i = 0; i < iterations; ++i) {
        bool run_omp_first = (i % 2 == 0); // Alternate execution order

        float centroid_throughput = -1.0; float openmp_throughput = -1.0; 
        if (run_omp_first) {
            // Cold start for OpenMP
            float openmp_cold_time_ms = measure_openmp_time(centroids_array,
                                                            num_centroids,
                                                            vector_dimension,
                                                            search_vectors_array,
                                                            num_search_vectors,
                                                            distances_array_omp,
                                                            cluster_vectors,
                                                            cluster_ids,
                                                            num_threads, 
                                                            clusters_to_process, 
                                                            total_return_vectors,
                                                            ids_array_omp, 
                                                            top_k_dist_array_omp);
            openmp_throughput = bytes_scanned_gb/(openmp_cold_time_ms/1000.0);

            // Hot start for Worker
            float worker_hot_time_ms = measure_worker_time(num_centroids,
                                                           vector_dimension,
                                                           search_vectors_array,
                                                           num_search_vectors,
                                                           distances_array,
                                                           clusters_to_process,
                                                           total_return_vectors,
                                                           ids_array, 
                                                           top_k_dist_array);
            centroid_throughput = bytes_scanned_gb/(worker_hot_time_ms/1000.0);
                                                        

            std::cout << "Iteration " << i + 1 << ": Cold OpenMP Time = " << openmp_cold_time_ms << " ms having throughput " << openmp_throughput << " GB/s" << std::endl;
            std::cout << "Iteration " << i + 1 << ": Hot Worker Time = " << worker_hot_time_ms << " ms having throughput " << centroid_throughput << " GB/s" << std::endl;
        } else {
            // Cold start for Worker
            float worker_cold_time_ms = measure_worker_time(num_centroids, vector_dimension,
                                                            search_vectors_array, num_search_vectors,
                                                            distances_array, clusters_to_process, 
                                                            total_return_vectors, ids_array, top_k_dist_array);
            centroid_throughput = bytes_scanned_gb/(worker_cold_time_ms/1000.0);

            // Hot start for OpenMP
            float openmp_hot_time_ms = measure_openmp_time(centroids_array, num_centroids, vector_dimension,
                                                           search_vectors_array, num_search_vectors,
                                                           distances_array_omp, cluster_vectors, cluster_ids, 
                                                           num_threads, clusters_to_process, total_return_vectors, 
                                                           ids_array_omp, top_k_dist_array_omp);
            openmp_throughput = bytes_scanned_gb/(openmp_hot_time_ms/1000.0);

            
            std::cout << "Iteration " << i + 1 << ": Hot OpenMP Time = " << openmp_hot_time_ms << " ms having throughput " << openmp_throughput << " GB/s" << std::endl;
            std::cout << "Iteration " << i + 1 << ": Cold Worker Time = " << worker_cold_time_ms << " ms having throughput " << centroid_throughput << " GB/s" << std::endl;
        }

        // Ensure that we get the correct result query by query
        for(int j = 0; j < num_search_vectors; j++) {
            // First ensure the distances are correct
            double* worker_query_distances = distances_array + j * num_centroids;
            double* openmp_query_distances = distances_array_omp + j * num_centroids;

            for(int k = 0; k < num_centroids; k++) {
                double worker_distance = worker_query_distances[k];
                double openmp_distance = openmp_query_distances[k];
                double difference = std::abs(worker_distance - openmp_distance);

                if(difference > std::numeric_limits<double>::epsilon()) {
                    std::cerr << "For iteration " << i << " query " << j << " centroid " << k << " openmp got distance " << openmp_distance << " but worker got distance " << worker_distance << std::endl;
                    std::terminate();
                }
            }

            // Next ensure the distances and ids are correct
            idx_t* worker_query_ids = ids_array + j * total_return_vectors;
            double* worker_query_dists = top_k_dist_array + j * total_return_vectors;
            idx_t* openmp_query_ids = ids_array_omp + j * total_return_vectors;
            double* openmp_query_dists = top_k_dist_array_omp + j * total_return_vectors;
            for(int k = 0; k < total_return_vectors; k++) {
                idx_t worker_vector_id = worker_query_ids[k]; double worker_vector_dist = worker_query_dists[k];
                idx_t openmp_vector_id = openmp_query_ids[k]; double openmp_vector_dist = openmp_query_dists[k];
                idx_t ids_difference = std::abs(worker_vector_id - openmp_vector_id); 
                double dists_difference = std::abs(openmp_vector_dist - worker_vector_dist); 

                if(ids_difference > std::numeric_limits<idx_t>::epsilon() || dists_difference > std::numeric_limits<double>::epsilon()) {
                    std::cerr << "For iteration " << i << " query " << j << " top " << k << "th vector openmp got (" << openmp_vector_id << "," << openmp_vector_dist;
                    std::cerr << ") but worker got (" << worker_vector_id << "," << worker_vector_dist << ")" << std::endl;
                    std::terminate();
                }
            }
        }

        // Record the metric
        if(i >= iterations_to_ignore) {
            centroid_throughput_avg += centroid_throughput;
            openmp_throughput_avg += openmp_throughput;
            counts += 1.0;
        }
    }

    // Perform cleanup
    delete[] search_vectors_array;
    delete[] distances_array;
    delete[] distances_array_omp;
    delete[] ids_array_omp;
    delete[] ids_array;
    delete[] top_k_dist_array_omp;
    delete[] top_k_dist_array;

    start_time = std::chrono::high_resolution_clock::now();
    global_centroid_worker->shutdown();
    delete global_centroid_worker;
    global_cluster_worker->shutdown();
    delete global_cluster_worker;
    end_time = std::chrono::high_resolution_clock::now();
    auto worker_shutdown_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Worker initialization time: " << worker_init_time << " ms" << std::endl;
    std::cout << "Worker shutdown time: " << worker_shutdown_time << " ms" << std::endl;

    // Log the summary of this run
    centroid_throughput_avg = centroid_throughput_avg/counts;
    openmp_throughput_avg = openmp_throughput_avg/counts;
    std::cout << "Centroid average throughput of " << centroid_throughput_avg << " GB/s to scan " << bytes_scanned_gb << " GB" << std::endl;
    std::cout << "Openmp average throughput of " << openmp_throughput_avg << " GB/s to scan " << bytes_scanned_gb << " GB" << std::endl;

    output_file << expected_bytes_scan << "," << num_threads << "," << centroid_throughput_avg << "," << openmp_throughput_avg << std::endl;
}

const std::string BASE_DIR = "/working_dir/dynamic_ivf_benchmarks";

void create_worker(int num_centroids, int file_id) {
    int vector_dimension = 128;
    faiss::MetricType metric_type = faiss::METRIC_L2;

    // Determine the save dir for this 
    std::string save_dir = BASE_DIR + "/" + std::to_string(num_centroids);
    std::filesystem::create_directories(save_dir);
    std::string save_path = save_dir + "/" + std::to_string(file_id) + ".index";
    
    // Generate random data
    std::random_device rd; 
    std::mt19937 generator(rd()); 
    std::uniform_int_distribution<> distribution(750, 1250);

    Tensor insert_centroids = torch::randn({num_centroids - 1, vector_dimension}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<Tensor> insert_vectors;
    std::vector<Tensor> insert_ids;
    int vectors_so_far = 0;
    for(int i = 0; i < num_centroids; i++) {
        int num_vectors = distribution(generator);
        insert_vectors.push_back(torch::randn({num_vectors, vector_dimension}, torch::TensorOptions().dtype(torch::kFloat32)));
        insert_ids.push_back(torch::arange(vectors_so_far, vectors_so_far + num_vectors, torch::kInt64));
        vectors_so_far += num_vectors;
    }

    // First build the index using the first centroids
    auto index = std::make_shared<DynamicIVF_C>(vector_dimension, 1, metric_type, 1);
    Tensor first_partition_vectors = insert_vectors[0]; insert_vectors.erase(insert_vectors.begin());
    Tensor first_partition_ids = insert_ids[0]; insert_ids.erase(insert_ids.begin());
    index->build(first_partition_vectors, first_partition_ids, true, false); 
    
    // Build index using that data
    index->add_partitions(insert_centroids, insert_vectors, insert_ids);
    index->save(save_path);
    std::cout << "Saved dataset to " << save_path << std::endl;
}

void create_dataset() {
    int num_datasets = 1;
    int min_cluster_size = 128;
    int max_cluster_size = 16384;

    // Launch the workers
    std::vector<std::thread> workers;
    for(int cluster_size = min_cluster_size; cluster_size <= max_cluster_size; cluster_size *= 2) {
        workers.emplace_back(&create_worker, cluster_size, 0);
    }

    // Join the workers
    for (std::thread &worker: workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers.clear();
}

int main(int argc, char* argv[]) {
    #ifdef __linux__
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system." << std::endl;
        exit(1);
    } else {
        std::cout << "NUMA is available on this system." << std::endl;
    }
#endif
    
    // If some argument is passed then create the dataset
    if(argc >= 2) {
        create_dataset();
        return 0;
    } 
    // Define the search space
    int num_threads_min = 1;
    int num_threads_max = 64;
    int num_centroids_min = 2048;
    int num_centroids_max = 32768;

    // Define the index data
    int vector_dimension = 128;
    int num_search_vectors = 1;
    int total_return_vectors = 10;

    // Create the output file
    std::string output_file_path = "../results/complete_partition_scan_throughput.csv";
    std::ofstream output_file(output_file_path);
    if(!output_file.is_open()) {
        throw std::runtime_error("Unable to open file " + output_file_path);
    }
    output_file << "buffer_size_bytes,num_threads,centroid_throughput_gb_s,openmp_throughput_gb_s" << std::endl;

    // Seed the random number generators
    srand(time(NULL));
    torch::manual_seed(std::chrono::system_clock::now().time_since_epoch().count());

    for(int num_centroids = num_centroids_min; num_centroids <= num_centroids_max; num_centroids *= 2) {
        // Extract the centroids from the saved index
        std::string load_dir = BASE_DIR + "/" + std::to_string(num_centroids);
        std::vector<std::string> index_files;
        for (const auto& entry : std::filesystem::directory_iterator(load_dir)) {
            if (entry.path().extension() == ".index") {
                index_files.push_back(entry.path().string());
            }
        }

        if(index_files.size() == 0) {
            std::cerr << "Skipping processing for num centroids of " << num_centroids << " as no index files" << std::endl;
            continue;
        }

        // Read the index from one of these files
        int file_to_load = abs(((int) rand())) % index_files.size();
        std::string index_path = index_files[file_to_load];
        std::cout << "Loading index " << index_path << std::endl;
        DynamicIVF_C index(vector_dimension, num_centroids, faiss::METRIC_L2, -1, -1, false);
        index.load(index_path);


        // Copy over the centroids and clusters
        Tensor all_centroids = index.centroids().contiguous();
        int num_centroid_values = num_centroids * vector_dimension;
        float* centroids_array = new float[num_centroid_values];
        std::memcpy(centroids_array, all_centroids.data_ptr<float>(), num_centroid_values * sizeof(float));

        auto index_partitions = index.get_partitions();
        std::vector<std::vector<uint8_t>> cluster_vectors = std::get<0>(index_partitions);
        std::vector<std::vector<idx_t>> cluster_ids = std::get<1>(index_partitions);
        std::cout << "Finished loading index " << index_path << std::endl;

        int clusters_to_process = num_centroids;
        for (int num_threads = num_threads_min; num_threads <= num_threads_max; num_threads *= 2) {
            // Run the experiment
            std::cout << std::endl
                    << "------- START: Num Centroids - " << num_centroids << ", Num Search Vectors - "
                    << num_search_vectors << ", Num Threads - " << num_threads << ", Num Partitions to Scan - " 
                    << clusters_to_process << ", Total Return Vectors - " << total_return_vectors << " -------" << std::endl;

            run_experiment(centroids_array, cluster_vectors, cluster_ids, num_centroids, vector_dimension, num_search_vectors, 
                clusters_to_process, total_return_vectors, num_threads, output_file);

            std::cout << "------- START: Num Centroids - " << num_centroids << ", Num Search Vectors - "
                << num_search_vectors << ", Num Threads - " << num_threads << ", Num Partitions to Scan - " << clusters_to_process 
                << ", Total Return Vectors - " << total_return_vectors << " -------" << std::endl;
        }

        // Perform cleanup
        delete[] centroids_array;
    }
}