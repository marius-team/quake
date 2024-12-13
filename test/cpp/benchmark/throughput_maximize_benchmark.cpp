#include <iostream>
#include <sched.h>
#include <torch/torch.h>
#include <cassert>
#include <chrono>
#include <vector>
#include <thread>
#include <string>
#include <stdint.h>
#include <stdio.h>
#include <stdexcept>
#include <math.h>
#include <mutex>
#include <atomic>
#include <thread>
#include <random>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <math.h>
#include <numeric>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <simsimd/simsimd.h>
#include <fstream>

#ifdef __linux__
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#endif

typedef unsigned long long longType;

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using torch::Tensor;
using std::vector;

constexpr bool DEBUG_MODE = false;
inline void summarize_values(char* buffer, int buffer_size, std::string label) {
    int sum = std::accumulate(buffer, buffer + buffer_size, 0, std::plus<int>());
    std::string txt_to_write = label + " has mean of " + std::to_string(sum) + "\n";
    std::cout << txt_to_write << std::endl;
}

// Global CentroidScanWorker declaration
class CentroidScanWorker;
CentroidScanWorker *global_centroid_worker = nullptr;

// CentroidScanWorker class definition
class CentroidScanWorker {
public:
    CentroidScanWorker(size_t num_threads, int num_centroids, int vector_dimension);

    ~CentroidScanWorker();

    // Initialize the worker threads
    void init(float *centroids, int num_centroids, int vector_dimension, bool use_memcpy_worker = false, bool verify_numa = false);

    // Submit a query to be processed
    void process_query(float* query_vector, double* distances_array, bool log_aggregate = false);

    // Stop the worker threads
    void shutdown();

    // Get the number of threads
    size_t num_threads() const { return workers.size(); }

    // See if all the workers have been initialized
    bool are_workers_ready() { return ready_threads.load() == num_threads(); }

private:
    // Worker function
    void memcpy_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa = false);
    void debug_scan_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa = false);
    void scan_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa = false);

    // Worker thread details>
    std::vector<std::thread> workers;
    std::vector<float> worker_avg_throughput;
    std::vector<std::shared_ptr<std::atomic<int>>> numa_cumulative_throughput;
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

    // Additional metadata
    std::atomic<int> ready_threads;
};

CentroidScanWorker::CentroidScanWorker(size_t num_threads, int num_centroids, int vector_dimension)
    : stop(false),
      current_query_vector(nullptr),
      centroids(nullptr),
      distances_array(nullptr),
      num_centroids(num_centroids),
      vector_dimension(vector_dimension),
      threads_remaining(0),
      query_id(-1),
      n_threads(num_threads),
      ready_threads(0) 
{
}

CentroidScanWorker::~CentroidScanWorker() {
    shutdown();
}

void CentroidScanWorker::init(float *centroids, int num_centroids, int vector_dimension, bool use_memcpy_worker, bool verify_numa) {
    std::unique_lock<std::mutex> lock(mutex);
    this->centroids = centroids;
    this->num_centroids = num_centroids;
    this->vector_dimension = vector_dimension;

    int chunk_size = (num_centroids + n_threads - 1) / n_threads;
    ready_threads.store(0);
    for (int t = 0; t < n_threads; t++) {
        // Thread should be responsible for at least one centroid
        int start_idx = std::min(t * chunk_size, num_centroids);
        if(start_idx == num_centroids) {
            break;
        }
        int end_idx = std::min(start_idx + chunk_size, num_centroids);

        if(use_memcpy_worker) {
            workers.emplace_back(&CentroidScanWorker::memcpy_worker_thread, this, t, start_idx, end_idx, verify_numa);
        } else if constexpr(DEBUG_MODE) {
            workers.emplace_back(&CentroidScanWorker::debug_scan_worker_thread, this, t, start_idx, end_idx, verify_numa);
        } else {
            workers.emplace_back(&CentroidScanWorker::scan_worker_thread, this, t, start_idx, end_idx, verify_numa);
        }
        worker_avg_throughput.push_back(0.0);
    }

    // Also allocate the buffer for numa node throughput
    int num_numa_nodes = numa_max_node() + 1;
    for(int i = 0; i < num_numa_nodes; i++) {
        numa_cumulative_throughput.push_back(std::make_shared<std::atomic<int>>(0.0));
    }
}

void CentroidScanWorker::process_query(float* query_vector, double* distances_array, bool log_aggregate) { 
    // Reset the numa throughput
    if(DEBUG_MODE) {
        int num_numa_nodes = numa_max_node() + 1;
        for(int i = 0; i < num_numa_nodes; i++) {
            numa_cumulative_throughput[i]->store(0);
        }
    }

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

    // Log the aggregate throughput
    if(DEBUG_MODE && log_aggregate) {
        // Get the throughput per numa node
        int num_numa_nodes = numa_max_node() + 1;
        std::cout << "Got aggregate throughput per numa node of: ";
        for(int i = 0; i < num_numa_nodes; i++) {
            std::cout << numa_cumulative_throughput[i]->load() << " ";
        }
        std::cout << std::endl;

        // Get the average throughput
        float average_throughput = std::accumulate(worker_avg_throughput.begin(), worker_avg_throughput.end(), 0.0)/(1.0 * worker_avg_throughput.size());
        std::cout << "Average thread throughput of " << average_throughput << " GB/s" << std::endl;
    }
}

void CentroidScanWorker::shutdown() { 
    {
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

void CentroidScanWorker::memcpy_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa) {
    // Bind the thread to a specific CPU
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
    int last_processed_query_id = -1;

    // Determine the sizes
    int64_t n_local_centroids = end_idx - start_idx;
    int64_t local_centroids_size = n_local_centroids * vector_dimension * sizeof(float);

    float* local_centroids = (float*) numa_alloc_onnode(local_centroids_size, numa_node_id);
    float* local_copy_buffer = (float*) numa_alloc_onnode(local_centroids_size, numa_node_id);
    if(local_centroids == NULL || local_copy_buffer == NULL) {
        std::string err_msg = "Unable to allocate data on numa node " + std::to_string(numa_node_id);
        throw std::runtime_error(err_msg);
    }

    // Mem lock the buffers
    mlock(local_centroids, local_centroids_size);
    mlock(local_copy_buffer, local_centroids_size);

    // Copy centroids data to local centroids
    std::memcpy(local_centroids, centroids + start_idx * vector_dimension, local_centroids_size);
    std::memset(local_copy_buffer, 0, local_centroids_size);

    float thread_centroids_size_gb = (1.0 * local_centroids_size)/(1.0 * pow(10, 9));
    std::stringstream mapping_log_stream;
    mapping_log_stream << "Thread " << thread_id << " responsible for " << thread_centroids_size_gb << " GB running on cpu " << worker_cpu << "/" << num_cpus << " mapping to numa node " << numa_node_id << std::endl;
    std::cout << mapping_log_stream.str();

    // Maark this thread as ready
    ready_threads.fetch_add(1);
    int current_query_id;
    while (true) {
        // Wait for a new query
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this, &last_processed_query_id]() {
                return query_id.load() != last_processed_query_id || stop;
            });

            if (stop) {
                numa_free(local_centroids, local_centroids_size);
                numa_free(local_copy_buffer, local_centroids_size);
                return;
            }

            // Load the current query id
            current_query_id = query_id.load();
        }
        
        if(verify_numa) {
            // Verify that the data and worker are on the correct numa node
            int curr_worker_numa_node = numa_node_of_cpu(sched_getcpu()); // Mainly checks that thread has not migrated
            if(curr_worker_numa_node != numa_node_id) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but cluster on node " + std::to_string(numa_node_id);
                throw std::runtime_error(err_msg);
            }

            int centroids_numa_node = -1;
            get_mempolicy(&centroids_numa_node, NULL, 0, (void*) local_centroids, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != centroids_numa_node) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but centroids on node " + std::to_string(centroids_numa_node);
                throw std::runtime_error(err_msg);
            }

            int copy_numa_node = -1;
            get_mempolicy(&copy_numa_node, NULL, 0, (void*) local_copy_buffer, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != copy_numa_node) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but centroids on node " + std::to_string(centroids_numa_node);
                throw std::runtime_error(err_msg);
            }
        }

        // Just perform a scan
        auto start = std::chrono::high_resolution_clock::now();
        std::memcpy(local_copy_buffer, local_centroids, local_centroids_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Log the thread average
        float worker_copy_time_ms = duration.count()/1000.0;
        float worker_throughput = thread_centroids_size_gb/(worker_copy_time_ms/1000.0);
        worker_avg_throughput[thread_id] = worker_throughput;
        
        // Log the numa cumalativate
        int curr_numa_node = numa_node_of_cpu(sched_getcpu());
        if(curr_numa_node != numa_node_id) {
            std::string err_msg = "Thread " + std::to_string(thread_id) + " migrated from numa node " + std::to_string(numa_node_id) + " to " + std::to_string(curr_numa_node);
            throw std::runtime_error(err_msg);
        }
        numa_cumulative_throughput[curr_numa_node]->fetch_add((int) worker_throughput);

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

void CentroidScanWorker::debug_scan_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa) {
    // Bind the thread to a specific CPU
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
    int last_processed_query_id = -1;

    // Determine the sizes
    int64_t n_local_centroids = end_idx - start_idx;
    int64_t local_centroids_size = n_local_centroids * vector_dimension * sizeof(float);
    int64_t local_vector_size = vector_dimension * sizeof(float);
    int64_t local_distances_size = n_local_centroids * sizeof(double);

    // Allocate the buffers
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

    // Mem lock the buffers
    mlock(local_centroids, local_centroids_size);
    mlock(local_query_vector, local_vector_size);
    mlock(local_distances, local_distances_size);

    // Copy centroids data to local centroids
    std::memcpy(local_centroids, centroids + start_idx * vector_dimension, local_centroids_size);
    std::memset(local_distances, 0, local_distances_size);

    float thread_centroids_size_gb = (1.0 * local_centroids_size)/(1.0 * pow(10, 9));
    std::stringstream mapping_log_stream;
    mapping_log_stream << "Thread " << thread_id << " responsible for " << thread_centroids_size_gb << " GB running on cpu " << worker_cpu << "/" << num_cpus << " mapping to numa node " << numa_node_id << std::endl;
    std::cout << mapping_log_stream.str();

    // Determine the distance function
    simsimd_metric_punned_t distance_function = simsimd_metric_punned(
        simsimd_metric_dot_k,   
        simsimd_datatype_f32_k, 
        simsimd_cap_any_k);

    // Maark this thread as ready
    ready_threads.fetch_add(1);
    int current_query_id;
    while (true) {
        // Wait for a new query
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this, &last_processed_query_id]() {
                return query_id.load() != last_processed_query_id || stop;
            });

            if (stop) {
                numa_free(local_centroids, local_centroids_size);
                numa_free(local_query_vector, local_vector_size);
                numa_free(local_distances, local_distances_size);
                return;
            }

            // Load the current query id
            current_query_id = query_id.load();
            std::memcpy(local_query_vector, current_query_vector, local_vector_size);
        }
        
        if(verify_numa) {
            // Verify that the data and worker are on the correct numa node
            int curr_worker_numa_node = numa_node_of_cpu(sched_getcpu()); // Mainly checks that thread has not migrated
            if(curr_worker_numa_node != numa_node_id) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but cluster on node " + std::to_string(numa_node_id);
                throw std::runtime_error(err_msg);
            }

            int centroids_numa_node = -1;
            get_mempolicy(&centroids_numa_node, NULL, 0, (void*) local_centroids, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != centroids_numa_node) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but centroids on node " + std::to_string(centroids_numa_node);
                throw std::runtime_error(err_msg);
            }

            int query_numa_node = -1;
            get_mempolicy(&query_numa_node, NULL, 0, (void*) local_query_vector, MPOL_F_NODE | MPOL_F_ADDR);
            if(query_numa_node != centroids_numa_node) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but query on node " + std::to_string(query_numa_node);
                throw std::runtime_error(err_msg);
            }

            int distances_numa_node = -1;
            get_mempolicy(&distances_numa_node, NULL, 0, (void*) local_distances, MPOL_F_NODE | MPOL_F_ADDR);
            if(distances_numa_node != centroids_numa_node) {
                std::string err_msg = "Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but distances on node " + std::to_string(distances_numa_node);
                throw std::runtime_error(err_msg);
            }
        }

        // Just perform a scan
        auto start = std::chrono::high_resolution_clock::now();
#pragma unroll
        for (int i = 0; i < n_local_centroids; i++) {
            distance_function(local_query_vector, local_centroids + i * vector_dimension, vector_dimension, local_distances + i);

            if constexpr(DEBUG_MODE) {
                summarize_values(reinterpret_cast<char*>(local_query_vector), vector_dimension * sizeof(float), "Numa search vector");
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Log the thread average
        float worker_copy_time_ms = duration.count()/1000.0;
        float worker_throughput = thread_centroids_size_gb/(worker_copy_time_ms/1000.0);
        worker_avg_throughput[thread_id] = worker_throughput;
        
        // Log the numa cumalativate
        int curr_numa_node = numa_node_of_cpu(sched_getcpu());
        if(curr_numa_node != numa_node_id) {
            std::string err_msg = "Thread " + std::to_string(thread_id) + " migrated from numa node " + std::to_string(numa_node_id) + " to " + std::to_string(curr_numa_node);
            throw std::runtime_error(err_msg);
        }
        numa_cumulative_throughput[curr_numa_node]->fetch_add((int) worker_throughput);

        // copy the local distances to the global distances
        std::memcpy(distances_array + start_idx, local_distances, local_distances_size);

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

void CentroidScanWorker::scan_worker_thread(int thread_id, int start_idx, int end_idx, bool verify_numa) {
    // Bind the thread to a specific CPU
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
    int last_processed_query_id = -1;

    // Determine the sizes
    int64_t n_local_centroids = end_idx - start_idx;
    int64_t local_centroids_size = n_local_centroids * vector_dimension * sizeof(float);
    int64_t local_vector_size = vector_dimension * sizeof(float);
    int64_t local_distances_size = n_local_centroids * sizeof(double);

    // Allocate the buffers
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

    // Mem lock the buffers
    mlock(local_centroids, local_centroids_size);
    mlock(local_query_vector, local_vector_size);
    mlock(local_distances, local_distances_size);

    // Copy centroids data to local centroids
    std::memcpy(local_centroids, centroids + start_idx * vector_dimension, local_centroids_size);
    std::memset(local_distances, 0, local_distances_size);

    // Maark this thread as ready
    ready_threads.fetch_add(1);
    int current_query_id;
    while (true) {
        // Wait for a new query
        {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [this, &last_processed_query_id]() {
                return query_id.load() != last_processed_query_id || stop;
            });

            if (stop) {
                numa_free(local_centroids, local_centroids_size);
                numa_free(local_query_vector, local_vector_size);
                numa_free(local_distances, local_distances_size);
                return;
            }

            // Load the current query id
            current_query_id = query_id.load();
            std::memcpy(local_query_vector, current_query_vector, local_vector_size);
        }

        // Just perform a scan
#pragma unroll
        for (int i = 0; i < n_local_centroids; i++) {
            simsimd_dot_f32(local_centroids + i * vector_dimension, local_query_vector, vector_dimension, local_distances + i);
        }

        // copy the local distances to the global distances
        std::memcpy(distances_array + start_idx, local_distances, local_distances_size);

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

float get_numa_time(float* search_vector_ptr, double* distances_ptr, bool log_aggregate) {
    auto start = std::chrono::high_resolution_clock::now();
    global_centroid_worker->process_query(search_vector_ptr, distances_ptr, log_aggregate);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count()/1000.0;
}

float get_openmp_time(float* centroids_data, longType num_centroids, int vector_dimension, int num_threads, float* search_vector_ptr, double* distances_ptr) {    
    // Perform the centroid search
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(num_threads)
    for(longType i = 0; i < num_centroids; i++) {
        double dist_result;
        simsimd_dot_f32(centroids_data + i * vector_dimension, search_vector_ptr, vector_dimension, &dist_result);
        distances_ptr[i] = dist_result;

        if constexpr(DEBUG_MODE) {
            summarize_values(reinterpret_cast<char*>(search_vector_ptr), vector_dimension * sizeof(float), "Openmp search vector");
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count()/1000.0;
}

void run_experiment(float* centroids_data, longType num_centroids, int vector_dimension, int num_threads, std::ofstream& output_file, 
int num_iterations = 5, int iterations_to_ignore = 1) {
    // Initialize the global centroid worker
    global_centroid_worker = new CentroidScanWorker(num_threads, num_centroids, vector_dimension);
    global_centroid_worker->init(centroids_data, num_centroids, vector_dimension);

    // Sleep until all threads are ready (increments of 2 seconds)
    while(!global_centroid_worker->are_workers_ready()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    }

    // Determine the centroids size
    longType expected_values = num_centroids * vector_dimension;
    longType expected_bytes_scan = expected_values * sizeof(float);
    float bytes_scanned_gb = (1.0 * expected_bytes_scan)/(1.0 * pow(10, 9));

    // Create the search vector and the numa output buffer
    Tensor search_vectors_tensor = torch::randn({vector_dimension}).contiguous();
    float* search_vector_ptr = search_vectors_tensor.data_ptr<float>();
    Tensor numa_distances_tensor = torch::empty({num_centroids}, torch::kFloat64).contiguous();
    double* numa_distances_ptr = numa_distances_tensor.data_ptr<double>();

    // Create the openmp output buffer
    Tensor openmp_distances_tensor = torch::empty({num_centroids}, torch::kFloat64).contiguous();
    double* openmp_distances_ptr = openmp_distances_tensor.data_ptr<double>();

    // Just benchmark single thread memcpy
    float centroid_throughput_avg = 0.0; float openmp_throughput_avg = 0.0; float counts = 0.0;
    for(int i = 0; i < num_iterations; i++) {
        // Benchmark the global centroid worker
        bool numa_first = i % 2 == 0;
        bool log_aggregate = i == (num_iterations - 1);
        float centroid_worker_time = -1.0; float openmp_worker_time = -1.0;
        if(numa_first) {
            centroid_worker_time = get_numa_time(search_vector_ptr, numa_distances_ptr, log_aggregate);
            openmp_worker_time = get_openmp_time(centroids_data, num_centroids, vector_dimension, num_threads, search_vector_ptr, openmp_distances_ptr);
        } else {
            openmp_worker_time = get_openmp_time(centroids_data, num_centroids, vector_dimension, num_threads, search_vector_ptr, openmp_distances_ptr);
            centroid_worker_time = get_numa_time(search_vector_ptr, numa_distances_ptr, log_aggregate);
        }

        // Make sure they produced the same response
        for(int j = 0; j < num_centroids; j++) {
            double difference = std::abs(numa_distances_ptr[j] - openmp_distances_ptr[j]);
            if(difference > std::numeric_limits<double>::epsilon()) {
                std::cerr << "For iteration " << i << " centroid " << j << " openmp got distance " << openmp_distances_ptr[j] << " but numa got distance " << numa_distances_ptr[j] << std::endl;
                throw std::runtime_error("Centroid scan worker produced incorrect result");
            }
        }
        
        float centroid_throughput = bytes_scanned_gb/(centroid_worker_time/1000.0);
        float openmp_throughput = bytes_scanned_gb/(openmp_worker_time/1000.0);

        // Record the metric
        if(i >= iterations_to_ignore) {
            centroid_throughput_avg += centroid_throughput;
            openmp_throughput_avg += openmp_throughput;
            counts += 1.0;
        }
    }
    global_centroid_worker->shutdown();
    delete global_centroid_worker;
    centroid_throughput_avg = centroid_throughput_avg/counts;
    openmp_throughput_avg = openmp_throughput_avg/counts;
    
    // Log the times
    std::cout << "Centroid average throughput of " << centroid_throughput_avg << " GB/s" << std::endl;
    std::cout << "Openmp average throughput of " << openmp_throughput_avg << " GB/s" << std::endl;
    output_file << expected_bytes_scan << "," << num_threads << "," << centroid_throughput_avg << "," << openmp_throughput_avg << std::endl;
}

void run_baseline_experiment(longType num_centroids, int vector_dimension) {
    // Determine the centroids size
    longType expected_values = num_centroids * vector_dimension;
    longType expected_bytes_scan = expected_values * sizeof(float);
    float bytes_scanned_gb = (1.0 * expected_bytes_scan)/(1.0 * pow(10, 9));
    
    // Allocate and set the buffers
    float* buffer1 = new float[expected_values];
    float* buffer2 = new float[expected_values];
    std::memset(buffer1, 0, expected_bytes_scan);
    std::memset(buffer1, 1, expected_bytes_scan);

    // Measure the memcpy time
    auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(buffer2, buffer1, expected_bytes_scan);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    float copy_time_sec = duration.count()/1000.0;

    // Log the details
    std::cout << "BASELINE: Standard memcpy of " << bytes_scanned_gb << " GB took " << copy_time_sec << " sec resulting in throughput " << (1.0 * bytes_scanned_gb)/copy_time_sec << " GB/s" << std::endl << std::endl;

    // Perform cleanuo
    delete[] buffer1;
    delete[] buffer2;
}

int main(int argc, char* argv[]) {
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system." << std::endl;
        exit(1);
    } else {
        std::cout << "NUMA is available on this system." << std::endl;
    }

    // Define the search space
    int num_threads_min = 1;
    int num_threads_max = 128;
    std::vector<longType> target_buffer_sizes = {
        (longType) pow(10, 7),
        (longType) pow(10, 8),
        (longType) pow(10, 9),
        8 * ((longType) pow(10, 9)),
    };

    // Define the parameters
    int vector_dimension = 128;

    // Create the output file
    std::string output_file_path = "../results/centroid_scan_throughput.csv";
    std::ofstream output_file(output_file_path);
    if(!output_file.is_open()) {
        throw std::runtime_error("Unable to open file " + output_file_path);
    }
    
    // Run the baseline
    output_file << "buffer_size_bytes,num_threads,centroid_throughput_gb_s,openmp_throughput_gb_s" << std::endl;
    omp_set_num_threads(1);
    for(longType curr_buffer_size : target_buffer_sizes) {
        int num_centroids = curr_buffer_size/(vector_dimension * sizeof(float));
        std::cout << "Creating tensor of shaped (" << num_centroids << "," << vector_dimension << ") for target size " << curr_buffer_size << std::endl;
        Tensor centroids_tensor = torch::randn({num_centroids, vector_dimension}).contiguous();
        float* centroids_data = centroids_tensor.data_ptr<float>();
        std::cout << "Finished creating tensor of shaped (" << num_centroids << "," << vector_dimension << ")" << std::endl;
        
        for(int num_threads = num_threads_min; num_threads <= num_threads_max; num_threads *= 2) {
            std::cout << "------- START: Num Centroids - " << num_centroids << ", Num Threads - " << num_threads << " -------" << std::endl;
            run_experiment(centroids_data, num_centroids, vector_dimension, num_threads, output_file);
            std::cout << "------- END: Num Centroids - " << num_centroids << ", Num Threads - " << num_threads << " -------" << std::endl << std::endl;
            break;
        }
        break;
    }

    output_file.close();
}