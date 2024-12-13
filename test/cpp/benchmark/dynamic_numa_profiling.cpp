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
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <tuple>
#include <algorithm>

#include "faiss/impl/platform_macros.h"
#include "faiss/IVFlib.h"
#include "faiss/index_io.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexFlat.h"
#include "dynamic_ivf.h"
#include "args.h"

#ifdef __linux__
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#endif

typedef unsigned long long longType;
typedef faiss::idx_t idx_t;
using torch::Tensor;
using std::vector;
using TopKClusters = TypedTopKBuffer<double, int>;

static constexpr bool LOG_MODE = false;
const std::string BASE_DIR = "/working_dir/dynamic_ivf_benchmarks";

// From https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos 
std::tuple<int, int> get_summary(std::vector<float> timings) {
    float sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    float mean = sum / (1.0 * timings.size());

    std::vector<float> diff(timings.size());
    std::transform(timings.begin(), timings.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float std_dev = std::sqrt(sq_sum / (1.0 * timings.size()));
    return std::make_tuple( (int) mean, (int) std_dev);
}

void run_experiment(std::string index_path, int num_centroids, int vector_dimension, int num_search_vectors, int percentage_clusters,  
int total_return_vectors, int num_threads, int target_query_latency_us, int flush_gap_time_us, std::ofstream& output_file, int iterations, 
bool log_mode, bool verify_numa, bool same_core) {

    // Calculate some necessary values
    int clusters_to_process = (num_centroids * percentage_clusters)/100;

    // Load the index
    auto openmp_index = std::make_shared<DynamicIVF_C>(vector_dimension, num_centroids, faiss::METRIC_L2, num_threads, -1, -1, false, log_mode, verify_numa, same_core);
    openmp_index->load(index_path);
    std::cout << "Loaded the OPENMP index" << std::endl;

    auto numa_index = std::make_shared<DynamicIVF_C>(vector_dimension, num_centroids, faiss::METRIC_L2, num_threads, -1, -1, true, log_mode, verify_numa, same_core);
    numa_index->load(index_path);
    numa_index->set_timeout_values(target_query_latency_us, flush_gap_time_us);
    std::cout << "Loaded the NUMA index" << std::endl;

    // Create the search vectors
    std::vector<Tensor> search_vectors;
    for(int i = 0; i < num_search_vectors; i++) {
        search_vectors.push_back(torch::randn({1, vector_dimension}).contiguous());
    }

    // Create the vector storing the timings
    std::vector<float> openmp_all_throughputs; std::vector<float> openmp_all_latencies;
    std::vector<float> numa_all_throughputs; std::vector<float> numa_all_latencies;
    int64_t openmp_scanned_bytes;
    int64_t numa_scanned_bytes;

    // Create the buffer to determine the ordering
    int ordering[2];
    for(int i = 0; i < 2; i++) {
        ordering[i] = i;
    }

    // Wait for the indexes to be ready
    while(!numa_index->index_ready()) {
        std::cout << "Sleeping due to the index not being ready" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Perform the actual trials
    for (int i = 0; i < iterations; i++) {
        for(int j = 0; j < num_search_vectors; j++) {
            Tensor curr_search_vector = search_vectors[j];
            int query_id = i * num_search_vectors + j;
            bool run_omp_first = (query_id % 2 == 0); // Alternate execution order
            std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > openmp_result; 
            std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > numa_result;

            // Determine the ordering
            std::random_shuffle ( std::begin(ordering), std::end(ordering) );
            for(int k = 0; k < 2; k++) {
                if(ordering[k] == 0) {
                    openmp_result = openmp_index->search(curr_search_vector, clusters_to_process, total_return_vectors); 
                    if(log_mode) {
                        std::cout << "Iteration " << i + 1 << ", Vector " << j + 1 << ": OpenMP details - " << std::endl;
                        std::get<2>(openmp_result)->print(); std::cout << std::endl;
                    }
                } else if(ordering[k] == 1) {
                    numa_result = numa_index->search(curr_search_vector, clusters_to_process, total_return_vectors); 
                    if(log_mode) {
                        std::cout << "Iteration " << i + 1 << ", Vector " << j + 1 << ": Numa Normal Alloc details - " << std::endl;
                        std::get<2>(numa_result)->print(); std::cout << std::endl;
                    }
                }
            }
            
            // Verify the output
            if(target_query_latency_us == -1) {
                float openmp_ids_mean = torch::mean(std::get<0>(openmp_result).to(torch::kFloat32)).item<float>();
                float numa_ids_mean =  torch::mean(std::get<0>(numa_result).to(torch::kFloat32)).item<float>();
                if(openmp_ids_mean != numa_ids_mean) {
                    std::string err_msg = "ERROR: Iteration " + std::to_string(i + 1) + " got Openmp IDs sum of " + std::to_string(openmp_ids_mean) + " but NUMA sum of " + std::to_string(numa_ids_mean);
                    throw std::runtime_error(err_msg);
                }
                
                float openmp_dists_mean = torch::mean(std::get<1>(openmp_result).to(torch::kFloat32)).item<float>();
                float numa_dists_mean = torch::mean(std::get<1>(numa_result).to(torch::kFloat32)).item<float>();
                if(openmp_dists_mean != numa_dists_mean) {
                    std::string err_msg = "ERROR: Iteration " + std::to_string(i + 1) + " got Openmp dists sum of " + std::to_string(openmp_ids_mean) + " but NUMA sum of " + std::to_string(numa_ids_mean);
                    throw std::runtime_error(err_msg);
                }
            }

            // Extract the numa and openmp times
            std::shared_ptr<SearchTimingInfo> openmp_result_values = std::get<2>(openmp_result);
            float openmp_latency = openmp_result_values->total_time_us; float openmp_throughput = openmp_result_values->get_scan_throughput();
            std::shared_ptr<SearchTimingInfo> numa_result_values = std::get<2>(numa_result);
            float numa_latency = numa_result_values->total_time_us; float numa_throughput = numa_result_values->get_scan_throughput();

            // Record those times
            openmp_all_throughputs.push_back(openmp_throughput); openmp_all_latencies.push_back(openmp_latency);
            numa_all_throughputs.push_back(numa_throughput); numa_all_latencies.push_back(numa_latency);

            if(i == 0 && j == 0) {
                openmp_scanned_bytes = openmp_result_values->get_scan_bytes();
                numa_scanned_bytes = numa_result_values->get_scan_bytes();
            }

            if(log_mode) {
                std::cout << "--- START ITERATION SUMMARY ---" << std::endl;
                std::cout << "Query " << i << " Iteration " << j << " Openmp Values: Latency - " << openmp_latency << " uS, Throughput - " << openmp_throughput << " GB/s" << std::endl;
                std::cout << "Query " << i << " Iteration " << j << " Numa Values: Latency - " << numa_latency << " uS, Throughput - " << numa_throughput << " GB/s" << std::endl;
                std::cout << "--- END ITERATION SUMMARY ---" << std::endl;
            } 
        }
    }

    // Get the openmp summary
    std::tuple<int, int> openmp_throughput_values = get_summary(openmp_all_throughputs); 
    std::string openmp_throughput_summary = std::to_string(std::get<0>(openmp_throughput_values)) + " +/- " + std::to_string(std::get<1>(openmp_throughput_values)) + " GB/s";
    std::tuple<int, int> openmp_latency_values = get_summary(openmp_all_latencies);
    std::string openmp_latency_summary = std::to_string(std::get<0>(openmp_latency_values)) + " +/- " + std::to_string(std::get<1>(openmp_latency_values)) + " uS";
    output_file << num_centroids << "," << openmp_scanned_bytes << "," << percentage_clusters << "," << num_threads << ",openmp," << openmp_latency_summary << "," << openmp_throughput_summary << std::endl;
    std::cout << "OPENMP Summary - Latency: " << openmp_latency_summary << ", Throughput: " << openmp_throughput_summary << std::endl;

    // Also get the openmp summary
    std::tuple<int, int> numa_throughput_values = get_summary(numa_all_throughputs); 
    std::string numa_throughput_summary = std::to_string(std::get<0>(numa_throughput_values)) + " +/- " + std::to_string(std::get<1>(numa_throughput_values)) + " GB/s";
    std::tuple<int, int> numa_latency_values = get_summary(numa_all_latencies);
    std::string numa_latency_summary = std::to_string(std::get<0>(numa_latency_values)) + " +/- " + std::to_string(std::get<1>(numa_latency_values)) + " uS";
    output_file << num_centroids << "," << numa_scanned_bytes << "," << percentage_clusters << "," << num_threads << ",numa," << numa_latency_summary << "," << numa_throughput_summary << std::endl;
    std::cout << "NUMA Summary - Latency: " << numa_latency_summary << ", Throughput: " << numa_throughput_summary << std::endl;
}

struct input_args {
    static const char* help() {
        return "Program to benchmark numa and openmp based cluster scan implementations";
    }

    int num_centroids_min;
    int num_centroids_max;
    int vector_dimension;
    int num_queries;
    int percent_clusters_min;
    int percent_clusters_max;
    int total_return_vectors;
    int num_threads_min;
    int num_iterations;
    int num_threads_max;
    int target_query_latency_us;
    int flush_gap_time_us;
    std::string mapping_method;
    std::string output_file_path;
    std::string log_method;
    std::string numa_verification_method;
    std::string core_mapping_method;

    input_args() : 
        num_threads_min(1), 
        num_threads_max(64), 
        output_file_path("results.csv"), 
        log_method("limited"),
        numa_verification_method("none"),
        num_centroids_min(1024),
        num_centroids_max(32768),
        vector_dimension(128), 
        num_queries(1), 
        percent_clusters_min(1),
        percent_clusters_max(10),
        total_return_vectors(16),
        num_iterations(5),
        core_mapping_method("same"),
        target_query_latency_us(-1),
        flush_gap_time_us(1500)
    {

    }

    template<class F>
    void parse(F f)
    {
        f(num_centroids_min, "--num_centroids_min", "-C", args::help("The minimium number of centroids in the index we want to benchmark"));
        f(num_centroids_max, "--num_centroids_max", "-M", args::help("The max number of centroids in the index we want to benchmark"));
        f(vector_dimension, "--vector_dimension", "-V", args::help("The dimension of the vector in the benchmark"));
        f(num_queries, "--num_queries", "-N", args::help("The number of queries in our benchmark"));
        f(percent_clusters_min, "--min_clusters_percentage", "-P", args::help("The minimum percentage of clusters we should consider"));
        f(percent_clusters_max, "--max_clusters_percentage", "-X", args::help("The maximum percentage of clusters we should consider"));
        f(total_return_vectors, "--total_return_vectors", "-T", args::help("The number of vectors that a scan query should return"));
        f(num_threads_min, "--num_threads_min", "-L", args::help("The minimum number of threads we want to use in our experiment"));
        f(num_threads_max, "--num_threads_max", "-U", args::help("The maximum number of threads we want to use in our experiment"));
        f(output_file_path, "--output_file_path", "-O", args::help("The output file we want to write the experimental results to"));
        f(log_method, "--log_method", "-D", args::help("The logging method we should when running this experiment"));
        f(num_iterations, "--num_iterations", "-I", args::help("The number of iterations we should have"));
        f(numa_verification_method, "--numa_verify_method", "-N", args::help("The level to which we should check if numa is being enforced"));
        f(core_mapping_method, "--core_mapping_method", "-R", args::help("The techinque we should use to map partition workers to cores"));
        f(target_query_latency_us, "--target_query_latency", "-S", "The target latency we want to hit for the query (in uS)");
        f(flush_gap_time_us, "--flush_gap_time", "-Q", "The gap between flushes on the main thread");
    }

    void run()
    {
        // Seed the random number generators
        torch::manual_seed(std::chrono::system_clock::now().time_since_epoch().count());

        // Create the stream to write the output
        bool verify_numa = numa_verification_method == "strict";
        bool log_mode = log_method == "full";
        bool same_core = core_mapping_method == "same";
        std::ofstream output_file(output_file_path);
        if(!output_file.is_open()) {
            throw std::runtime_error("Unable to open file " + output_file_path);
        }
        output_file << "num_centroids,scanned_partition_bytes,percentage_clusters,num_threads,method,latency_summary,throughput_summary" << std::endl;

        for(int num_centroids = num_centroids_min; num_centroids <= num_centroids_max; num_centroids *= 2) {
            // Read the index from one of these files
            srand(time(NULL));
            std::string index_path = BASE_DIR + "/" + std::to_string(num_centroids) + "/0.index";
            if(log_mode) {
                std::cout << "Using index " << index_path << std::endl;
            }

            for(int percentage_cluster = percent_clusters_min; percentage_cluster <= percent_clusters_max; percentage_cluster *= 2) {
                for (int num_threads = num_threads_min; num_threads <= num_threads_max; num_threads *= 2) {
                    // Run the experiment
                    std::cout << std::endl << "------- START: Num Centroids - " << num_centroids << ", Num Search Vectors - "
                        << num_queries << ", Num Threads - " << num_threads << ", Percentage Search Clusters - " << percentage_cluster 
                        << ", Total Return Vectors - " << total_return_vectors << " -------" << std::endl;

                    run_experiment(index_path, num_centroids, vector_dimension, num_queries, percentage_cluster, total_return_vectors, num_threads, 
                        target_query_latency_us, flush_gap_time_us, output_file, num_iterations, log_mode, verify_numa, same_core);

                    std::cout << "------- END: Num Centroids - " << num_centroids << ", Num Search Vectors - "
                        << num_queries << ", Num Threads - " << num_threads << ", Percentage Search Clusters - " 
                        << percentage_cluster << ", Total Return Vectors - " << total_return_vectors << " -------" << std::endl;
                    
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            }
        }

        output_file.close();
    }
};

int main(int argc, char const *argv[]) {
    #ifdef __linux__
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system." << std::endl;
        exit(1);
    } 
#endif

    args::parse<input_args>(argc, argv);
}