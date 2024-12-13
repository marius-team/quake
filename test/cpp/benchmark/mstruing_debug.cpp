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

std::shared_ptr<DynamicIVF_C> load_index(std::string index_path, int num_partitions) {
    // Load the actual index
    int vector_dimension = 100;
    faiss::MetricType metric_type = faiss::METRIC_L2;
    auto index = std::make_shared<DynamicIVF_C>(vector_dimension, num_partitions, metric_type, 32, -1, -1, true, false, false, true, true);
    index->load(index_path, true);

    // set maintenance policy
    MaintenancePolicyParams params;
    params.window_size = 9500;
    params.refinement_radius = 50;
    params.alpha = .75;
    params.enable_split_rejection = true;
    params.enable_delete_rejection = true;
    params.delete_threshold_ns = 10;
    params.split_threshold_ns = 1000;
    index->maintenance_policy_->set_params(params);

    return index;
}

int main() {
    // Load the index
    int vector_dimension = 100;
    int num_partitions = 6449;
    int batch_offset = 4;
    std::string index_path = "/working_dir/compass_datasets/readonly_maintenance_msturing10m/indexes/DynamicIVF_post_mainteance_4_6449.index";
    auto index = load_index(index_path, num_partitions);
    std::cout << "Loaded index from " << index_path << std::endl;

    // Wait for the indexes to be ready
    while(!index->index_ready()) {
        std::cout << "Sleeping due to the index not being ready" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Index is ready" << std::endl;

    // Run some query batches with mainteance
    int nprobe = 640;
    int num_queries = 1000;
    int query_batches = 25;
    int k = 100;
    for(int i = 0; i < query_batches; i++) {
        // First run some queries
        std::cout << "Starting batch " << i << std::endl;
        for(int j = 0; j < num_queries; j++) {
            Tensor curr_query = torch::randn({1, vector_dimension}).contiguous();
            auto result = index->search(curr_query, nprobe, k);
        }
        std::cout << "Finished running " << num_queries << " queries" << std::endl;

        // Now run some mainteance and save the index
        std::cout << "Starting maintenance" << std::endl;
        index->maintenance_policy_->maintenance();
        int post_nlist = index->nlist();
        std::cout << "Finished batch " << i << " index has " << post_nlist << " partitions" << std::endl;

        // Save and reload the index
        int curr_shifted_batch = batch_offset + i + 1;
        std::string save_path = "/working_dir/compass_datasets/readonly_maintenance_msturing10m/indexes/DynamicIVF_post_mainteance_" + std::to_string(curr_shifted_batch) + "_" + std::to_string(post_nlist) + ".index";
        index->save(save_path);
        std::cout << "Saved index to " << save_path << std::endl;

        index = load_index(save_path, post_nlist);
        std::cout << "Reloaded index from " << save_path << std::endl;
    }
}