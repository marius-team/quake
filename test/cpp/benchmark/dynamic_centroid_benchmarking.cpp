#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <vector>
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

#include <pthread.h>
#include <numa.h>
#include <numaif.h>

#include <torch/torch.h>
#include "dynamic_centroid_store.h"

using torch::Tensor;
using faiss::idx_t;

int main() {
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system." << std::endl;
        exit(1);
    } else {
        std::cout << "NUMA is available on this system." << std::endl;
    }

    srand(time(NULL));
    int vector_dimension = 128;
    faiss::DynamicCentroidStore centroid_store(vector_dimension);

    // Populate the tensors
    int centroids_to_far = 0;
    int num_iterations = 5;
    int vectors_to_insert = 500;
    for(int i = 0; i < num_iterations; i++) {
        // Add in some vectors into the index
        int num_tensors_in_index = rand() % vectors_to_insert;
        Tensor insert_vectors = torch::randn({num_tensors_in_index, vector_dimension}).contiguous();
        Tensor insert_ids = torch::arange(centroids_to_far, centroids_to_far + num_tensors_in_index, torch::kInt64).contiguous();
        centroid_store.add_centroids(num_tensors_in_index, insert_vectors.data_ptr<float>(), insert_ids.data_ptr<idx_t>());

        // Remove some tensors from the index
        int tensors_to_remove = num_tensors_in_index/3;
        std::set<idx_t> ids_to_remove;
        for(idx_t i = centroids_to_far + tensors_to_remove; i < centroids_to_far + 2 * tensors_to_remove; i++) {
            ids_to_remove.insert(i);
        }
        centroid_store.remove_centroids(ids_to_remove);

        centroids_to_far += num_tensors_in_index;
    }

    // Distribute the clusters
    int num_numa_nodes = numa_max_node() + 1;
    int num_workers = 16;
    int worker_numa_nodes[num_workers];
    for(int i = 0; i < num_workers; i++) {
        worker_numa_nodes[i] = i % num_numa_nodes;
    }
    centroid_store.distribute_centroids(num_workers, worker_numa_nodes, false);

    // Add in some vectors into the index
    int num_tensors_in_index = rand() % vectors_to_insert;
    Tensor insert_vectors = torch::randn({num_tensors_in_index, vector_dimension}).contiguous();
    Tensor insert_ids = torch::arange(centroids_to_far, centroids_to_far + num_tensors_in_index, torch::kInt64).contiguous();
    centroid_store.add_centroids(num_tensors_in_index, insert_vectors.data_ptr<float>(), insert_ids.data_ptr<idx_t>());

    // Remove some tensors from the index
    int tensors_to_remove = num_tensors_in_index/3;
    std::set<idx_t> ids_to_remove;
    for(idx_t i = centroids_to_far + tensors_to_remove; i < centroids_to_far + 2 * tensors_to_remove; i++) {
        ids_to_remove.insert(i);
    }
    centroid_store.remove_centroids(ids_to_remove);

    for(int i = 0; i < num_workers; i++) {
        std::cout << "Worker " << i << " has " << centroid_store.num_centroids_for_worker(i) << " centroids" << std::endl;
    }
}