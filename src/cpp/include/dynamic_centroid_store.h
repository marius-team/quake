
#ifndef DYNAMIC_CENTROID_STORE_H
#define DYNAMIC_CENTROID_STORE_H

#include <faiss/invlists/InvertedLists.h>
#include <unordered_map>
#include <vector>
#include <set>
#include <tuple>
#include "faiss/impl/platform_macros.h"

using std::size_t;

namespace faiss {
    class DynamicCentroidStore {
    public:
        DynamicCentroidStore(int vector_dimension, bool log_mode = false);

        ~DynamicCentroidStore();

        void add_centroids(int num_centroids, float* centroid_vectors, idx_t* centroid_ids);

        void remove_centroids(std::set<idx_t> ids_to_remove);

        void distribute_centroids(int num_workers, int* worker_numa_nodes, bool using_numa = true);

        float* get_vectors_for_worker(int worker_id);

        idx_t* get_ids_for_worker(int worker_id);

        int num_centroids_for_worker(int worker_id);

        bool get_vector_for_id(idx_t id, float* vector_values);

        std::vector<idx_t> get_all_ids();

        int total_centroids_;

    private:
        int vector_dimension_;
        std::vector<float> unassigned_vectors_;
        std::vector<idx_t> unassigned_ids_;
        bool initialized_workers_;
        bool using_numa_;
        bool log_mode_;

        std::unordered_map<int, int> curr_buffer_sizes_;
        std::unordered_map<int, int> curr_numa_node_;
        std::unordered_map<int, int> centroids_per_worker_;
        std::unordered_map<int, float*> vectors_per_worker_;
        std::unordered_map<int, idx_t*> ids_per_worker_;

        void print_vectors();
    }; 
    
}

#endif // DYNAMIC_CENTROID_STORE_H