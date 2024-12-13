
#include "dynamic_centroid_store.h"

#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

namespace faiss {
    DynamicCentroidStore::DynamicCentroidStore(int vector_dimension, bool log_mode) : 
    vector_dimension_(vector_dimension), initialized_workers_(false), using_numa_(false), 
    total_centroids_(0), log_mode_(log_mode) {

    }

    DynamicCentroidStore::~DynamicCentroidStore() {
        for(auto& worker : centroids_per_worker_) {
            int worker_id = worker.first;
            int num_centroids = worker.second;
            if(num_centroids > 0) {
                if(using_numa_) {
                    #ifdef __linux__
                    numa_free(vectors_per_worker_[worker_id], num_centroids * vector_dimension_ * sizeof(float));
                    numa_free(ids_per_worker_[worker_id], num_centroids * sizeof(idx_t));
                    #endif
                } else {
                    free(vectors_per_worker_[worker_id]);
                    free(ids_per_worker_[worker_id]);
                }
            }
        }
    }

    void DynamicCentroidStore::add_centroids(int num_centroids, float* centroid_vectors, idx_t* centroid_ids) {
        // Create a set of the centroids to actually add
        std::vector<int> new_idxs_to_add;
        for(int i = 0; i < num_centroids; i++) {
            new_idxs_to_add.push_back(i);
        }   
        
        int num_new_added = 0;
        idx_t* last_centroids_ids = centroid_ids + num_centroids;
        if(!initialized_workers_) {
            // First see if we tried to add these centroids previously
            for(int i = 0; i < num_centroids; i++) {
                idx_t curr_partition_id = centroid_ids[i];
                auto search_result = std::find(unassigned_ids_.begin(), unassigned_ids_.end(), curr_partition_id);
                if(search_result != unassigned_ids_.end()) {
                    int unassigned_partition_idx = std::distance(unassigned_ids_.begin(), search_result);
                    std::memcpy(unassigned_vectors_.data() + unassigned_partition_idx * vector_dimension_, centroid_vectors + i * vector_dimension_, vector_dimension_ * sizeof(float));

                    new_idxs_to_add.erase(std::remove(new_idxs_to_add.begin(), new_idxs_to_add.end(), i), new_idxs_to_add.end());
                }
            }

            num_new_added = new_idxs_to_add.size();
            if(num_new_added == 0) {
                return;
            }

            // If the vectors have not been intialized then add to the unassigned vectors
            for(int i = 0; i < num_new_added; i++) {
                int curr_centroid_idx = new_idxs_to_add[i];
                unassigned_vectors_.insert(unassigned_vectors_.end(), centroid_vectors + curr_centroid_idx * vector_dimension_, centroid_vectors + (curr_centroid_idx + 1) * vector_dimension_);
                unassigned_ids_.push_back(centroid_ids[curr_centroid_idx]);
            }
        } else {
            // First see if any of the workers already have some of these centroids
            for(auto& curr_partition_ids : ids_per_worker_) {
                int worker_id = curr_partition_ids.first;
                idx_t* partition_all_ids = curr_partition_ids.second;
                int partition_num_vectors = centroids_per_worker_[worker_id];

                for(int i = 0; i < partition_num_vectors; i++) {
                    // Check if this centroid is the one being added
                    idx_t curr_partition_id = partition_all_ids[i];
                    auto search_result = std::find(centroid_ids, last_centroids_ids, curr_partition_id);
                    if(search_result != last_centroids_ids) {
                        // If so then just override the vector rather than adding it in
                        int centroid_idx = std::distance(centroid_ids, search_result);
                        std::memcpy(vectors_per_worker_[worker_id] + i * vector_dimension_, centroid_vectors + centroid_idx * vector_dimension_, vector_dimension_ * sizeof(float));
                        
                        // Also remove this centroid as a new centroid
                        new_idxs_to_add.erase(std::remove(new_idxs_to_add.begin(), new_idxs_to_add.end(), centroid_idx), new_idxs_to_add.end());
                    }
                } 
            }

            num_new_added = new_idxs_to_add.size();
            if(num_new_added == 0) {
                return;
            }

            // First create the sorted vector of centroids per worker
            std::vector<std::pair<int, int>> sorted_workers;
            int total_centroids = 0;
            for(const auto& pair : centroids_per_worker_) {
                sorted_workers.push_back(pair);
                total_centroids += pair.second;
            }
            std::sort(sorted_workers.begin(), sorted_workers.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

            // Determine how many centroids to assign to each worker
            int num_workers = sorted_workers.size();
            int new_centroids_count = total_centroids + num_new_added;
            int new_centroids_per_worker = (new_centroids_count + num_workers - 1)/num_workers;
            
            // Assign the centroids to worker with less nodes first
            int centroid_read_idx = 0;
            int centroids_remaining = num_new_added;
            int curr_idx = 0;
            while(curr_idx < num_workers && centroids_remaining > 0) {
                // Decide centroids to add to this worker
                std::pair<int, int> curr_worker = sorted_workers[curr_idx];
                int worker_prev_count = curr_worker.second;
                int curr_worker_id = curr_worker.first;
                int centroids_to_add = std::min(new_centroids_per_worker - worker_prev_count, centroids_remaining);
                int new_centroids_count = worker_prev_count + centroids_to_add;
                centroids_per_worker_[curr_worker_id] = new_centroids_count;

                // Determine the updated size
                int prev_buffer_size = curr_buffer_sizes_[curr_worker_id];
                int new_buffer_size = prev_buffer_size;
                while(new_buffer_size <= new_centroids_count) {
                    new_buffer_size *= 2;
                }

                // See if then need to reallocate the buffers
                if(new_buffer_size > prev_buffer_size) {
                    if(using_numa_) {
                        #ifdef __linux__
                        // First allocate the new buffer
                        int worker_numa_node = curr_numa_node_[curr_worker_id];
                        float* old_vector_ptr = vectors_per_worker_[curr_worker_id];
                        vectors_per_worker_[curr_worker_id] = reinterpret_cast<float*>(numa_alloc_onnode(
                            new_buffer_size * vector_dimension_ * sizeof(float),
                            worker_numa_node
                        ));

                        // Now copy from and free the old buffer
                        std::memcpy(vectors_per_worker_[curr_worker_id], old_vector_ptr, prev_buffer_size * vector_dimension_ * sizeof(float));
                        numa_free(old_vector_ptr, prev_buffer_size * vector_dimension_ * sizeof(float));

                        // First allocate the new buffer
                        idx_t* old_ids_ptr = ids_per_worker_[curr_worker_id];
                        ids_per_worker_[curr_worker_id] = reinterpret_cast<idx_t*>(numa_alloc_onnode(
                            new_buffer_size * sizeof(idx_t),
                            worker_numa_node
                        ));
                        

                        // Now copy from and free the old buffer
                        std::memcpy(ids_per_worker_[curr_worker_id], old_ids_ptr, prev_buffer_size * sizeof(idx_t));
                        numa_free(old_ids_ptr, prev_buffer_size * sizeof(idx_t));
                        #endif
                    } else {
                        vectors_per_worker_[curr_worker_id] = (float*) realloc(
                            vectors_per_worker_[curr_worker_id],
                            new_buffer_size * vector_dimension_ * sizeof(float)
                        );

                        ids_per_worker_[curr_worker_id] = (idx_t*) realloc(
                            ids_per_worker_[curr_worker_id],
                            new_buffer_size * sizeof(idx_t)
                        );
                    }

                    if(vectors_per_worker_[curr_worker_id] == nullptr || ids_per_worker_[curr_worker_id] == nullptr) {
                        throw std::runtime_error("Unable to reallocate the vectors during add"); 
                    }
                    curr_buffer_sizes_[curr_worker_id] = new_buffer_size;
                }

                // Copy the vector and ids
                float* curr_worker_vectors = vectors_per_worker_[curr_worker_id];
                idx_t* curr_worker_ids = ids_per_worker_[curr_worker_id];
                for(int i = 0; i < centroids_to_add; i++) {
                    int curr_centroid_idx = new_idxs_to_add[centroid_read_idx];
                    std::memcpy(
                        curr_worker_vectors + (worker_prev_count + i) * vector_dimension_,
                        centroid_vectors + curr_centroid_idx * vector_dimension_,
                        vector_dimension_ * sizeof(float)
                    );
                    curr_worker_ids[worker_prev_count + i] = centroid_ids[curr_centroid_idx];
                    centroid_read_idx += 1;
                }

                // Update the counters
                curr_idx += 1;
                centroids_remaining -= centroids_to_add;
            }
        }

        total_centroids_ += num_new_added;
    }

    void DynamicCentroidStore::remove_centroids(std::set<idx_t> ids_to_remove) {
        int new_count = 0;

        if(!initialized_workers_) {
            // Use a two pointer to remove the elements from the unassigned buffers
            int write_ptr = 0;
            int num_centroids = unassigned_ids_.size();
            for(int read_ptr = 0; read_ptr < num_centroids; read_ptr++) {
                // First check if this centroid needs to be kept
                idx_t curr_id = unassigned_ids_[read_ptr];
                bool keep_centroid = ids_to_remove.find(curr_id) == ids_to_remove.end();
                if(keep_centroid) {
                    // See if we actually need to do a copy
                    if(write_ptr != read_ptr) {
                        unassigned_ids_[write_ptr] = unassigned_ids_[read_ptr];

                        auto vector_write_offset = unassigned_vectors_.begin() + write_ptr * vector_dimension_;
                        auto vector_read_offset = unassigned_vectors_.begin() + read_ptr * vector_dimension_;
                        std::copy( 
                            vector_read_offset,
                            vector_read_offset + vector_dimension_,
                            vector_write_offset
                        );
                    }
                    write_ptr += 1;
                }
            }
            
            unassigned_ids_.resize(write_ptr);
            unassigned_vectors_.resize(write_ptr * vector_dimension_);
            new_count += write_ptr;            
        } else {
            // We need to run the two pointer removal for each workers centroids
            for(const auto& pair : centroids_per_worker_) {
                int curr_worker_id = pair.first;
                int curr_worker_centroids = pair.second;
                if(curr_worker_centroids == 0) {
                    continue;
                }

                // Get the ptrs for this worker
                float* worker_vectors = vectors_per_worker_[curr_worker_id];
                idx_t* worker_ids = ids_per_worker_[curr_worker_id];
                int write_ptr = 0;
                for(int read_ptr = 0; read_ptr < curr_worker_centroids; read_ptr++) {
                    // First check if this centroid needs to be kept
                    idx_t curr_id = worker_ids[read_ptr];
                    bool keep_centroid = ids_to_remove.find(curr_id) == ids_to_remove.end();
                    if(keep_centroid) {
                        if(write_ptr != read_ptr) {
                            worker_ids[write_ptr] = worker_ids[read_ptr];

                            float* worker_write_offset = worker_vectors + write_ptr * vector_dimension_;
                            float* worker_read_offset = worker_vectors + read_ptr * vector_dimension_;
                            std::memcpy(
                                worker_write_offset,
                                worker_read_offset,
                                vector_dimension_ * sizeof(float)
                            );
                        }

                        write_ptr += 1;
                    }
                }

                // Update this count
                new_count += write_ptr;
                centroids_per_worker_[curr_worker_id] = write_ptr;
            }
        }

        total_centroids_ = new_count;
    }

    void DynamicCentroidStore::distribute_centroids(int num_workers, int* worker_numa_nodes, bool using_numa) {
        if(initialized_workers_) {
            throw std::runtime_error("Distribute centroids can only be called once"); 
        }

        int num_centroids = unassigned_ids_.size();
        int centroids_per_worker = (num_centroids + num_workers - 1)/num_workers;
        using_numa_ = using_numa;
        curr_numa_node_.clear();
        for(int i = 0; i < num_workers; i++) {
            // Determine the range that this worker is responsible for
            int curr_worker_numa_node = worker_numa_nodes[i];
            curr_numa_node_[i] = curr_worker_numa_node;
            int worker_start_idx = std::min(i * centroids_per_worker, num_centroids);
            int worker_end_idx = std::min(worker_start_idx + centroids_per_worker, num_centroids);
            int curr_worker_centroids = worker_end_idx - worker_start_idx;

            // Initialize the ptrs for this worker
            centroids_per_worker_[i] = curr_worker_centroids;
            int starting_buffer_size = std::max(curr_worker_centroids, static_cast<int>(1024));
            curr_buffer_sizes_[i] = starting_buffer_size;
            int num_vector_values = curr_worker_centroids * vector_dimension_;

            // Allocate and copy over the vectors
            if(using_numa_) {
                #ifdef __linux__
                vectors_per_worker_[i] = reinterpret_cast<float*>(numa_alloc_onnode(starting_buffer_size * vector_dimension_ * sizeof(float), curr_worker_numa_node));
                #endif
            } else {
                vectors_per_worker_[i] = (float*) malloc(starting_buffer_size * vector_dimension_ * sizeof(float));
            }
            if(vectors_per_worker_[i] == NULL) {
                throw std::runtime_error("Unable to allocate the vectors"); 
            }
            std::memcpy(
                vectors_per_worker_[i], 
                unassigned_vectors_.data() + worker_start_idx * vector_dimension_, 
                num_vector_values * sizeof(float)
            );

            // Do the same for the ids
            if(using_numa_) {
                #ifdef __linux__
                ids_per_worker_[i] = reinterpret_cast<idx_t*>(numa_alloc_onnode(starting_buffer_size * sizeof(idx_t), curr_worker_numa_node));
                #endif
            } else {
                ids_per_worker_[i] = (idx_t*) malloc(starting_buffer_size * sizeof(idx_t));
            }

            if(ids_per_worker_[i] == NULL) {
                throw std::runtime_error("Unable to allocate the vectors"); 
            } 

            std::memcpy(
                ids_per_worker_[i], 
                unassigned_ids_.data() + worker_start_idx, 
                curr_worker_centroids * sizeof(idx_t)
            );
        }

        initialized_workers_ = true;
    }

    float* DynamicCentroidStore::get_vectors_for_worker(int worker_id) {
        return vectors_per_worker_[worker_id];
    }

    idx_t* DynamicCentroidStore::get_ids_for_worker(int worker_id) {
        return ids_per_worker_[worker_id];
    }

    int DynamicCentroidStore::num_centroids_for_worker(int worker_id) {
        return centroids_per_worker_[worker_id];
    }

    bool DynamicCentroidStore::get_vector_for_id(idx_t id, float* vector_values) {
        for(auto& curr_partition_ids : ids_per_worker_) {
            const int partition_id = curr_partition_ids.first;
            const idx_t* partition_all_ids = curr_partition_ids.second;
            const int partition_num_vectors = centroids_per_worker_.at(partition_id); 

            const idx_t* ids_end_ptr = partition_all_ids + partition_num_vectors;
            auto id_ptr = std::find(partition_all_ids, ids_end_ptr, id);
            if(id_ptr != ids_end_ptr) {
                // We found a partition with this id so copy over the associated vector
                int id_index = std::distance(partition_all_ids, id_ptr);
                float* src_vector_vals = vectors_per_worker_[partition_id] + id_index * vector_dimension_;
                std::memcpy(vector_values, src_vector_vals, vector_dimension_ * sizeof(float));
                return true;
            }
        }
        return false;
    }

    std::vector<int64_t> DynamicCentroidStore::get_all_ids() {
        // Allocate the vector with initial capacity
        std::vector<int64_t> all_centroids;
        if(!initialized_workers_) {
            for(int i = 0; i < unassigned_ids_.size(); i++) {
                all_centroids.push_back(static_cast<int64_t>(unassigned_ids_[i]));
            }
            return all_centroids;
        }

        if(total_centroids_ > 0) {
            all_centroids.reserve(total_centroids_);
        }
    
        for(auto& curr_partition_ids : ids_per_worker_) {
            int worker_id = curr_partition_ids.first;
            idx_t* partition_all_ids = curr_partition_ids.second;
            int partition_num_vectors = centroids_per_worker_[worker_id]; 

            for(int i = 0; i < partition_num_vectors; i++) {
                all_centroids.push_back(static_cast<int64_t>(partition_all_ids[i]));
            }
        }

        return all_centroids;
    }
}