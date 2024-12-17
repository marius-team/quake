// dynamic_inverted_list.cpp

#include "dynamic_inverted_list.h"

namespace faiss {
    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists, std::unordered_map<size_t, size_t>& remap_ids) {
        auto ret = new ArrayInvertedLists(invlists->nlist, invlists->code_size);

        // iterate over all lists in the unordered_map
        size_t new_list_no = 0;
        for (auto &entry: invlists->ids_) {
            size_t old_list_no = entry.first;
            auto &ids = entry.second;
            auto &codes = invlists->codes_[old_list_no];
            size_t num_vectors = invlists->num_vectors_[old_list_no];
            ret->add_entries(new_list_no, num_vectors, ids, codes);
            remap_ids[old_list_no] = new_list_no;
            new_list_no += 1;
        }

        return ret;
    }

    DynamicInvertedLists* convert_from_array_invlists(ArrayInvertedLists *invlists) {
        auto ret = new DynamicInvertedLists(invlists->nlist, invlists->code_size);
        
        for (size_t list_no = 0; list_no < invlists->nlist; list_no++) {
            auto &ids = invlists->ids[list_no];
            auto &codes = invlists->codes[list_no];

            if (ids.size() > 0) {
                ret->add_entries(list_no, invlists->list_size(list_no), invlists->get_ids(list_no), invlists->get_codes(list_no));
            }
        }

        return ret;
    }

    DynamicInvertedLists::DynamicInvertedLists(size_t nlist, size_t code_size, bool use_map_for_ids)
        : InvertedLists(nlist, code_size) {
#ifdef QUAKE_NUMA
        // Initialize as empty lists
        for(size_t i = 0; i < nlist; i++) {
            curr_thread_[i] = -1;
            curr_numa_node_[i] = -1;
            num_vectors_[i] = 0;
            curr_buffer_sizes_[i] = 0;
            codes_[i] = nullptr;
            ids_[i] = nullptr;
            unassigned_clusters_.insert(i);
        }
#else
        // allocate the initial lists
        for (size_t i = 0; i < nlist; i++) {
            codes_[i] = nullptr;
            ids_[i] = nullptr;
            num_vectors_[i] = 0;
            curr_buffer_sizes_[i] = 0;
        }
#endif
        curr_list_id_ = nlist;
    } 

    DynamicInvertedLists::~DynamicInvertedLists() {
#ifdef QUAKE_NUMA
        // Free all allocated buffers
        for(size_t i = 0; i < nlist; i++) {
            if(codes_[i] == nullptr) continue;

            if(curr_numa_node_[i] == -1) {
                std::free(codes_[i]);
                std::free(ids_[i]);
            } else {
                size_t vectors_in_cluster = curr_buffer_sizes_[i];
                numa_free(codes_[i], code_size * vectors_in_cluster * sizeof(uint8_t));
                numa_free(ids_[i], vectors_in_cluster * sizeof(idx_t));
            }
        }
#else
        // Destructor: Unordered maps will clean up themselves
#endif
    }

    size_t DynamicInvertedLists::list_size(size_t list_no) const {
        if(num_vectors_.find(list_no) != num_vectors_.end()) {
            return num_vectors_.at(list_no);
        } else {
            std::cout << "List Size: " << list_no << std::endl;
            throw std::runtime_error("List does not exist in list_size");
            return 0;
        }
    }

    const uint8_t *DynamicInvertedLists::get_codes(size_t list_no) const {
        if(codes_.find(list_no) != codes_.end()) {
            return codes_.at(list_no);
        } else {
            throw std::runtime_error("List does not exist in get_codes");
            return 0;
        }
    }

    const idx_t *DynamicInvertedLists::get_ids(size_t list_no) const {
        if(ids_.find(list_no) != ids_.end()) {
            return ids_.at(list_no);
        } else {
            throw std::runtime_error("List does not exist in get_ids");
            return 0;
        }
    }

    void DynamicInvertedLists::release_codes(size_t list_no, const uint8_t *codes) const {
        // No action needed because get_codes does not allocate new memory
    }

    void DynamicInvertedLists::release_ids(size_t list_no, const idx_t *ids) const {
        // No action needed because get_ids does not allocate new memory
    }

    void DynamicInvertedLists::remove_entry(size_t list_no, idx_t id) {
        // Ensure the list exists
        if (codes_.find(list_no) == codes_.end()) {
            throw std::runtime_error("List does not exist in remove_entry");
        }

        // Get the metadata for this partition
        int curr_num_vectors = static_cast<int>(num_vectors_[list_no]);
        if(curr_num_vectors == 0) {
            return;
        }
        uint8_t* curr_codes = codes_[list_no];
        idx_t* curr_ids = ids_[list_no];

        // Iterate through performing the remove
        int swap_idx = curr_num_vectors - 1;
        for(int i = 0; i < curr_num_vectors; i++) {
            if(curr_ids[i] == id) {
                // Move this element to the end
                curr_ids[i] = curr_ids[swap_idx];
                std::memcpy(curr_codes + i * code_size, curr_codes + swap_idx * code_size, code_size);
                swap_idx -= 1;
                break;
            }
        }
        num_vectors_[list_no] = static_cast<size_t>(swap_idx + 1);
    }

    void DynamicInvertedLists::remove_entries_from_partition(size_t list_no, std::set<idx_t> vectors_to_remove) {
        // Verify that the list valid
        if(ids_.find(list_no) == ids_.end() || codes_.find(list_no) == codes_.end() || num_vectors_.find(list_no) == num_vectors_.end()) {
            throw std::runtime_error("List does not exist in remove_entries_from_partition");
        }

        // Get the metadata for this partition
        int curr_num_vectors = static_cast<int>(num_vectors_[list_no]);
        if(curr_num_vectors == 0) {
            return;
        }
        uint8_t* curr_codes = codes_[list_no];
        idx_t* curr_ids = ids_[list_no];

        // Iterate through performing the removes
        int swap_idx = curr_num_vectors - 1;
        int j = 0;
        size_t vector_size = code_size * sizeof(uint8_t);
        while (swap_idx >= 0 && j <= swap_idx) {
            // See if we have to delete this element
            idx_t curr_element_id = curr_ids[j];
            if(vectors_to_remove.find(curr_element_id) == vectors_to_remove.end()) {
                j++;
                continue;
            }

            // Move the elements to the end
            curr_ids[j] = curr_ids[swap_idx];
            std::memcpy(curr_codes + j * vector_size, curr_codes + swap_idx * vector_size, vector_size);
            swap_idx -= 1;
        }
        num_vectors_[list_no] = static_cast<size_t>(swap_idx + 1);
    }

    void DynamicInvertedLists::remove_vectors(std::set<idx_t> vectors_to_remove) {
        // Get the current partition ids
        size_t curr_worker_ids[ids_.size()];
        int write_offset = 0;
        for(auto& pair : ids_) {
            curr_worker_ids[write_offset] = pair.first;
            write_offset += 1;
        }

        // Iterate through them performing the deletes
        size_t vector_size = code_size * sizeof(uint8_t);
        for(int i = 0; i < write_offset; i++) {
            // Get the metadata for this partition
            size_t curr_partition_id = curr_worker_ids[i];
            int curr_num_vectors = static_cast<int>(num_vectors_[curr_partition_id]);
            if(curr_num_vectors == 0) {
                continue;
            }
        
            uint8_t* curr_codes = codes_[curr_partition_id];
            idx_t* curr_ids = ids_[curr_partition_id];

            // Iterate through performing the removes
            int swap_idx = curr_num_vectors - 1;
            int j = 0;
            while (swap_idx >= 0 && j <= swap_idx) {
                // See if we have to delete this element
                idx_t curr_element_id = curr_ids[j];
                if(vectors_to_remove.find(curr_element_id) == vectors_to_remove.end()) {
                    j++;
                    continue;
                }

                // Move the elements to the end
                curr_ids[j] = curr_ids[swap_idx];
                std::memcpy(curr_codes + j * vector_size, curr_codes + swap_idx * vector_size, vector_size);
                swap_idx -= 1;
            }
            num_vectors_[curr_partition_id] = static_cast<size_t>(swap_idx + 1);
        }
    }

void DynamicInvertedLists::batch_update_entries(
        size_t old_vector_partition,
        int64_t* new_vector_partitions,
        uint8_t* new_vectors,
        int64_t* new_vector_ids,
        int num_vectors)
    {
        if (num_vectors <= 0) {
            // No vectors to update
            return;
        }

        // Step 1: Count new vectors per partition
        std::unordered_map<size_t, size_t> new_vectors_per_partition;
        for(int i = 0; i < num_vectors; i++) {
            size_t curr_new_partition = static_cast<size_t>(new_vector_partitions[i]);
            if(curr_new_partition != old_vector_partition) {
                new_vectors_per_partition[curr_new_partition]++;
            }
        }

        // Step 2: Allocate/Reallocate buffers for each affected partition
        for(auto& pair : new_vectors_per_partition) {
            size_t curr_partition_id = pair.first;
            size_t vectors_to_add = pair.second;
            size_t existing_vectors = 0;

            // Retrieve existing vector count safely
            auto it = num_vectors_.find(curr_partition_id);
            if(it != num_vectors_.end()) {
                existing_vectors = it->second;
            } else {
                // If the partition does not exist, initialize it
                existing_vectors = 0;
                num_vectors_[curr_partition_id] = 0;
            }

            size_t new_total_vectors = existing_vectors + vectors_to_add;

#ifdef QUAKE_NUMA
            int list_numa_node = -1;
            auto numa_it = curr_numa_node_.find(curr_partition_id);
            if(numa_it != curr_numa_node_.end()) {
                list_numa_node = numa_it->second;
            }

            if(list_numa_node == -1) {
                // Non-NUMA allocation
                if(codes_[curr_partition_id] == nullptr) {
                    // Initial allocation
                    codes_[curr_partition_id] = reinterpret_cast<uint8_t*>(
                        std::malloc(new_total_vectors * code_size * sizeof(uint8_t))
                    );
                    if (codes_[curr_partition_id] == nullptr) {
                        throw std::bad_alloc();
                    }
                    ids_[curr_partition_id] = reinterpret_cast<idx_t*>(
                        std::malloc(new_total_vectors * sizeof(idx_t))
                    );
                    if (ids_[curr_partition_id] == nullptr) {
                        std::free(codes_[curr_partition_id]);
                        throw std::bad_alloc();
                    }
                } else {
                    // Reallocate existing buffers
                    uint8_t* temp_codes = reinterpret_cast<uint8_t*>(
                        std::realloc(codes_[curr_partition_id], new_total_vectors * code_size * sizeof(uint8_t))
                    );
                    if (temp_codes == nullptr) {
                        throw std::bad_alloc();
                    }
                    codes_[curr_partition_id] = temp_codes;

                    idx_t* temp_ids = reinterpret_cast<idx_t*>(
                        std::realloc(ids_[curr_partition_id], new_total_vectors * sizeof(idx_t))
                    );
                    if (temp_ids == nullptr) {
                        throw std::bad_alloc();
                    }
                    ids_[curr_partition_id] = temp_ids;
                }
            } else {
                // NUMA-aware allocation
                size_t prev_buffer_size = 0;
                auto buffer_it = curr_buffer_sizes_.find(curr_partition_id);
                if(buffer_it != curr_buffer_sizes_.end()) {
                    prev_buffer_size = buffer_it->second;
                }

                size_t new_buffer_size = std::max(prev_buffer_size, static_cast<size_t>(1024));
                while(new_buffer_size < new_total_vectors) {
                    new_buffer_size *= 2;
                }

                if(new_buffer_size > prev_buffer_size) {
                    // Allocate new NUMA buffers
                    uint8_t* new_codes = reinterpret_cast<uint8_t*>(
                        numa_alloc_onnode(new_buffer_size * code_size * sizeof(uint8_t), list_numa_node)
                    );
                    if(new_codes == nullptr) {
                        throw std::runtime_error("NUMA allocation failed for codes.");
                    }
                    idx_t* new_ids = reinterpret_cast<idx_t*>(
                        numa_alloc_onnode(new_buffer_size * sizeof(idx_t), list_numa_node)
                    );
                    if(new_ids == nullptr) {
                        numa_free(new_codes, new_buffer_size * code_size * sizeof(uint8_t));
                        throw std::runtime_error("NUMA allocation failed for ids.");
                    }

                    // Copy existing data to new buffers
                    if(codes_[curr_partition_id] != nullptr && existing_vectors > 0) {
                        std::memcpy(new_codes, codes_[curr_partition_id], existing_vectors * code_size * sizeof(uint8_t));
                        std::memcpy(new_ids, ids_[curr_partition_id], existing_vectors * sizeof(idx_t));

                        // Free old buffers
                        numa_free(codes_[curr_partition_id], prev_buffer_size * code_size * sizeof(uint8_t));
                        numa_free(ids_[curr_partition_id], prev_buffer_size * sizeof(idx_t));
                    }

                    // Update buffer pointers and sizes
                    codes_[curr_partition_id] = new_codes;
                    ids_[curr_partition_id] = new_ids;
                    curr_buffer_sizes_[curr_partition_id] = new_buffer_size;
                }
            }
#else
            // Non-NUMA allocation
            if(codes_[curr_partition_id] == nullptr) {
                // Initial allocation
                codes_[curr_partition_id] = reinterpret_cast<uint8_t*>(
                    std::malloc(new_total_vectors * code_size * sizeof(uint8_t))
                );
                if (codes_[curr_partition_id] == nullptr) {
                    throw std::bad_alloc();
                }
                ids_[curr_partition_id] = reinterpret_cast<idx_t*>(
                    std::malloc(new_total_vectors * sizeof(idx_t))
                );
                if (ids_[curr_partition_id] == nullptr) {
                    std::free(codes_[curr_partition_id]);
                    throw std::bad_alloc();
                }
            } else {
                // Reallocate existing buffers
                uint8_t* temp_codes = reinterpret_cast<uint8_t*>(
                    std::realloc(codes_[curr_partition_id], new_total_vectors * code_size * sizeof(uint8_t))
                );
                if (temp_codes == nullptr) {
                    throw std::bad_alloc();
                }
                codes_[curr_partition_id] = temp_codes;

                idx_t* temp_ids = reinterpret_cast<idx_t*>(
                    std::realloc(ids_[curr_partition_id], new_total_vectors * sizeof(idx_t))
                );
                if (temp_ids == nullptr) {
                    throw std::bad_alloc();
                }
                ids_[curr_partition_id] = temp_ids;
            }
#endif

            // Note: Do NOT update num_vectors_ here. It will be updated after writing.
        }

        // Step 3: Prepare write indices
        // Initialize write indices to existing vectors (before addition)
        std::unordered_map<size_t, size_t> write_indices;
        for(auto& pair : new_vectors_per_partition) {
            size_t curr_partition_id = pair.first;
            size_t vectors_to_add = pair.second;

            // Retrieve existing vector count
            size_t existing_vectors = 0;
            auto it = num_vectors_.find(curr_partition_id);
            if(it != num_vectors_.end()) {
                existing_vectors = it->second;
            }

            write_indices[curr_partition_id] = existing_vectors;
        }

        // Step 4: Add new entries
        for(int i = 0; i < num_vectors; i++) {
            if(old_vector_partition == new_vector_partitions[i]) {
                continue; // No update needed for this vector
            }

            size_t curr_new_partition = static_cast<size_t>(new_vector_partitions[i]);

            // Ensure the partition exists
            if (codes_.find(curr_new_partition) == codes_.end() || ids_.find(curr_new_partition) == ids_.end()) {
                throw std::runtime_error("Target partition does not exist during batch_update_entries.");
            }

            // Get the current write index
            size_t write_idx = write_indices[curr_new_partition];
            size_t allocated_vectors = 0;

            // Retrieve allocated vectors (new_total_vectors = existing + vectors_to_add)
            auto it = num_vectors_.find(curr_new_partition);
            if(it != num_vectors_.end()) {
                allocated_vectors = it->second;
            } else {
                throw std::runtime_error("Partition not found in num_vectors_ during writing.");
            }

            // Safety check: Ensure we do not exceed allocated space
            if(write_idx >= allocated_vectors) {
                throw std::runtime_error("Write index exceeds allocated buffer size.");
            }

            // Add the new vector ID
            ids_[curr_new_partition][write_idx] = static_cast<idx_t>(new_vector_ids[i]);

            // Add the new vector codes
            std::memcpy(
                codes_[curr_new_partition] + write_idx * code_size,
                new_vectors + i * code_size,
                code_size * sizeof(uint8_t)
            );

            // Update the write index
            write_indices[curr_new_partition]++;

            // Optionally, you can add additional safety checks here
            // For example, ensure that write_indices[curr_new_partition] does not exceed allocated_vectors
            if(write_indices[curr_new_partition] > allocated_vectors) {
                throw std::runtime_error("Write index exceeded allocated buffer during writing.");
            }
        }

        // Step 5: Update num_vectors_ after successful writing
        for(auto& pair : new_vectors_per_partition) {
            size_t curr_partition_id = pair.first;
            size_t vectors_added = pair.second;

            // Retrieve existing vector count before addition
            size_t existing_vectors = 0;
            auto it = num_vectors_.find(curr_partition_id);
            if(it != num_vectors_.end()) {
                existing_vectors = it->second - vectors_added;
            } else {
                throw std::runtime_error("Partition not found in num_vectors_ during final update.");
            }

            // Update the total number of vectors
            num_vectors_[curr_partition_id] = existing_vectors + vectors_added;
        }

        // Optionally, you can add final assertions to ensure consistency
        for(auto& pair : new_vectors_per_partition) {
            size_t curr_partition_id = pair.first;
            size_t vectors_to_add = pair.second;
            size_t allocated_vectors = num_vectors_[curr_partition_id];
            size_t final_count = write_indices[curr_partition_id];

            assert(final_count == allocated_vectors && "Final count does not match allocated vectors after batch update.");
        }
    }

    size_t DynamicInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes) {
        
        if (n_entry == 0) {
            throw std::runtime_error("n_entry is 0 in add_entries");
            return 0;
        }

        // Ensure the list exists
        if (codes_.find(list_no) == codes_.end()) {
            throw std::runtime_error("List does not exist in add_entries");
        }

#ifdef QUAKE_NUMA
        // First determine if this node is handled using numa or not
        int list_numa_node = curr_numa_node_[list_no];
        size_t num_existing_vectors = num_vectors_[list_no];
        size_t num_total_vectors = num_existing_vectors + n_entry;
        if(list_numa_node == -1) {
            if(codes_[list_no] == nullptr) {
                // Allocate the buffer for the first time
                codes_[list_no] = reinterpret_cast<uint8_t*>(std::malloc(num_total_vectors * code_size * sizeof(uint8_t)));
                ids_[list_no] = reinterpret_cast<idx_t*>(std::malloc(num_total_vectors * sizeof(idx_t)));
            } else {
                // Perform a realloc of the buffer to the new size
                codes_[list_no] = reinterpret_cast<uint8_t*>(std::realloc(codes_[list_no], num_total_vectors * code_size * sizeof(uint8_t)));
                ids_[list_no] = reinterpret_cast<idx_t*>(std::realloc(ids_[list_no], num_total_vectors * sizeof(idx_t)));
            }
        } else {
            size_t prev_buffer_size = curr_buffer_sizes_[list_no];
            size_t new_buffer_size = std::max(prev_buffer_size, static_cast<size_t>(1024));
            while(new_buffer_size <= num_total_vectors) {
                new_buffer_size *= 2;
            }

            if(new_buffer_size > prev_buffer_size) {
                if(codes_[list_no] == nullptr) {
                    // Allocate the buffer for the first time on the specified numa node
                    codes_[list_no] = reinterpret_cast<uint8_t*>(numa_alloc_onnode(new_buffer_size * code_size * sizeof(uint8_t), list_numa_node));
                    ids_[list_no] = reinterpret_cast<idx_t*>(numa_alloc_onnode(new_buffer_size * sizeof(idx_t), list_numa_node));
                } else {
                    // Realloc the codes
                    uint8_t* prev_codes = codes_[list_no];
                    codes_[list_no] = reinterpret_cast<uint8_t*>(numa_alloc_onnode(new_buffer_size * code_size * sizeof(uint8_t), list_numa_node));
                    std::memcpy(codes_[list_no], prev_codes, prev_buffer_size * code_size * sizeof(uint8_t));
                    numa_free(prev_codes, prev_buffer_size * code_size * sizeof(uint8_t));
                    
                    // Realloc the ids
                    idx_t* prev_ids = ids_[list_no];
                    ids_[list_no] = reinterpret_cast<idx_t*>(numa_alloc_onnode(new_buffer_size * sizeof(idx_t), list_numa_node));
                    std::memcpy(ids_[list_no], prev_ids, prev_buffer_size * sizeof(idx_t));
                    numa_free(prev_ids, prev_buffer_size * sizeof(idx_t));
                }

                curr_buffer_sizes_[list_no] = new_buffer_size;
            }
        }

        // Ensure that the buffers were updated properly
        if(codes_[list_no] == nullptr || ids_[list_no] == nullptr) {
            throw std::runtime_error("Unable to allocate space for new cluster on specified numa node");
        }

        // Append in the new entries
        std::memcpy(codes_[list_no] + code_size * num_existing_vectors, codes, code_size * n_entry * sizeof(uint8_t));
        std::memcpy(ids_[list_no] + num_existing_vectors, ids, n_entry * sizeof(idx_t));
        
        // Record the new cluster size
        num_vectors_[list_no] = num_total_vectors;
#else
        // First determine if this node is handled using numa or not
        size_t num_existing_vectors = num_vectors_[list_no];
        size_t num_total_vectors = num_existing_vectors + n_entry;
        if(codes_[list_no] == nullptr) {
            // Allocate the buffer for the first time
            codes_[list_no] = reinterpret_cast<uint8_t*>(std::malloc(num_total_vectors * code_size * sizeof(uint8_t)));
            ids_[list_no] = reinterpret_cast<idx_t*>(std::malloc(num_total_vectors * sizeof(idx_t)));
        } else {
            // Perform a realloc of the buffer to the new size
            codes_[list_no] = reinterpret_cast<uint8_t*>(std::realloc(codes_[list_no], num_total_vectors * code_size * sizeof(uint8_t)));
            ids_[list_no] = reinterpret_cast<idx_t*>(std::realloc(ids_[list_no], num_total_vectors * sizeof(idx_t)));
        }

        // Ensure that the buffers were updated properly
        if(codes_[list_no] == NULL || ids_[list_no] == NULL) {
            throw std::runtime_error("Unable to alloc space for new cluster");
        }

        // Append in the new entries
        std::memcpy(codes_[list_no] + code_size * num_existing_vectors, codes, code_size * n_entry * sizeof(uint8_t));
        std::memcpy(ids_[list_no] + num_existing_vectors, ids, n_entry * sizeof(idx_t));

        // Record the new cluster size
        num_vectors_[list_no] = num_total_vectors;
#endif
        return n_entry;
    }

    void DynamicInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes) {

        if(codes_.find(list_no) == codes_.end() || ids_.find(list_no) == ids_.end()) {
            throw std::runtime_error("List does not exist in update_entries");
        }

        if(offset + n_entry > num_vectors_[list_no]) {
            throw std::runtime_error("Offset + n_entry exceeds list size in update_entries");
        }

        // Update the entries
        std::memcpy(codes_[list_no] + code_size * offset, codes, code_size * n_entry * sizeof(uint8_t));
        std::memcpy(ids_[list_no] + offset, ids, n_entry * sizeof(idx_t));
    }

    void DynamicInvertedLists::remove_list(size_t list_no) {
        codes_.erase(list_no);
        ids_.erase(list_no);
        num_vectors_.erase(list_no);
        nlist--;
#ifdef QUAKE_NUMA
        curr_numa_node_.erase(list_no);
        unassigned_clusters_.erase(list_no);
#endif
    }

    void DynamicInvertedLists::add_list(size_t list_no) {
        if (codes_.find(list_no) != codes_.end() || ids_.find(list_no) != ids_.end()) {
            throw std::runtime_error("List already exists in add_list");
        }
#ifdef QUAKE_NUMA
        codes_[list_no] = nullptr;
        ids_[list_no] = nullptr;
        curr_numa_node_[list_no] = -1;
        num_vectors_[list_no] = 0;
        curr_buffer_sizes_[list_no] = 0;
        unassigned_clusters_.insert(list_no);
#endif
        codes_[list_no] = nullptr;
        ids_[list_no] = nullptr;
        num_vectors_[list_no] = 0;
        curr_buffer_sizes_[list_no] = 0;
        nlist++;
    }

    void DynamicInvertedLists::reset() {
        codes_.clear();
        ids_.clear();
#ifdef QUAKE_NUMA
        curr_thread_.clear();
        curr_numa_node_.clear();
#endif
        curr_buffer_sizes_.clear();
        num_vectors_.clear();
        nlist = 0;
        curr_list_id_ = 0;
    }

    void DynamicInvertedLists::resize(size_t nlist, size_t code_size) {
        // Resize is not needed because unordered_maps will resize themselves
    }

    // Method to check if a given id is in a specific list
    bool DynamicInvertedLists::id_in_list(size_t list_no, idx_t id) const {
        if (ids_.find(list_no) == ids_.end()) {
            return false;
        }

        // Get the ids data
        const idx_t* list_ids = ids_.at(list_no);
        const size_t list_num_vectors = num_vectors_.at(list_no);

        // Perform the search
        const idx_t* ids_end_ptr = list_ids + list_num_vectors;
        return std::find(list_ids, ids_end_ptr, id) != ids_end_ptr;
    }

    bool DynamicInvertedLists::get_vector_for_id(idx_t id, float* vector_values) {
        for(auto& curr_partition_ids : ids_) {
            const size_t partition_id = curr_partition_ids.first;
            const idx_t* partition_all_ids = curr_partition_ids.second;
            const size_t partition_num_vectors = num_vectors_.at(partition_id);

            const idx_t* ids_end_ptr = partition_all_ids + partition_num_vectors;
            auto id_ptr = std::find(partition_all_ids, ids_end_ptr, id);
            if(id_ptr != ids_end_ptr) {
                // We found a partition with this id so copy over the associated vector
                int id_index = std::distance(partition_all_ids, id_ptr);
                float* src_vector_vals = reinterpret_cast<float*>(codes_[partition_id] + id_index * code_size);
                std::memcpy(vector_values, src_vector_vals, code_size);
                return true;
            }
        }
        return false;
    }

    size_t DynamicInvertedLists::get_new_list_id() {
        return curr_list_id_++;
    }

#ifdef QUAKE_NUMA
    int DynamicInvertedLists::get_numa_node(size_t list_no) {
        if(curr_numa_node_.find(list_no) != curr_numa_node_.end()) {
            return curr_numa_node_.at(list_no);
        } else {
            throw std::runtime_error("List does not exist in get_numa_node");
            return 0;
        }
    }

    int DynamicInvertedLists::get_thread(size_t list_no) {
        if(curr_thread_.find(list_no) != curr_thread_.end()) {
            return curr_thread_.at(list_no);
        } else {
            throw std::runtime_error("List does not exist in get_numa_node");
            return 0;
        }
    }

    void DynamicInvertedLists::set_thread(size_t list_no, int new_thread_id) {
        // First get the current numa node
        if(curr_thread_.find(list_no) == curr_thread_.end()) {
            throw std::runtime_error("List does not exist in set_thread");
        }
        curr_thread_[list_no] = new_thread_id;
    }

    void DynamicInvertedLists::set_numa_node(size_t list_no, int new_numa_node, bool interleaved) {
        // First get the current numa node
        if(curr_numa_node_.find(list_no) == curr_numa_node_.end() || ids_.find(list_no) == ids_.end() || codes_.find(list_no) == codes_.end()) {
            throw std::runtime_error("List does not exist in set_numa_node");
        }

        // Check if the buffer is already in the same numa node
        if(curr_numa_node_[list_no] == new_numa_node) {
            return;
        }

        // Verify that this is a valid numa node (currently don't support moving from specific to general numa node)
        bool is_valid_numa_node = new_numa_node == -1 || (numa_available() != -1 && new_numa_node <= numa_max_node());
        if(!is_valid_numa_node) {
            throw std::runtime_error("Invalid numa node specified");
        }

        // Get the old numa node and set the new numa node
        int old_numa_node = curr_numa_node_[list_no];
        curr_numa_node_[list_no] = new_numa_node;

        // See if we need to move anything
        if(codes_[list_no] == nullptr || ids_[list_no] == nullptr) {
            return;
        }
        
        // Get the old buffers
        size_t prev_buffer_size = curr_buffer_sizes_[list_no];
        size_t vectors_in_cluster = num_vectors_[list_no];
        uint8_t* old_codes_buffer = codes_[list_no]; 
        idx_t* old_ids_buffer = ids_[list_no];

        // Determine the initial buffer size
        size_t new_buffer_size = std::max(prev_buffer_size, static_cast<size_t>(1024));
        while(new_buffer_size <= vectors_in_cluster) {
            new_buffer_size *= 2;
        }

        // Allocate the new buffers
        if(new_numa_node == -1) {
            codes_[list_no] = reinterpret_cast<uint8_t*>(std::malloc(vectors_in_cluster * code_size * sizeof(uint8_t)));
            ids_[list_no] = reinterpret_cast<idx_t*>(std::malloc(vectors_in_cluster * sizeof(idx_t)));
        } else {
            codes_[list_no] = reinterpret_cast<uint8_t*>(numa_alloc_onnode(new_buffer_size * code_size * sizeof(uint8_t), new_numa_node));
            ids_[list_no] = reinterpret_cast<idx_t*>(numa_alloc_onnode(new_buffer_size * sizeof(idx_t), new_numa_node));
        }

        if(codes_[list_no] == nullptr || ids_[list_no] == nullptr) {
            throw std::runtime_error("Unable to allocate buffers on the new numa node");
        }

        // Copy over the data to the new buffer
        std::memcpy(codes_[list_no], old_codes_buffer, code_size * vectors_in_cluster * sizeof(uint8_t));
        std::memcpy(ids_[list_no], old_ids_buffer, vectors_in_cluster * sizeof(idx_t));

        // Release the old buffers
        if(old_numa_node == -1) {
            std::free(old_codes_buffer); 
            std::free(old_ids_buffer);
        } else if(prev_buffer_size > 0) {
            numa_free(old_codes_buffer, prev_buffer_size * code_size * sizeof(uint8_t));
            numa_free(old_ids_buffer, prev_buffer_size * sizeof(idx_t));
        }
        curr_buffer_sizes_[list_no] = new_buffer_size;

        if(new_numa_node != -1 && unassigned_clusters_.find(list_no) != unassigned_clusters_.end()) {
            // Deal with the case that we are setting the node for a previously unassigned cluster
            unassigned_clusters_.erase(list_no);
        } else if(new_numa_node == -1) {
            // Mark that this cluster was marked as unassigned
            unassigned_clusters_.insert(list_no);
        }
    }

    std::set<size_t> DynamicInvertedLists::get_unassigned_clusters() {
        return unassigned_clusters_;
    }

#else
    // Placeholder to add non linux functions
#endif

} // namespace faiss
