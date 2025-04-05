// dynamic_inverted_list.cpp

#include "dynamic_inverted_list.h"
#include <iostream>
#include <fstream>

namespace faiss {
    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists,
                                                  std::unordered_map<size_t, size_t> &remap_ids) {
        auto ret = new ArrayInvertedLists(invlists->nlist, invlists->code_size);

        // iterate over all partitions
        size_t new_list_no = 0;
        for (auto &p: invlists->partitions_) {
            size_t old_list_no = p.first;
            shared_ptr<IndexPartition> part = p.second;

            if (part->num_vectors_ > 0) {
                ret->add_entries(new_list_no, part->num_vectors_, part->ids_, part->codes_);
            }
            remap_ids[old_list_no] = new_list_no;
            new_list_no += 1;
        }

        return ret;
    }

    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists) {
        auto ret = new DynamicInvertedLists(invlists->nlist, invlists->code_size);
        for (size_t list_no = 0; list_no < invlists->nlist; list_no++) {
            size_t list_size = invlists->list_size(list_no);
            if (list_size > 0) {
                ret->add_entries(list_no, list_size, invlists->get_ids(list_no), invlists->get_codes(list_no));
            } else {
                ret->add_list(list_no); // ensure partition exists even if empty
            }
        }
        return ret;
    }


    DynamicInvertedLists::DynamicInvertedLists(size_t nlist, size_t code_size)
        : InvertedLists(nlist, code_size) {
        d_ = code_size / sizeof(float);
        code_size_ = code_size;
        // Initialize empty partitions
        for (size_t i = 0; i < nlist; i++) {
            // IndexPartition ip;
            shared_ptr<IndexPartition> ip = std::make_shared<IndexPartition>();
            ip->set_code_size(code_size);
            partitions_[i] = ip;
        }
        curr_list_id_ = nlist;
    }

    DynamicInvertedLists::~DynamicInvertedLists() {
        // partitions_ will clean themselves up as IndexPartition destructor frees memory
    }

    size_t DynamicInvertedLists::ntotal() const {
        size_t ntotal = 0;
        for (auto &kv: partitions_) {
            ntotal += kv.second->num_vectors_;
        }
        return ntotal;
    }

    size_t DynamicInvertedLists::list_size(size_t list_no) const {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in list_size");
        }
        return static_cast<size_t>(it->second->num_vectors_);
    }

    const uint8_t *DynamicInvertedLists::get_codes(size_t list_no) const {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in get_codes");
        }
        return it->second->codes_;
    }

    const idx_t *DynamicInvertedLists::get_ids(size_t list_no) const {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in get_ids");
        }
        return it->second->ids_;
    }

    void DynamicInvertedLists::release_codes(size_t list_no, const uint8_t *codes) const {
        // No action needed because get_codes does not allocate new memory
    }

    void DynamicInvertedLists::release_ids(size_t list_no, const idx_t *ids) const {
        // No action needed because get_ids does not allocate new memory
    }

    void DynamicInvertedLists::remove_entry(size_t list_no, idx_t id) {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in remove_entry");
        }

        shared_ptr<IndexPartition> part = it->second;
        if (part->num_vectors_ == 0) return;

        int64_t idx_to_remove = part->find_id(id);
        if (idx_to_remove != -1) {
            part->remove(idx_to_remove);
        }
    }

    void DynamicInvertedLists::remove_entries_from_partition(size_t list_no, vector<idx_t> vectors_to_remove) {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in remove_entries_from_partition");
        }
        shared_ptr<IndexPartition> part = it->second;

        // create set from vector for faster lookup
        std::set<idx_t> vectors_to_remove_set(vectors_to_remove.begin(), vectors_to_remove.end());

        // We'll perform removals by scanning and removing matches.
        // Because remove() swaps last element in, we must be careful with iteration.
        for (int64_t i = 0; i < part->num_vectors_;) {
            if (vectors_to_remove_set.find(part->ids_[i]) != vectors_to_remove_set.end()) {
                part->remove(i);
                // don't increment i, because we just swapped a new element into i
            } else {
                i++;
            }
        }
    }

    void DynamicInvertedLists::remove_vectors(std::set<idx_t> vectors_to_remove) {
        // Remove from all partitions
        for (auto &kv: partitions_) {
            shared_ptr<IndexPartition> part = kv.second;
            for (int64_t i = 0; i < part->num_vectors_;) {
                if (vectors_to_remove.find(part->ids_[i]) != vectors_to_remove.end()) {
                    part->remove(i);
                } else {
                    i++;
                }
            }
        }
    }

    size_t DynamicInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes,
        shared_ptr<arrow::Table> attributes_table
    )  {
        if (n_entry == 0) {
            return 0;
        }

        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in add_entries");
        }

        shared_ptr<IndexPartition> part = it->second;
        // Ensure code_size is set
        if (part->code_size_ != static_cast<int64_t>(code_size)) {
            part->set_code_size(static_cast<int64_t>(code_size));
        }

        part->append((int64_t) n_entry, ids, codes, attributes_table);
        return n_entry;
    }

    size_t DynamicInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes
        ) {
        return add_entries(list_no, n_entry, ids, codes, nullptr);
    }

    void DynamicInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t *ids,
        const uint8_t *codes) {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            throw std::runtime_error("List does not exist in update_entries");
        }
        shared_ptr<IndexPartition> part = it->second;

        part->update((int64_t) offset, (int64_t) n_entry, ids, codes);
    }

    void DynamicInvertedLists::batch_update_entries(
        size_t old_vector_partition,
        int64_t *new_vector_partitions,
        uint8_t *new_vectors,
        int64_t *new_vector_ids,
        int num_vectors) {
        // This logic will:
        // 1. Remove all vectors from old_vector_partition that moved to a new partition
        // 2. Append them to their new partitions

        // Identify which vectors belong to old_vector_partition and distribute them
        // to new partitions.
        std::unordered_map<size_t, std::vector<int> > vectors_for_new_partition;

        for (int i = 0; i < num_vectors; i++) {
            size_t new_p = static_cast<size_t>(new_vector_partitions[i]);
            if (new_p != old_vector_partition) {
                vectors_for_new_partition[new_p].push_back(i);
            }
        }

        // Append entries to new partitions
        for (auto &kv: vectors_for_new_partition) {
            size_t new_p = kv.first;
            auto it = partitions_.find(new_p);
            if (it == partitions_.end()) {
                // Create a new partition if needed
                add_list(new_p);
                it = partitions_.find(new_p);
            }
            shared_ptr<IndexPartition> new_part = it->second;
            if (new_part->code_size_ != static_cast<int64_t>(code_size)) {
                new_part->set_code_size((int64_t) code_size);
            }

            // Gather all IDs and codes to append at once
            std::vector<idx_t> tmp_ids;
            tmp_ids.reserve(kv.second.size());
            std::vector<uint8_t> tmp_codes;
            tmp_codes.reserve(kv.second.size() * code_size);

            for (int idx: kv.second) {
                tmp_ids.push_back((idx_t) new_vector_ids[idx]);
                tmp_codes.insert(tmp_codes.end(),
                                 new_vectors + idx * code_size,
                                 new_vectors + (idx + 1) * code_size);
            }

            new_part->append((int64_t) kv.second.size(), tmp_ids.data(), tmp_codes.data());
        }

        // If needed, remove them from old_vector_partition
        auto old_it = partitions_.find(old_vector_partition);
        if (old_it != partitions_.end()) {
            shared_ptr<IndexPartition> old_part = old_it->second;
            // remove vectors that moved
            for (auto &kv: vectors_for_new_partition) {
                for (int idx: kv.second) {
                    idx_t old_id = (idx_t) new_vector_ids[idx];
                    int64_t pos = old_part->find_id(old_id);
                    if (pos != -1) {
                        old_part->remove(pos);
                    }
                }
            }
        }
    }

    void DynamicInvertedLists::remove_list(size_t list_no) {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            // Already doesn't exist
            return;
        }

        partitions_.erase(it);
        nlist--;
    }

    void DynamicInvertedLists::add_list(size_t list_no) {
        if (partitions_.find(list_no) != partitions_.end()) {
            throw std::runtime_error("List already exists in add_list");
        }
        shared_ptr<IndexPartition> ip = std::make_shared<IndexPartition>();
        ip->set_code_size((int64_t) code_size);
        partitions_[list_no] = ip;
        nlist++;
    }

    bool DynamicInvertedLists::id_in_list(size_t list_no, idx_t id) const {
        auto it = partitions_.find(list_no);
        if (it == partitions_.end()) {
            return false;
        }
        shared_ptr<IndexPartition> part = it->second;
        return part->find_id(id) != -1;
    }

    bool DynamicInvertedLists::get_vector_for_id(idx_t id, float *vector_values) {
        for (auto &kv: partitions_) {
            shared_ptr<IndexPartition> part = kv.second;
            int64_t pos = part->find_id(id);
            if (pos != -1) {
                // Found it, copy vector
                // code_size_ is in bytes. Assuming float vectors of dimension (code_size_/sizeof(float))
                std::memcpy(vector_values, part->codes_ + pos * part->code_size_, part->code_size_);
                return true;
            }
        }
        return false;
    }

    vector<float *> DynamicInvertedLists::get_vectors_by_id(vector<int64_t> ids) {

        vector<float *> ret;
        for (int64_t id : ids) {
            bool found = false;
            for (auto &kv: partitions_) {
                shared_ptr<IndexPartition> part = kv.second;
                int64_t pos = part->find_id(id);
                if (pos != -1) {
                    ret.push_back(reinterpret_cast<float *>(part->codes_ + pos * part->code_size_));
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("ID not found in any partition");
            }
        }
        return ret;
    }

    size_t DynamicInvertedLists::get_new_list_id() {
        return curr_list_id_++;
    }

    void DynamicInvertedLists::reset() {
        partitions_.clear();
        nlist = 0;
        curr_list_id_ = 0;
    }

    void DynamicInvertedLists::resize(size_t nlist, size_t code_size) {
        // Not strictly needed because we use a map. But if required,
        // we can add or remove partitions. For now, do nothing.
    }

    void DynamicInvertedLists::save(const string &filename) {
        /**
         * 1) Serialization Format:
         *    - 32-byte header:
         *        [ magic(4) | version(4) | nlist(8) | code_size(8) | num_partitions(8) ]
         *    - Offsets array (num_partitions + 1) of uint64_t
         *    - Partition ID array (num_partitions) of uint64_t
         *    - Concatenated chunks:
         *        For each partition i:
         *          [ codes (num_vectors * code_size) | ids (num_vectors * sizeof(idx_t)) ]
         *      Each chunk starts at offsets[i] (relative to start of chunks), ends at offsets[i+1].
         *
         * 2) Serialization Logic:
         *    - Gather partition IDs in a chosen order
         *    - Build offsets array by writing each partition’s codes/IDs
         *    - Write header
         *    - Write offsets array
         *    - Write partition ID array
         *    - Write partition chunks
         */
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + std::string(filename));
        }

        // Write header
        ofs.write(reinterpret_cast<const char *>(&SerializationMagicNumber), sizeof(SerializationMagicNumber));
        ofs.write(reinterpret_cast<const char *>(&SerializationVersion), sizeof(SerializationVersion));

        uint64_t nlist_64 = static_cast<uint64_t>(nlist);
        uint64_t code_size_64 = static_cast<uint64_t>(code_size);
        uint64_t num_partitions = static_cast<uint64_t>(partitions_.size());

        ofs.write(reinterpret_cast<const char *>(&nlist_64), sizeof(nlist_64));
        ofs.write(reinterpret_cast<const char *>(&code_size_64), sizeof(code_size_64));
        ofs.write(reinterpret_cast<const char *>(&num_partitions), sizeof(num_partitions));

        // Gather partition IDs
        vector<size_t> part_ids = vector<size_t>(partitions_.size());
        int i = 0;
        for (auto &kv: partitions_) {
            part_ids[i++] = kv.first;
        }
        // (Optional) sort(part_ids.begin(), part_ids.end());

        // Prepare offsets
        std::vector<uint64_t> offsets(num_partitions + 1, 0ULL);
        uint64_t offset_table_bytes = (num_partitions + 1) * sizeof(uint64_t);
        uint64_t partition_ids_bytes = num_partitions * sizeof(uint64_t);
        uint64_t start_of_chunks = 32 + offset_table_bytes + partition_ids_bytes;

        // Move file pointer to where chunks begin
        ofs.seekp(start_of_chunks, std::ios::beg);

        uint64_t current_offset = 0;
        for (size_t i = 0; i < num_partitions; i++) {
            offsets[i] = current_offset;
            shared_ptr<IndexPartition> part = partitions_.at(part_ids[i]);

            size_t nv = static_cast<size_t>(part->num_vectors_);
            size_t csize = nv * static_cast<size_t>(part->code_size_);
            size_t isize = nv * sizeof(idx_t);

            ofs.write(reinterpret_cast<const char *>(part->codes_), csize);
            ofs.write(reinterpret_cast<const char *>(part->ids_), isize);

            current_offset += (csize + isize);
        }
        offsets[num_partitions] = current_offset;

        // Go back and write offsets array, then partition ID array
        ofs.seekp(32, std::ios::beg);
        ofs.write(reinterpret_cast<const char *>(offsets.data()),
                  offsets.size() * sizeof(uint64_t));

        for (size_t i = 0; i < num_partitions; i++) {
            uint64_t pid_64 = static_cast<uint64_t>(part_ids[i]);
            ofs.write(reinterpret_cast<const char *>(&pid_64), sizeof(pid_64));
        }

        ofs.close();
    }

    void DynamicInvertedLists::load(const string &filename) {
        /**
         * Deserialization Logic:
         *  - Read header (magic, version, nlist, code_size, num_partitions)
         *  - Read offsets array (num_partitions+1)
         *  - Read partition ID array (num_partitions)
         *  - For each partition i:
         *      chunk_size = offsets[i+1] - offsets[i]
         *      num_vectors = chunk_size / (code_size + sizeof(idx_t))
         *      Seek to start_of_chunks + offsets[i]
         *      Read codes (num_vectors*code_size)
         *      Read ids   (num_vectors*sizeof(idx_t))
         *      Construct IndexPartition and store in partitions_[pid].
         */
        reset();
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + std::string(filename));
        }

        // Read header
        uint32_t file_magic = 0;
        uint32_t file_version = 0;
        ifs.read(reinterpret_cast<char *>(&file_magic), sizeof(file_magic));
        ifs.read(reinterpret_cast<char *>(&file_version), sizeof(file_version));

        if (file_magic != SerializationMagicNumber) {
            throw std::runtime_error("Invalid file format (bad magic number).");
        }
        if (file_version != SerializationVersion) {
            throw std::runtime_error("Unsupported file version: " + std::to_string(file_version));
        }

        uint64_t nlist_64, code_size_64, num_partitions;
        ifs.read(reinterpret_cast<char *>(&nlist_64), sizeof(nlist_64));
        ifs.read(reinterpret_cast<char *>(&code_size_64), sizeof(code_size_64));
        ifs.read(reinterpret_cast<char *>(&num_partitions), sizeof(num_partitions));

        nlist = static_cast<size_t>(nlist_64);
        code_size = static_cast<size_t>(code_size_64);
        d_ = code_size / sizeof(float);

        // Read offsets
        std::vector<uint64_t> offsets(num_partitions + 1);
        ifs.read(reinterpret_cast<char *>(offsets.data()),
                 offsets.size() * sizeof(uint64_t));

        // Read partition IDs
        std::vector<uint64_t> pid_array(num_partitions);
        ifs.read(reinterpret_cast<char *>(pid_array.data()),
                 pid_array.size() * sizeof(uint64_t));

        // Calculate where chunks begin
        uint64_t offset_table_bytes = (num_partitions + 1) * sizeof(uint64_t);
        uint64_t partition_ids_bytes = num_partitions * sizeof(uint64_t);
        uint64_t start_of_chunks = 32 + offset_table_bytes + partition_ids_bytes;

        // Read each partition chunk
        for (uint64_t i = 0; i < num_partitions; i++) {
            size_t pid = static_cast<size_t>(pid_array[i]);

            uint64_t chunk_start = offsets[i];
            uint64_t chunk_end = offsets[i + 1];
            uint64_t chunk_size = chunk_end - chunk_start;

            uint64_t record_size = static_cast<uint64_t>(code_size) + sizeof(idx_t);
            if (chunk_size % record_size != 0) {
                throw std::runtime_error("Partition chunk size not divisible by (code_size+sizeof(idx_t))");
            }
            uint64_t nv64 = chunk_size / record_size; // num_vectors

            ifs.seekg(start_of_chunks + chunk_start, std::ios::beg);

            size_t csize = static_cast<size_t>(nv64) * code_size;
            size_t isize = static_cast<size_t>(nv64) * sizeof(idx_t);
            uint8_t *codes = new uint8_t[csize];
            idx_t *ids = new idx_t[nv64];

            // Read codes and ids from file into allocated buffers
            ifs.read(reinterpret_cast<char*>(codes), csize);
            ifs.read(reinterpret_cast<char*>(ids), isize);

            // IndexPartition part = IndexPartition(nv64, codes, ids, code_size);
            shared_ptr<IndexPartition> part = std::make_shared<IndexPartition>(nv64, codes, ids, code_size);
            partitions_[pid] = part;

            // save to free codes and ids since IndexPartition makes its own copies
            delete[] codes;
            delete[] ids;
        }

        // Update curr_list_id_
        size_t max_list_id = 0;
        for (auto &kv: partitions_) {
            max_list_id = std::max(max_list_id, kv.first);
        }
        curr_list_id_ = max_list_id + 1;

        ifs.close();
    }

    Tensor DynamicInvertedLists::get_partition_ids() {
        // Return a 1D tensor of partition IDs
        Tensor result = torch::empty({(int64_t) partitions_.size()}, torch::kInt64);
        auto result_accessor = result.accessor<int64_t, 1>();
        size_t i = 0;
        for (auto &kv: partitions_) {
            result_accessor[i] = static_cast<int64_t>(kv.first);
            i++;
        }
        return result;
    }

#ifdef QUAKE_USE_NUMA
void DynamicInvertedLists::set_numa_details(int num_numa_nodes, int next_numa_node) {
    total_numa_nodes_ = num_numa_nodes;
    next_numa_node_ = next_numa_node;
}

int DynamicInvertedLists::get_numa_node(size_t list_no) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_numa_node");
    }
    return it->second->numa_node_;
}

void DynamicInvertedLists::set_numa_node(size_t list_no, int new_numa_node, bool interleaved) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_numa_node");
    }
    it->second->set_numa_node(new_numa_node);
}

std::set<size_t> DynamicInvertedLists::get_unassigned_clusters() {
    // Now we need a way to track unassigned clusters.
    // If you consider "unassigned" as numa_node_ = -1:
    std::set<size_t> result;
    for (auto &kv : partitions_) {
        if (kv.second->numa_node_ == -1) {
            result.insert(kv.first);
        }
    }
    return result;
}

int DynamicInvertedLists::get_thread(size_t list_no) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_thread");
    }
    return it->second->core_id_;
}

void DynamicInvertedLists::set_thread(size_t list_no, int new_thread_id) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_thread");
    }
    it->second->core_id_ = new_thread_id;
}

#endif
} // namespace faiss
