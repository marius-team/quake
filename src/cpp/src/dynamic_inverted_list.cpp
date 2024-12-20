// dynamic_inverted_list.cpp

#include "dynamic_inverted_list.h"

namespace faiss {

ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists, std::unordered_map<size_t, size_t>& remap_ids) {
    auto ret = new ArrayInvertedLists(invlists->nlist, invlists->code_size);

    // iterate over all partitions
    size_t new_list_no = 0;
    for (auto &p: invlists->partitions_) {
        size_t old_list_no = p.first;
        const IndexPartition &part = p.second;

        if (part.num_vectors_ > 0) {
            ret->add_entries(new_list_no, part.num_vectors_, part.ids_, part.codes_);
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


DynamicInvertedLists::DynamicInvertedLists(size_t nlist, size_t code_size, bool use_map_for_ids)
    : InvertedLists(nlist, code_size) {
    // Initialize empty partitions
    for (size_t i = 0; i < nlist; i++) {
        IndexPartition ip;
        ip.set_code_size(code_size);
        partitions_[i] = std::move(ip);
    }
    curr_list_id_ = nlist;
}

DynamicInvertedLists::~DynamicInvertedLists() {
    // partitions_ will clean themselves up as IndexPartition destructor frees memory
}

size_t DynamicInvertedLists::list_size(size_t list_no) const {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        std::cerr << "List Size: " << list_no << std::endl;
        throw std::runtime_error("List does not exist in list_size");
    }
    return static_cast<size_t>(it->second.num_vectors_);
}

const uint8_t *DynamicInvertedLists::get_codes(size_t list_no) const {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_codes");
    }
    return it->second.codes_;
}

const idx_t *DynamicInvertedLists::get_ids(size_t list_no) const {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in get_ids");
    }
    return it->second.ids_;
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

    IndexPartition &part = it->second;
    if (part.num_vectors_ == 0) return;

    int64_t idx_to_remove = part.find_id(id);
    if (idx_to_remove != -1) {
        part.remove(idx_to_remove);
    }
}

void DynamicInvertedLists::remove_entries_from_partition(size_t list_no, std::set<idx_t> vectors_to_remove) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in remove_entries_from_partition");
    }
    IndexPartition &part = it->second;

    // We'll perform removals by scanning and removing matches.
    // Because remove() swaps last element in, we must be careful with iteration.
    for (int64_t i = 0; i < part.num_vectors_;) {
        if (vectors_to_remove.find(part.ids_[i]) != vectors_to_remove.end()) {
            part.remove(i);
            // don't increment i, because we just swapped a new element into i
        } else {
            i++;
        }
    }
}

void DynamicInvertedLists::remove_vectors(std::set<idx_t> vectors_to_remove) {
    // Remove from all partitions
    for (auto &kv : partitions_) {
        IndexPartition &part = kv.second;
        for (int64_t i = 0; i < part.num_vectors_;) {
            if (vectors_to_remove.find(part.ids_[i]) != vectors_to_remove.end()) {
                part.remove(i);
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
    const uint8_t *codes) {

    if (n_entry == 0) {
        throw std::runtime_error("n_entry is 0 in add_entries");
    }

    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in add_entries");
    }

    IndexPartition &part = it->second;
    // Ensure code_size is set
    if (part.code_size_ != static_cast<int64_t>(code_size)) {
        part.set_code_size(static_cast<int64_t>(code_size));
    }

    part.append((int64_t)n_entry, ids, codes);
    return n_entry;
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
    IndexPartition &part = it->second;

    part.update((int64_t)offset, (int64_t)n_entry, ids, codes);
}

void DynamicInvertedLists::batch_update_entries(
    size_t old_vector_partition,
    int64_t* new_vector_partitions,
    uint8_t* new_vectors,
    int64_t* new_vector_ids,
    int num_vectors) {

    // This logic will:
    // 1. Remove all vectors from old_vector_partition that moved to a new partition
    // 2. Append them to their new partitions

    // Identify which vectors belong to old_vector_partition and distribute them
    // to new partitions.
    std::unordered_map<size_t, std::vector<int>> vectors_for_new_partition;

    for (int i = 0; i < num_vectors; i++) {
        size_t new_p = static_cast<size_t>(new_vector_partitions[i]);
        if (new_p != old_vector_partition) {
            vectors_for_new_partition[new_p].push_back(i);
        }
    }

    // Append entries to new partitions
    for (auto &kv : vectors_for_new_partition) {
        size_t new_p = kv.first;
        auto it = partitions_.find(new_p);
        if (it == partitions_.end()) {
            // Create a new partition if needed
            add_list(new_p);
            it = partitions_.find(new_p);
        }
        IndexPartition &new_part = it->second;
        if (new_part.code_size_ != static_cast<int64_t>(code_size)) {
            new_part.set_code_size((int64_t)code_size);
        }

        // Gather all IDs and codes to append at once
        std::vector<idx_t> tmp_ids;
        tmp_ids.reserve(kv.second.size());
        std::vector<uint8_t> tmp_codes;
        tmp_codes.reserve(kv.second.size() * code_size);

        for (int idx : kv.second) {
            tmp_ids.push_back((idx_t)new_vector_ids[idx]);
            tmp_codes.insert(tmp_codes.end(),
                             new_vectors + idx * code_size,
                             new_vectors + (idx+1) * code_size);
        }

        new_part.append((int64_t)kv.second.size(), tmp_ids.data(), tmp_codes.data());
    }

    // If needed, remove them from old_vector_partition
    // The problem statement doesn't show how they are originally removed from old_partition,
    // but presumably we should remove them. If old_partition no longer exists or if these
    // vectors are logically moved, remove them from old_vector_partition:
    auto old_it = partitions_.find(old_vector_partition);
    if (old_it != partitions_.end()) {
        IndexPartition &old_part = old_it->second;
        // remove vectors that moved
        for (auto &kv : vectors_for_new_partition) {
            // kv.first = new partition, kv.second = vector indices
            // We must identify vector IDs from old_partition and remove them.
            // However, the provided arrays (new_vectors, new_vector_ids) presumably correspond
            // to vectors that were part of old_vector_partition originally.
            // We can remove by IDs directly:
            for (int idx : kv.second) {
                idx_t old_id = (idx_t)new_vector_ids[idx];
                int64_t pos = old_part.find_id(old_id);
                if (pos != -1) {
                    old_part.remove(pos);
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
    IndexPartition ip;
    ip.set_code_size((int64_t)code_size);
    partitions_[list_no] = std::move(ip);
    nlist++;
}

bool DynamicInvertedLists::id_in_list(size_t list_no, idx_t id) const {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        return false;
    }
    const IndexPartition &part = it->second;
    return part.find_id(id) != -1;
}

bool DynamicInvertedLists::get_vector_for_id(idx_t id, float* vector_values) {
    // This assumes that the stored codes are actually floats or float-like data.
    // If codes_ are PQ codes or compressed, you'd need decompression.
    // If they are raw float vectors, this works.
    // If code_size does not match sizeof(float)*dim, you need to adjust accordingly.

    for (auto &kv : partitions_) {
        const IndexPartition &part = kv.second;
        int64_t pos = part.find_id(id);
        if (pos != -1) {
            // Found it, copy vector
            // code_size_ is in bytes. Assuming float vectors of dimension (code_size_/sizeof(float))
            std::memcpy(vector_values, part.codes_ + pos * part.code_size_, part.code_size_);
            return true;
        }
    }
    return false;
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
    return it->second.numa_node_;
}

void DynamicInvertedLists::set_numa_node(size_t list_no, int new_numa_node, bool interleaved) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_numa_node");
    }
    it->second.set_numa_node(new_numa_node);
}

std::set<size_t> DynamicInvertedLists::get_unassigned_clusters() {
    // Now we need a way to track unassigned clusters.
    // If you consider "unassigned" as numa_node_ = -1:
    std::set<size_t> result;
    for (auto &kv : partitions_) {
        if (kv.second.numa_node_ == -1) {
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
    return it->second.thread_id_;
}

void DynamicInvertedLists::set_thread(size_t list_no, int new_thread_id) {
    auto it = partitions_.find(list_no);
    if (it == partitions_.end()) {
        throw std::runtime_error("List does not exist in set_thread");
    }
    it->second.thread_id_ = new_thread_id;
}

#endif
} // namespace faiss
