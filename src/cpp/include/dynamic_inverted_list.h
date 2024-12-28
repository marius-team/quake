//
// Created by Jason on 9/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef DYNAMIC_INVERTED_LIST_H
#define DYNAMIC_INVERTED_LIST_H

#include <common.h>
#include <faiss/invlists/InvertedLists.h>
#include <index_partition.h>

namespace faiss {
    /**
     * @brief InvertedLists implementation using std::unordered_map.
     *
     * This class stores codes and IDs for each list in a map of IndexPartition objects.
     */
    class DynamicInvertedLists : public InvertedLists {
    public:

        int curr_list_id_ = 0;
        int total_numa_nodes_ = 0;
        int next_numa_node_ = 0;
        std::unordered_map<size_t, IndexPartition> partitions_;
        int d_;

        /**
         * @brief Constructor for DynamicInvertedLists.
         *
         * @param nlist          Number of lists (partitions).
         * @param code_size      Size of each code in bytes.
         * @param use_map_for_ids (Currently unused, kept for compatibility)
         */
        DynamicInvertedLists(size_t nlist, size_t code_size, bool use_map_for_ids = false);

        ~DynamicInvertedLists() override;

        size_t list_size(size_t list_no) const override;

        const uint8_t* get_codes(size_t list_no) const override;

        const idx_t* get_ids(size_t list_no) const override;

        void release_codes(size_t list_no, const uint8_t *codes) const override;

        void release_ids(size_t list_no, const idx_t *ids) const override;

        void remove_entry(size_t list_no, idx_t id);

        void remove_entries_from_partition(size_t list_no, std::set<idx_t> vectors_to_remove);

        void remove_vectors(std::set<idx_t> vectors_to_remove);

        size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t *ids,
            const uint8_t *codes) override;

        void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t *ids,
            const uint8_t *codes) override;

        void batch_update_entries(
            size_t old_vector_partition,
            int64_t* new_vector_partitions,
            uint8_t* new_vectors,
            int64_t* new_vector_ids,
            int num_vectors);

        void remove_list(size_t list_no);

        void add_list(size_t list_no);

        bool id_in_list(size_t list_no, idx_t id) const;

        bool get_vector_for_id(idx_t id, float* vector_values);

        size_t get_new_list_id();

        void reset() override;

        void resize(size_t nlist, size_t code_size) override;

        void set_numa_details(int num_numa_nodes, int next_numa_node);

        int get_numa_node(size_t list_no);

        // Set the NUMA node for this partition
        void set_numa_node(size_t list_no, int new_numa_node, bool interleaved = false);

        // Returns the cluster IDs currently unassigned to a NUMA node
        std::set<size_t> get_unassigned_clusters();

        // Get/set the thread mapped to this partition
        int get_thread(size_t list_no);
        void set_thread(size_t list_no, int new_thread_id);

        void save(const std::string &path);

        void load(const std::string &path);

        Tensor get_partition_ids();
    };

    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists, std::unordered_map<size_t, size_t>& remap_ids);
    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists);
} // namespace faiss

#endif //DYNAMIC_INVERTED_LIST_H