//
// Created by Jason on 9/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef DYNAMIC_INVERTED_LIST_H
#define DYNAMIC_INVERTED_LIST_H

#include <faiss/invlists/InvertedLists.h>
#include <unordered_map>
#include <vector>
#include <set>
#include <faiss/invlists/InvertedListsIOHook.h>

using std::size_t;

namespace faiss {
    /**
     * @brief InvertedLists implementation using std::unordered_map.
     *
     * This class stores the codes and IDs for each list in unordered_maps.
     */
    class DynamicInvertedLists : public InvertedLists {
    public:
        /**
         * @brief Constructor for UnorderedMapInvertedLists.
         *
         * @param nlist      Number of lists (partitions).
         * @param code_size  Size of each code in bytes.
         * @param use_map_for_ids Whether we should use to a map to determine the location of the map (optional)
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

        int curr_list_id_ = 0;
#ifdef __linux__
        int total_numa_nodes_;
        int next_numa_node_;

        void set_numa_details(int num_numa_nodes, int next_numa_node);

        int get_numa_node(size_t list_no);

        // This can be called before adding/updating the entry and the numa node will get assigned to that node
        void set_numa_node(size_t list_no, int new_numa_node, bool interleaved = false);

        // Returns the cluster in the current inverted list that are not assigned to a numa node
        std::set<size_t> get_unassigned_clusters();
        
        // Get the thread that this cluster is mapped to
        int get_thread(size_t list_no);

        // Record that the cluster is mapped to this thread
        void set_thread(size_t list_no, int new_thread_id);

        std::unordered_map<size_t, int> curr_thread_;
        std::set<size_t> unassigned_clusters_;
#endif

        std::unordered_map<size_t, int> curr_numa_node_;
        std::unordered_map<size_t, size_t> curr_buffer_sizes_;
        std::unordered_map<size_t, size_t> num_vectors_;
        std::unordered_map<size_t, uint8_t*> codes_;
        std::unordered_map<size_t, idx_t*> ids_;
    };

    ArrayInvertedLists *convert_to_array_invlists(DynamicInvertedLists *invlists, std::unordered_map<size_t, size_t>& remap_ids);

    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists);
} // namespace faiss

#endif //DYNAMIC_INVERTED_LIST_H
