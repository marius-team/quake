//
// Created by Jason on 9/11/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LIST_SCANNING_H
#define LIST_SCANNING_H

#include "faiss/index_io.h"
#include "faiss/Clustering.h"
#include <faiss/impl/platform_macros.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexRefine.h>
#include "simsimd/simsimd.h"

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <cassert>
#include <mutex>
#include <chrono>
#include <atomic>

using torch::Tensor;
using std::vector;
using std::shared_ptr;
using std::chrono::high_resolution_clock;

// float DynamicIVF_C::calculate_recall(Tensor ids, Tensor gt_ids) {
//     int64_t num_correct = 0;
//     int num_queries = ids.size(0);
//     int k = ids.size(1);
//
//     int64_t* ids_ptr = ids.data_ptr<int64_t>();
//     int64_t* gt_ids_ptr = gt_ids.data_ptr<int64_t>();
//
//     for (int i = 0; i < num_queries; i++) {
//         std::unordered_set<int64_t> gt_label_set;
//         for (int j = 0; j < k; j++) {
//             gt_label_set.insert(gt_ids_ptr[i * k + j]);
//         }
//         for (int j = 0; j < k; j++) {
//             if (gt_label_set.find(ids_ptr[i * k + j]) != gt_label_set.end()) {
//                 num_correct++;
//             }
//         }
//     }
//     float recall = static_cast<float>(num_correct) / (num_queries * k);
//     return recall;
// }

inline Tensor calculate_recall(Tensor ids, Tensor gt_ids) {
    Tensor num_correct = torch::zeros(ids.size(0), torch::kInt64);
    int num_queries = ids.size(0);
    int k = ids.size(1);

    int64_t* ids_ptr = ids.data_ptr<int64_t>();
    int64_t* gt_ids_ptr = gt_ids.data_ptr<int64_t>();

    for (int i = 0; i < num_queries; i++) {
        std::unordered_set<int64_t> gt_label_set;
        for (int j = 0; j < k; j++) {
            gt_label_set.insert(gt_ids_ptr[i * k + j]);
        }
        for (int j = 0; j < k; j++) {
            if (gt_label_set.find(ids_ptr[i * k + j]) != gt_label_set.end()) {
                num_correct[i]+=1;
            }
        }
    }

    Tensor recall = num_correct.to(torch::kFloat32) / k;

    return recall;
}

#define TOP_K_BUFFER_CAPACITY (1024 * 128)
template<typename DistanceType = float, typename IdType = int>
class TypedTopKBuffer {
public:
    int k_; // Number of top elements to keep
    int curr_offset_ = 0; // Current offset in the buffer
    std::vector<std::pair<DistanceType, IdType>> topk_; // Buffer to store top-k elements
    bool is_descending_; // Flag to indicate sorting order
    std::recursive_mutex buffer_mutex_;
    std::atomic<bool> processing_query_;
    std::atomic<int> jobs_left_;

    TypedTopKBuffer(int k, bool is_descending)
        : k_(k), is_descending_(is_descending), topk_(TOP_K_BUFFER_CAPACITY), processing_query_(true) {
        assert(k <= TOP_K_BUFFER_CAPACITY); // Ensure k is smaller than or equal to buffer size
        
        for (int i = 0; i < topk_.size(); i++) {
            if (is_descending_) {
                topk_[i] = {std::numeric_limits<DistanceType>::min(), -1};
            } else {
                topk_[i] = {std::numeric_limits<DistanceType>::max(), -1};
            }
        }
    }

    ~TypedTopKBuffer() = default;

    // Copy constructor
    TypedTopKBuffer(const TypedTopKBuffer& other)
        : k_(other.k_), curr_offset_(other.curr_offset_),
          topk_(other.topk_), is_descending_(other.is_descending_), processing_query_(true)
    {


    }

    // Move constructor
    TypedTopKBuffer(TypedTopKBuffer&& other) noexcept
        : k_(other.k_), curr_offset_(other.curr_offset_),
          topk_(std::move(other.topk_)), is_descending_(other.is_descending_), processing_query_(true)
    {

    }

    // Copy assignment operator
    TypedTopKBuffer& operator=(const TypedTopKBuffer& other) {
        if (this != &other) {
            k_ = other.k_;
            curr_offset_ = other.curr_offset_;
            topk_ = other.topk_;
            is_descending_ = other.is_descending_;
        }
        return *this;
    }

    // Move assignment operator
    TypedTopKBuffer& operator=(TypedTopKBuffer&& other) noexcept {
        if (this != &other) {
            k_ = other.k_;
            curr_offset_ = other.curr_offset_;
            topk_ = std::move(other.topk_);
            is_descending_ = other.is_descending_;
        }
        return *this;
    }

    void set_k(int new_k) {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        assert(new_k <= topk_.size());
        k_ = new_k;
        reset();
    }

    void set_processing_query(bool new_value) {
        processing_query_.store(new_value, std::memory_order_relaxed);
    }

    inline bool currently_processing_query() {
        return processing_query_.load(std::memory_order_relaxed);
    }

    void set_jobs_left(int total_jobs) {
        jobs_left_.store(total_jobs, std::memory_order_relaxed);
    }

    void record_skipped_jobs(int skipped_jobs) {
        jobs_left_.fetch_sub(skipped_jobs, std::memory_order_relaxed);
    }

    void record_empty_job() {
        jobs_left_.fetch_sub(1, std::memory_order_relaxed);
    }

    inline bool finished_all_jobs() {
        int curr_jobs_left = jobs_left_.load(std::memory_order_relaxed);
        return jobs_left_.load(std::memory_order_relaxed) <= 0;
    }

    void reset() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        curr_offset_ = 0;
        for (int i = 0; i < k_; i++) {
            if (is_descending_) {
                topk_[i] = {std::numeric_limits<float>::min(), -1};
            }
            else {
                topk_[i] = {std::numeric_limits<float>::max(), -1};
            }
        }
    }

    void add(DistanceType distance, IdType index) {
        if (curr_offset_ >= topk_.size()) {
            flush(); // Flush the buffer if it is full
        }
        topk_[curr_offset_++] = {distance, index};
    }

    void batch_add(DistanceType* distances, IdType* indicies, int num_values) {
        if (num_values == 0) {
            jobs_left_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }

        // See if we can currently process queries
        if(!currently_processing_query()) {
            return;
        }

        // Get the offset to write the result to
        int write_offset;
        {
            std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
            
            if (curr_offset_ + num_values >= topk_.size()) {
                flush(); // Flush the buffer if it is full
            }

            write_offset = curr_offset_;
            curr_offset_ += num_values;
        }

        for(int i = 0; i < num_values; i++) {
            topk_[write_offset + i] = {distances[i], indicies[i]};
        }
        jobs_left_.fetch_sub(1, std::memory_order_relaxed);
    }

    DistanceType flush() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        if (curr_offset_ > k_) {
            if (is_descending_) {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            curr_offset_ = k_; // After flush, retain only the top-k elements
        }
        return topk_[std::min(curr_offset_, k_ - 1)].first;
    }

    std::vector<DistanceType> get_topk() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed

        std::vector<DistanceType> topk_distances(std::min(curr_offset_, k_));
        for (int i = 0; i < std::min(curr_offset_, k_); i++) {
            topk_distances[i] = topk_[i].first;
        }

        return topk_distances;
    }

    DistanceType get_kth_distance() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed
        return topk_[std::min(curr_offset_, k_ - 1)].first;
    }

    // Get the current top-k indices (after final flush)
    std::vector<IdType> get_topk_indices() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed

        std::vector<IdType> topk_indices(std::min(curr_offset_, k_));
        for (int i = 0; i < std::min(curr_offset_, k_); i++) {
            topk_indices[i] = topk_[i].second;
        }
        return topk_indices;
    }
};

// Type alias for convenience
using TopkBuffer = TypedTopKBuffer<float, int64_t>;

inline std::tuple<Tensor, Tensor> buffers_to_tensor(vector<TopkBuffer *> buffers, bool clear_buffers = true) {
    int n = buffers.size();
    int k = buffers[0]->k_;
    Tensor topk_distances = torch::empty({n, k}, torch::kFloat32);
    Tensor topk_indices = torch::empty({n, k}, torch::kInt64);

    for (int i = 0; i < n; i++) {
        vector<float> distances = buffers[i]->get_topk();
        vector<int64_t> indices = buffers[i]->get_topk_indices();

        for (int j = 0; j < k; j++) {
            topk_distances[i][j] = distances[j];
            topk_indices[i][j] = indices[j];
        }
    }

    if (clear_buffers) {
        for (int i = 0; i < n; i++) {
            free(buffers[i]);
        }
    }

    return std::make_tuple(topk_indices, topk_distances);
}

inline vector<TopkBuffer *> create_buffers(int n, int k, bool is_descending) {
    vector<TopkBuffer *> buffers(n);
    for (int i = 0; i < n; i++) {
        buffers[i] = new TopkBuffer(k, is_descending);
    }
    return buffers;
}

inline void scan_list(const float *query_vec,
                      const float *list_vecs,
                      const int64_t *list_ids,
                      int list_size,
                      int d,
                      TopkBuffer &buffer,
                      faiss::MetricType metric = faiss::METRIC_L2) {
    simsimd_distance_t dist;
    const float *vec;

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        if (list_ids == nullptr) {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_dot_f32(query_vec, vec, d, &dist);
                buffer.add(dist, l);
            }
        } else {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_dot_f32(query_vec, vec, d, &dist);
                buffer.add(dist, list_ids[l]);
            }
        }
    } else {
        if (list_ids == nullptr) {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_l2sq_f32(query_vec, vec, d, &dist);
                buffer.add(dist, l);
            }
        } else {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_l2sq_f32(query_vec, vec, d, &dist);
                buffer.add(dist, list_ids[l]);
            }
        }
    }
}

// inline void batched_list_scan(const float *query_vecs,
//                               const float *list_vecs,
//                               const int64_t *list_ids,
//                               int n,
//                               int list_size,
//                               int d,
//                               vector<TopkBuffer *> buffers) {
//
//     simsimd_distance_t dist;
//     const float *vec;
//
//     if (list_ids == nullptr) {
//         for (int l = 0; l < list_size; l++) {
//             for (int i = 0; i < n; i++) {
//                 vec = list_vecs + l * d;
//                 simsimd_l2sq_f32(query_vecs + i * d, vec, d, &dist);
//                 buffers[i]->add(dist, l);
//             }
//         }
//     } else {
//         for (int l = 0; l < list_size; l++) {
//             vec = list_vecs + l * d;
//             for (int i = 0; i < n; i++) {
//                 simsimd_l2sq_f32(query_vecs + i * d, vec, d, &dist);
//                 buffers[i]->add(dist, list_ids[l]);
//             }
//         }
//     }
// }
#endif //LIST_SCANNING_H
