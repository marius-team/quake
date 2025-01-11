//
// Created by Jason on 9/11/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LIST_SCANNING_H
#define LIST_SCANNING_H

#include <common.h>
#include "simsimd/simsimd.h"
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"

inline Tensor calculate_recall(Tensor ids, Tensor gt_ids) {
    Tensor num_correct = torch::zeros(ids.size(0), torch::kInt64);
    int num_queries = ids.size(0);
    int k = ids.size(1);

    int64_t *ids_ptr = ids.data_ptr<int64_t>();
    int64_t *gt_ids_ptr = gt_ids.data_ptr<int64_t>();

    for (int i = 0; i < num_queries; i++) {
        std::unordered_set<int64_t> gt_label_set;
        for (int j = 0; j < k; j++) {
            gt_label_set.insert(gt_ids_ptr[i * k + j]);
        }
        for (int j = 0; j < k; j++) {
            if (gt_label_set.find(ids_ptr[i * k + j]) != gt_label_set.end()) {
                num_correct[i] += 1;
            }
        }
    }

    Tensor recall = num_correct.to(torch::kFloat32) / k;

    return recall;
}

#define TOP_K_BUFFER_CAPACITY (8 * 1024)

template<typename DistanceType = float, typename IdType = int>
class TypedTopKBuffer {
public:
    int k_; // Number of top elements to keep
    int curr_offset_ = 0; // Current offset in the buffer
    std::vector<std::pair<DistanceType, IdType> > topk_; // Buffer to store top-k elements
    bool is_descending_; // Flag to indicate sorting order
    std::recursive_mutex buffer_mutex_;
    std::atomic<bool> processing_query_;
    std::atomic<int> jobs_left_;

    TypedTopKBuffer(int k, bool is_descending, int buffer_capacity = TOP_K_BUFFER_CAPACITY)
        : k_(k), is_descending_(is_descending), topk_(buffer_capacity), processing_query_(true) {
        assert(k <= buffer_capacity); // Ensure k is smaller than or equal to buffer size

        for (int i = 0; i < topk_.size(); i++) {
            if (is_descending_) {
                topk_[i] = {-std::numeric_limits<DistanceType>::infinity(), -1};
            } else {
                topk_[i] = {std::numeric_limits<DistanceType>::max(), -1};
            }
        }
    }

    ~TypedTopKBuffer() = default;

    // Copy constructor
    TypedTopKBuffer(const TypedTopKBuffer &other)
        : k_(other.k_), curr_offset_(other.curr_offset_),
          topk_(other.topk_), is_descending_(other.is_descending_), processing_query_(true) {
    }

    // Move constructor
    TypedTopKBuffer(TypedTopKBuffer &&other) noexcept
        : k_(other.k_), curr_offset_(other.curr_offset_),
          topk_(std::move(other.topk_)), is_descending_(other.is_descending_), processing_query_(true) {
    }

    // Copy assignment operator
    TypedTopKBuffer &operator=(const TypedTopKBuffer &other) {
        if (this != &other) {
            k_ = other.k_;
            curr_offset_ = other.curr_offset_;
            topk_ = other.topk_;
            is_descending_ = other.is_descending_;
        }
        return *this;
    }

    // Move assignment operator
    TypedTopKBuffer &operator=(TypedTopKBuffer &&other) noexcept {
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
                topk_[i] = { -std::numeric_limits<DistanceType>::infinity(), -1 };
            } else {
                topk_[i] = { std::numeric_limits<DistanceType>::max(), -1 };
            }
        }
    }

    void add(DistanceType distance, IdType index) {
        if (curr_offset_ >= topk_.size()) {
            flush(); // Flush the buffer if it is full
        }
        topk_[curr_offset_++] = {distance, index};
    }

    void batch_add(DistanceType *distances, const IdType *indices, int num_values) {
        if (num_values == 0) {
            jobs_left_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }

        if (!currently_processing_query()) {
            // If not processing, still decrement job count to avoid deadlock
            jobs_left_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }

        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);

        // Ensure buffer has enough capacity
        if (curr_offset_ + num_values > static_cast<int>(topk_.size())) {
            flush();
            // After flushing, if still not enough room, handle error or perform additional flushing/expansion
            if (curr_offset_ + num_values > static_cast<int>(topk_.size())) {
                throw std::runtime_error("Insufficient buffer capacity even after flush");
            }
        }

        int write_offset = curr_offset_;
        curr_offset_ += num_values;

        // Write new entries under the lock to maintain thread safety
        for (int i = 0; i < num_values; i++) {
            topk_[write_offset + i] = {distances[i], indices[i]};
        }

        jobs_left_.fetch_sub(1, std::memory_order_relaxed);
    }

    DistanceType flush() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        if (curr_offset_ > k_) {
            if (is_descending_) {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto &a, const auto &b) { return a.first > b.first; });
            } else {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto &a, const auto &b) { return a.first < b.first; });
            }
            curr_offset_ = k_; // After flush, retain only the top-k elements
        } else {
            // sort the curr_offset_ elements
            if (is_descending_) {
                std::sort(topk_.begin(), topk_.begin() + curr_offset_,
                          [](const auto &a, const auto &b) { return a.first > b.first; });
            } else {
                std::sort(topk_.begin(), topk_.begin() + curr_offset_,
                          [](const auto &a, const auto &b) { return a.first < b.first; });
            }
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

inline std::tuple<Tensor, Tensor> buffers_to_tensor(vector<shared_ptr<TopkBuffer>> buffers) {
    int n = buffers.size();
    int k = buffers[0]->k_;
    Tensor topk_distances = torch::empty({n, k}, torch::kFloat32);
    Tensor topk_indices = torch::empty({n, k}, torch::kInt64);

    auto topk_distances_accessor = topk_distances.accessor<float, 2>();
    auto topk_indices_accessor = topk_indices.accessor<int64_t, 2>();

    for (int i = 0; i < n; i++) {
        vector<float> distances = buffers[i]->get_topk();
        vector<int64_t> indices = buffers[i]->get_topk_indices();

        int curr_k = std::min(k, (int) distances.size());

        for (int j = 0; j < curr_k; j++) {
            topk_distances_accessor[i][j] = distances[j];
            topk_indices_accessor[i][j] = indices[j];
        }
    }

    return std::make_tuple(topk_indices, topk_distances);
}

inline vector<shared_ptr<TopkBuffer>> create_buffers(int n, int k, bool is_descending) {
    vector<shared_ptr<TopkBuffer>> buffers(n);
    for (int i = 0; i < n; i++) {
        buffers[i] = make_shared<TopkBuffer>(k, is_descending, 4 * k);
    }
    return buffers;
}

inline void scan_list(const float *query_vec,
                      const float *list_vecs,
                      const int64_t *list_ids,
                      int list_size,
                      int d,
                      shared_ptr<TopkBuffer> buffer,
                      faiss::MetricType metric = faiss::METRIC_L2) {
    simsimd_distance_t dist;
    const float *vec;

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        if (list_ids == nullptr) {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_dot_f32(query_vec, vec, d, &dist);
                buffer->add(dist, l);
            }
        } else {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_dot_f32(query_vec, vec, d, &dist);
                buffer->add(dist, list_ids[l]);
            }
        }
    } else {
        if (list_ids == nullptr) {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_l2sq_f32(query_vec, vec, d, &dist);
                buffer->add(sqrt(dist), l);
            }
        } else {
#pragma unroll
            for (int l = 0; l < list_size; l++) {
                vec = list_vecs + l * d;
                simsimd_l2sq_f32(query_vec, vec, d, &dist);
                buffer->add(sqrt(dist), list_ids[l]);
            }
        }
    }
}

inline void batched_scan_list(const float *query_vecs,
                              const float *list_vecs,
                              const int64_t *list_ids,
                              int num_queries,
                              int list_size,
                              int dim,
                              vector<shared_ptr<TopkBuffer>> &topk_buffers,
                              MetricType metric = faiss::METRIC_L2,
                              int batch_size = 1024 * 32) {
    // Determine if larger values are better based on the metric
    bool largest = (metric == faiss::METRIC_INNER_PRODUCT);

    // Handle the case when list_size == 0
    if (list_size == 0 || list_vecs == nullptr) {
        // No list vectors to process; all TopkBuffers remain with default values
        return;
    }

    // Ensure k does not exceed list_size
    int k = topk_buffers[0]->k_;
    int k_max = std::min(k, list_size);

    // Create tensors from raw pointers
    Tensor query_tensor = torch::from_blob((void *) query_vecs, {num_queries, dim}, torch::kFloat32);
    Tensor list_tensor = torch::from_blob((void *) list_vecs, {list_size, dim}, torch::kFloat32);

    // Determine batching strategy
    if (num_queries >= list_size && list_size > 0) {
        // Batch over queries
        for (int start = 0; start < num_queries; start += batch_size) {
            int end = std::min(start + batch_size, num_queries);
            int curr_size = end - start;

            Tensor curr_queries = query_tensor.slice(0, start, end); // Shape: [curr_size, dim]
            Tensor dist_matrix;
            if (metric == faiss::METRIC_L2) {
                // Compute L2 squared distances
                dist_matrix = torch::cdist(curr_queries, list_tensor, 2.0); // Shape: [curr_size, list_size]
            } else {
                // Compute Inner Product
                dist_matrix = torch::matmul(curr_queries, list_tensor.t()); // Shape: [curr_size, list_size]
            }

            // Perform top-k with k_max
            auto topk = torch::topk(dist_matrix, k_max, /*dim=*/1, /*largest=*/largest, /*sorted=*/true);
            auto topk_values_accessor = std::get<0>(topk).accessor<float, 2>();
            auto topk_indices_accessor = std::get<1>(topk).accessor<int64_t, 2>();

            // Update TopkBuffers
            for (int i = 0; i < curr_size; i++) {
                for (int j = 0; j < k_max; j++) {
                    float distance = topk_values_accessor[i][j];
                    int64_t idx = topk_indices_accessor[i][j];
                    int64_t actual_id = (list_ids == nullptr) ? idx : list_ids[idx];
                    topk_buffers[start + i]->add(distance, actual_id);
                }
            }
        }
    } else {
        // Batch over list vectors
        for (int start = 0; start < list_size; start += batch_size) {
            int end = std::min(start + batch_size, list_size);
            int curr_size = end - start;

            Tensor curr_list = list_tensor.slice(0, start, end); // Shape: [curr_size, dim]
            Tensor dist_matrix;
            if (metric == faiss::METRIC_L2) {
                // Compute L2 squared distances
                dist_matrix = torch::cdist(query_tensor, curr_list, 2.0); // Shape: [num_queries, curr_size]
            } else {
                // Compute Inner Product
                dist_matrix = torch::matmul(query_tensor, curr_list.t()); // Shape: [num_queries, curr_size]
            }

            // Perform top-k with k_max
            auto topk = torch::topk(dist_matrix, k_max, /*dim=*/1, /*largest=*/largest, /*sorted=*/true);
            auto topk_values_accessor = std::get<0>(topk).accessor<float, 2>();
            auto topk_indices_accessor = std::get<1>(topk).accessor<int64_t, 2>();

            // Update TopkBuffers
            for (int q = 0; q < num_queries; q++) {
                for (int j = 0; j < k_max; j++) {
                    float distance = topk_values_accessor[q][j];
                    int64_t idx = topk_indices_accessor[q][j] + start;
                    int64_t actual_id = (list_ids == nullptr) ? idx : list_ids[idx];
                    topk_buffers[q]->add(distance, actual_id);
                }
            }
        }
    }
}

// }
#endif //LIST_SCANNING_H
