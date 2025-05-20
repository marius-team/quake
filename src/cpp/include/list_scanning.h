//
// Created by Jason on 9/11/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LIST_SCANNING_H
#define LIST_SCANNING_H

#include <common.h>
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"
#include "sorting/pdqsort.h"
#include "sorting/floyd_rivest_select.h"
#include "sorting/heap_select.h"
#include "parallel.h"

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

template<typename T, typename I>
class TypedTopKBuffer {
public:
    T *vals_;    // size = capacity
    I *ids_;
    int capacity_, head_, k_;
    bool is_desc_;
    int *ord_;
    bool owns_memory_ = true;
    TypedTopKBuffer(int k, bool desc, int cap, int node)
            : capacity_(cap), head_(0), k_(k) {
        vals_ = static_cast<T *>(quake_alloc(sizeof(T) * cap, node));
        ids_ = static_cast<I *>(quake_alloc(sizeof(I) * cap, node));
        ord_ = static_cast<int *>(quake_alloc(sizeof(int) * cap, node));

        if (capacity_ < k_) {
            throw std::invalid_argument("capacity must be greater than k");
        }

        if (desc) {
            is_desc_ = true;
            for (int i = 0; i < capacity_; i++) {
                vals_[i] = -std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            }
        } else {
            is_desc_ = false;
            for (int i = 0; i < capacity_; i++) {
                vals_[i] = std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            }
        }
    }

    // Create buffer but do not allocate memory (except for ord_)
    TypedTopKBuffer(T *vals, I* ids, int cap, int k, bool desc, int node) {
        capacity_ = cap;
        head_ = 0;
        k_ = k;
        is_desc_ = desc;
        vals_ = vals;
        ids_ = ids;
        ord_ = static_cast<int *>(quake_alloc(sizeof(int) * cap, node));
        owns_memory_ = false;

        if (capacity_ < k_) {
            throw std::invalid_argument("capacity must be greater than k");
        }

        if (desc) {
            for (int i = 0; i < capacity_; i++) {
                vals_[i] = -std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            }
        } else {
            for (int i = 0; i < capacity_; i++) {
                vals_[i] = std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            }
        }
    }

    ~TypedTopKBuffer() {
        if (owns_memory_) {
            if (vals_) {
                quake_free(vals_, sizeof(T) * capacity_);
            }

            if (ids_) {
                quake_free(ids_, sizeof(I) * capacity_);
            }
        }

        if (ord_) {
            quake_free(ord_, sizeof(int) * capacity_);
        }

        vals_ = nullptr;
        ids_ = nullptr;
        ord_ = nullptr;
    }

    void set_k(int new_k) {
        if (new_k > capacity_) {
            throw std::invalid_argument("k must be less than the capacity");
        }
        k_ = new_k;
        reset();
    }

    void reset() {
        head_ = 0;
        for (int i = 0; i < k_; i++) {
            if (is_desc_) {
                vals_[i] = -std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            } else {
                vals_[i] = std::numeric_limits<T>::infinity();
                ids_[i] = -1;
            }
        }
    }

    inline void add(T dist, I idx) {
        vals_[head_] = dist;
        ids_[head_] = idx;
        if (++head_ == capacity_) flush();
    }

    void batch_add(T *distances, I *indices, int num_values) {
        int pos = 0;
        while (pos < num_values) {
            int available = capacity_ - head_;
            if (available <= 0) {
                flush();
                available = capacity_ - head_;
            }
            int to_copy = std::min(num_values - pos, available);
            for (int i = 0; i < to_copy; i++) {
                vals_[head_] = distances[pos + i];
                ids_[head_] = indices[pos + i];
                head_++;
            }
            pos += to_copy;
        }
    }

    T flush() {


        int n = head_;
        int m = std::min(n, k_);

        // 1) build identity permutation
        for (int i = 0; i < n; ++i) {
            ord_[i] = i;
        }
        // comparator by value
        auto cmpIdx = [&](int a, int b) {
            return is_desc_ ? (vals_[a] > vals_[b])
                            : (vals_[a] < vals_[b]);
        };

        // 2) select top‐m indices into ord_[0..m)
        if (n > m) {
            if (m < 10) {
                miniselect::heap_select(ord_, ord_ + m, ord_ + n, cmpIdx);
            } else if (m < n * 0.001) {
                miniselect::floyd_rivest_select(ord_, ord_ + m, ord_ + n, cmpIdx);
            } else {
                miniselect::pdqpartial_sort_branchless(ord_, ord_ + m, ord_ + n, cmpIdx);
            }

        } else {
            miniselect::pdqsort_branchless(ord_, ord_ + n, cmpIdx);
        }

        // 4) fully sort the top-m indices so they are in strictly correct order
        miniselect::pdqsort_branchless(ord_, ord_ + m, cmpIdx);

        // 5) copy the winners back to vals_/ids_ and clamp head_
        std::vector<T> temp_v(m);
        std::vector<I> temp_i(m);

        for (int i = 0; i < m; ++i) {
            int original_slot_idx = ord_[i]; // ord_[i] is the original index of the i-th best item
            // (e.g. ord_[0] is index of best, ord_[1] of 2nd best)
            temp_v[i] = vals_[original_slot_idx];
            temp_i[i] = ids_[original_slot_idx];
        }

        // Now copy from temporary buffers to the main buffers
        for (int i = 0; i < m; ++i) {
            vals_[i] = temp_v[i];
            ids_[i]  = temp_i[i];
        }

        // 5) clamp head_ and return the k-th value (or extreme if too few)
        head_ = m;
        if (head_ == 0) {
            return is_desc_
                   ? -std::numeric_limits<T>::infinity()
                   :  std::numeric_limits<T>::infinity();
        }
        int ret_i = std::min(k_ - 1, head_ - 1);
        return vals_[ret_i];
    }


    std::vector<T> get_topk() {
        flush();
        int n = head_;
        std::vector<T> out;
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            out.push_back(vals_[i]);
        }
        return out;
    }

    T get_kth_distance() {
        flush();
        if (head_ < k_) {
            // not enough elements: return the sentinel extreme
            return is_desc_
                   ? -std::numeric_limits<T>::infinity()
                   : std::numeric_limits<T>::infinity();
        }
        return vals_[k_ - 1];
    }

    std::vector<I> get_topk_indices() {
        flush();
        int n = head_;
        std::vector<I> out;
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            out.push_back(ids_[i]);
        }
        return out;
    }
};

// Type alias for convenience
using TopkBuffer = TypedTopKBuffer<float, int64_t>;

inline vector<shared_ptr<TopkBuffer>> create_buffers(int batch_size, int k, bool is_desc) {
    vector<shared_ptr<TopkBuffer>> buffers;
    for (int i = 0; i < batch_size; i++) {
        buffers.push_back(make_shared<TopkBuffer>(k, is_desc, k*100, 0));
    }
    return buffers;
}

//vector<shared_ptr<TopkBuffer>> local_buffers = create_buffers(batch_size, k, (metric_ == faiss::METRIC_INNER_PRODUCT));

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

inline void scan_list_no_ids_inner_product(const float *query_vec,
                                                   const float *list_vecs,
                                                   int list_size,
                                                   int d,
                                                   TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(faiss::fvec_inner_product(query_vec, vec, d), l);
        vec += d;  // move pointer to next vector
    }
}

inline void scan_list_no_ids_l2(const float *query_vec,
                                      const float *list_vecs,
                                      int list_size,
                                      int d,
                                      TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(sqrt(faiss::fvec_L2sqr(query_vec, vec, d)), l);
        vec += d;
    }
}

inline void scan_list_with_ids_inner_product(const float *query_vec,
                                                     const float *list_vecs,
                                                     const int64_t *list_ids,
                                                     int list_size,
                                                     int d,
                                                     TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(faiss::fvec_inner_product(query_vec, vec, d), list_ids[l]);
        vec += d;
    }
}

inline void scan_list_with_ids_l2(const float *query_vec,
                                        const float *list_vecs,
                                        const int64_t *list_ids,
                                        int list_size,
                                        int d,
                                        TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(sqrt(faiss::fvec_L2sqr(query_vec, vec, d)), list_ids[l]);
        vec += d;
    }
}

// The main scan_list function that dispatches to one of the specialized functions.
inline void scan_list(const float *query_vec,
                            const float *list_vecs,
                            const int64_t *list_ids,
                            int list_size,
                            int d,
                            TopkBuffer &buffer,
                            faiss::MetricType metric = faiss::METRIC_L2) {
    // Dispatch based on metric type and whether list_ids is provided.
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        if (list_ids == nullptr)
            scan_list_no_ids_inner_product(query_vec, list_vecs, list_size, d, buffer);
        else
            scan_list_with_ids_inner_product(query_vec, list_vecs, list_ids, list_size, d, buffer);
    } else { // Assume L2 (or similar)
        if (list_ids == nullptr)
            scan_list_no_ids_l2(query_vec, list_vecs, list_size, d, buffer);
        else
            scan_list_with_ids_l2(query_vec, list_vecs, list_ids, list_size, d, buffer);
    }
}

//inline void batched_scan_list(
//        const float *query_vecs,
//        const float *list_vecs,
//        const int64_t *list_ids,
//        int num_queries,
//        int list_size,
//        int dim,
//        vector<shared_ptr<TopkBuffer>> &topk_buffers,
//        MetricType metric = faiss::METRIC_L2)
//{
//    if (list_size == 0 || list_vecs == nullptr) {
//        return;
//    }
//
//    // Wrap raw arrays in torch Tensors, no copy
//    Tensor query = torch::from_blob((void*)query_vecs, {num_queries, dim}, torch::kFloat32);
//    Tensor list  = torch::from_blob((void*)list_vecs,  {list_size, dim}, torch::kFloat32);
//
//    torch::Tensor distances;
//    if (metric == faiss::METRIC_L2) {
//        // Returns [num_queries, list_size], each entry is the Euclidean distance
//        distances = torch::cdist(query, list, 2.0);
//    } else if (metric == faiss::METRIC_INNER_PRODUCT) {
//        // [num_queries, list_size], each entry is dot product
//        distances = torch::matmul(query, list.t());
//    } else {
//        throw std::runtime_error("Metric type not supported");
//    }
//
//    // For each query, push all list vectors and their distances into TopkBuffer
//    auto distances_acc = distances.accessor<float,2>();
//    for (int i = 0; i < num_queries; ++i) {
//        std::vector<float> dists(list_size);
//        std::vector<int64_t> ids(list_size);
//
//        for (int j = 0; j < list_size; ++j) {
//            dists[j] = distances_acc[i][j];
//        }
//
//        if (list_ids) {
//            for (int j = 0; j < list_size; ++j) {
//                ids[j] = list_ids[j];
//            }
//        } else {
//            for (int j = 0; j < list_size; ++j) {
//                ids[j] = j;
//            }
//        }
//
//        topk_buffers[i]->batch_add(dists.data(), ids.data(), list_size);
//    }
//}

inline void batched_scan_list(const float *query_vecs,
                              const float *list_vecs,
                              const int64_t *list_ids,
                              int num_queries,
                              int list_size,
                              int dim,
                              vector<shared_ptr<TopkBuffer>> &topk_buffers,
                              MetricType metric = faiss::METRIC_L2,
                              float *distances = nullptr,
                              int64_t *labels = nullptr) {
    if (list_size == 0 || list_vecs == nullptr) {
        // No list vectors to process;
        return;
    }

    // Ensure k does not exceed list_size
    int k = topk_buffers[0]->k_;
    int k_max = std::min(k, list_size);

    bool alloc_results = false;
    if (distances == nullptr) {
        alloc_results = true;
        labels = (int64_t *) malloc(num_queries * k_max * sizeof(int64_t));
        distances = (float *) malloc(num_queries * k_max * sizeof(float));
    }

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        faiss::float_minheap_array_t res = {size_t(num_queries), size_t(k_max), labels, distances};
        faiss::knn_inner_product(query_vecs, list_vecs, dim, num_queries, list_size, &res, nullptr);
    } else if (metric == faiss::METRIC_L2) {
        faiss::float_maxheap_array_t res = {size_t(num_queries), size_t(k_max), labels, distances};
        faiss::knn_L2sqr(query_vecs, list_vecs, dim, num_queries, list_size, &res, nullptr, nullptr);
    } else {
        throw std::runtime_error("Metric type not supported");
    }

    // map the labels to the actual list_ids
    if (list_ids != nullptr) {
        for (int i = 0; i < num_queries; i++) {
            for (int j = 0; j < k_max; j++) {
                labels[i * k_max + j] = list_ids[labels[i * k_max + j]];
            }
        }
    }

    // if the metric is l2, convert the distances to sqrt
    if (metric == faiss::METRIC_L2) {
        for (int i = 0; i < num_queries * k_max; i++) {
            distances[i] = sqrt(distances[i]);
        }
    }

    // add distances to the topk buffers
    for (int i = 0; i < num_queries; i++) {
        topk_buffers[i]->batch_add(distances + i * k_max, labels + i * k_max, k_max);
    }

    if (alloc_results) {
        free(distances);
        free(labels);
    }
}

// }
#endif //LIST_SCANNING_H