//
// Created by Jason on 8/28/24.
// Conform to the Google style guide
// Use descriptive variable names

#include "dynamic_ivf.h"

#include <torch/torch.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexRefine.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/Index.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/IndexIVF.h>
#include <faiss/utils/utils.h>
#include "dynamic_inverted_list.h"
#include "geometry.h"
#include "clustering.h"


/**
 * @brief Reorders search results by merging two heaps.
 *
 * @tparam Comparator Comparison functor (e.g., faiss::CMax<float, int64_t>).
 * @param n Number of queries.
 * @param k Number of nearest neighbors to return.
 * @param labels Output labels (indices of nearest neighbors).
 * @param distances Output distances to nearest neighbors.
 * @param k_base Number of initial candidates.
 * @param base_labels Initial labels.
 * @param base_distances Initial distances.
 */
template<class Comparator>
static void reorder_two_heaps(
    int64_t n, int64_t k,
    int64_t *labels, float *distances,
    int64_t k_base,
    const int64_t *base_labels,
    const float *base_distances) {
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        int64_t *idx_out = labels + i * k;
        float *dist_out = distances + i * k;
        const int64_t *idx_in = base_labels + i * k_base;
        const float *dist_in = base_distances + i * k_base;

        faiss::heap_heapify<Comparator>(k, dist_out, idx_out, dist_in, idx_in, k);
        if (k_base != k) {
            faiss::heap_addn<Comparator>(k, dist_out, idx_out, dist_in + k, idx_in + k, k_base - k);
        }
        faiss::heap_reorder<Comparator>(k, dist_out, idx_out);
    }
}

inline void printCurrentTime(std::string label) {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);

    std::ostringstream oss;
    oss << label << " epoch time: " << epoch_duration.count() << std::endl;
    std::cout << oss.str();
}

DynamicIVF_C::DynamicIVF_C(int d, int nlist, int metric, int num_workers, int m, int code_size, bool use_numa, bool verbose, bool verify_numa, 
    bool same_core, bool use_centroid_workers, bool use_adaptive_nprobe) : d_(d), num_codebooks_(m), code_size_(code_size), num_scan_workers_(num_workers), 
    max_vectors_per_cluster_(-1), clusters_distributed_(false), using_numa_optimizations_(use_numa),log_mode_(verbose), num_partitions_to_scan_(-1),centroid_query_id_(0), 
    workers_initialized_(false), max_jobs_(0), total_outputs_per_query_(0), same_core_(same_core), query_latency_target_time_us_(-1), partition_search_flush_gap_us_(-1), 
    curr_depth_(0), use_centroid_workers_(use_centroid_workers), use_adpative_nprobe_(use_adaptive_nprobe) {

    faiss::MetricType metric_type = static_cast<faiss::MetricType>(metric);
    if (nlist < 1) {
        index_ = new faiss::IndexIDMap2(new faiss::IndexFlat(d, metric_type));
        centroid_store_ = new faiss::DynamicCentroidStore(d, verbose);
        index_->verbose = verbose;
        use_refine_ = false;
        refine_index_ = nullptr;
        ivf_index_ = nullptr;
    } else {
        centroid_store_ = nullptr;
        auto quantizer = new faiss::IndexFlat(d, metric_type);
        if (m == -1) {
            index_ = new faiss::IndexIVFFlat(quantizer, d, nlist, metric_type);
            index_->verbose = verbose;
            use_refine_ = false;
            refine_index_ = nullptr;
        } else {
            index_ = new faiss::IndexIVFPQ(quantizer, d, nlist, m, code_size);
            refine_index_ = new faiss::IndexRefineFlat(index_);
            index_->verbose = verbose;
            use_refine_ = true;
        }

        ivf_index_ = dynamic_cast<faiss::IndexIVF*>(index_);
        ivf_index_->quantizer->verbose = false;
        ivf_index_->own_invlists = true;
        ivf_index_->replace_invlists(new faiss::DynamicInvertedLists(nlist, ivf_index_->code_size), true);

        parent_ = nullptr;
    }

    // Initialize the worker specific fields
    verify_numa_ = using_numa_optimizations_ && verify_numa;
    workers_numa_nodes_.reserve(num_scan_workers_);

#ifdef QUAKE_NUMA
    int num_numa_nodes = get_num_numa_nodes();
    for(int i = 0; i < num_scan_workers_; i++) {
        workers_numa_nodes_[i] = i % num_numa_nodes;
    }

    curr_queries_per_node_.reserve(num_numa_nodes);
    int total_buffer_size = QUERY_BUFFER_REUSE_THRESHOLD * d_;
    for(int i = 0; i < num_numa_nodes; i++) {
        if(total_buffer_size > 0) {
            curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(total_buffer_size * sizeof(float), i));
            if(curr_queries_per_node_[i] == NULL) {
                throw std::runtime_error("Unable to allocate vector on numa node");
            }
        } else {
            curr_queries_per_node_[i] = nullptr;
        }
    }
#endif

    job_query_id_ = nullptr;
    job_search_cluster_id_ = nullptr;
    all_vectors_scanned_ptr_ = new int[num_scan_workers_];

    if(log_mode_) {
        all_counts_ptr_ = new int[num_scan_workers_];
        all_job_times_ptr_ = new int[num_scan_workers_];
        all_scan_times_ptr_ = new int[num_scan_workers_];
        all_throughputs_ptr_ = new float[num_scan_workers_];
    }


}

DynamicIVF_C::~DynamicIVF_C() {
    if (use_refine_) {
        delete refine_index_;
    }

#ifdef QUAKE_NUMA
    if(using_numa_optimizations_) {
        // Stop all of the running workers
        if(scan_workers_.size() > 0) {
            for(int i = 0; i < jobs_queue_.size(); i++) {
                for(int j = 0; j < num_scan_workers_; j++) {
                    jobs_queue_[i].enqueue(-1);
                }
            }

            // Join all cluster scan workers
            for (std::thread &worker: scan_workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

        // Free up the numa buffers
        int total_buffer_size = QUERY_BUFFER_REUSE_THRESHOLD * d_;
        for(int i = 0; i < curr_queries_per_node_.size(); i++) {
            if(curr_queries_per_node_[i] != nullptr) {
                numa_free(curr_queries_per_node_[i], total_buffer_size * sizeof(float));
            }
            curr_queries_per_node_[i] = nullptr;
        }

        if(job_query_id_ != nullptr) {
            delete[] job_query_id_;
        }

        if(job_search_cluster_id_ != nullptr) {
            delete[] job_search_cluster_id_;
        }
    }
#endif

    delete index_;
    delete centroid_store_;
}

int DynamicIVF_C::nlist() const {
    if (parent_ != nullptr) {
        return parent_->ntotal();
    } else {
        if (ivf_index_ != nullptr) {
            return ivf_index_->nlist;
        } else {
            return 1;
        }
    }
}

float DynamicIVF_C::get_scan_fraction() const {
    if (parent_ != nullptr) {
        float parent_scan_fraction = parent_->get_scan_fraction();
        int64_t parent_ntotal = parent_->ntotal();
        int64_t this_ntotal = ntotal();
        parent_scan_fraction *= (parent_ntotal / this_ntotal);
        float total_scan_fraction = maintenance_policy_->current_scan_fraction_ + parent_scan_fraction;
        return total_scan_fraction;
    } else {
        return 1.0;
    }
}

int64_t DynamicIVF_C::ntotal() const {
    return index_->ntotal;
}

void DynamicIVF_C::reset_workers(int num_workers, bool same_core, bool use_numa_optimizations) {

    // spin down the workers
    if (scan_workers_.size() > 0) {
        for (int i = 0; i < jobs_queue_.size(); i++) {
            for (int j = 0; j < num_scan_workers_; j++) {
                jobs_queue_[i].enqueue(-1);
            }
        }

        // Join all cluster scan workers
        for (std::thread &worker: scan_workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    workers_initialized_ = false;
    num_scan_workers_ = num_workers;
    same_core_ = same_core;
    using_numa_optimizations_ = use_numa_optimizations;

    workers_numa_nodes_.reserve(num_scan_workers_);
#ifdef QUAKE_NUMA

    int num_numa_nodes = get_num_numa_nodes();
    for(int i = 0; i < num_scan_workers_; i++) {
        workers_numa_nodes_[i] = i % num_numa_nodes;
    }

    if(using_numa_optimizations_) {
        curr_queries_per_node_.reserve(num_numa_nodes);
        int total_buffer_size = QUERY_BUFFER_REUSE_THRESHOLD * d_;
        for(int i = 0; i < num_numa_nodes; i++) {
            // Free up the existing buffer
            if(curr_queries_per_node_[i] != nullptr) {
                numa_free(curr_queries_per_node_[i], total_buffer_size * sizeof(float));
            }

            // Allocate the new buffer
            curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(total_buffer_size * sizeof(float), i));
            if(curr_queries_per_node_[i] == NULL) {
                throw std::runtime_error("Unable to allocate vector on numa node");
            }
        }
    }
#endif

    // Reset the intermediate buffers
    job_query_id_ = nullptr;
    job_search_cluster_id_ = nullptr;
    all_vectors_scanned_ptr_ = new int[num_scan_workers_];
    final_result_mergers_.clear();

    // launch the workers
    launch_cluster_scan_workers(true);
}

std::tuple<std::vector<std::vector<uint8_t>>, std::vector<std::vector<idx_t>>> DynamicIVF_C::get_partitions() {
    // Create the inital buffers
    auto casted_index = ivf_index_;
    std::vector<std::vector<uint8_t>> all_vectors;
    std::vector<std::vector<idx_t>> all_ids;
    all_vectors.reserve(nlist());
    all_ids.reserve(nlist());

    for (int i = 0; i < nlist(); i++) {
        size_t list_size = casted_index->invlists->list_size(i);
        auto list_codes = casted_index->invlists->get_codes(i);
        auto list_ids = casted_index->invlists->get_ids(i);

        std::vector<uint8_t> curr_vec_values;
        size_t num_vec_values = list_size * d_ * sizeof(float);
        for(size_t j = 0; j < num_vec_values; j++) {
            curr_vec_values.push_back(list_codes[j]);
        }
        all_vectors.push_back(curr_vec_values);

        std::vector<idx_t> curr_ids;
        for(size_t j = 0; j < list_size; j++) {
            curr_ids.push_back(list_ids[j]);
        }
        all_ids.push_back(curr_ids);
    }

    return std::make_pair(all_vectors, all_ids);
}

std::tuple<Tensor, Tensor> DynamicIVF_C::get_vectors_and_ids(bool use_centroid_store) {
    if (ivf_index_ == nullptr) {
        auto flat_index = dynamic_cast<faiss::IndexFlat*>(dynamic_cast<faiss::IndexIDMap2*>(index_)->index);
        assert(flat_index != nullptr && "Can't cast index into an flat index");

        Tensor vectors = torch::zeros({flat_index->ntotal, flat_index->d}, torch::kFloat32).contiguous();
        flat_index->reconstruct_n(0, flat_index->ntotal, vectors.data_ptr<float>());

        // Create the ids
        if(use_centroid_store) {
            std::vector<int64_t> all_ids = centroid_store_->get_all_ids();
            Tensor ids = torch::from_blob((void *) all_ids.data(), {(int64_t) all_ids.size()}, torch::kInt64).clone();
            return std::make_tuple(vectors, ids);
        } else {
            Tensor ids = torch::arange(flat_index->ntotal, torch::kInt64);
            return std::make_tuple(vectors, ids);
        }
    } else {
        vector<Tensor> vectors;
        vector<Tensor> ids;
        for (int i = 0; i < nlist(); i++) {
            int64_t list_size = ivf_index_->invlists->list_size(i);
            if (list_size == 0) {
                continue;
            }

            auto codes = ivf_index_->invlists->get_codes(i);
            auto list_ids = ivf_index_->invlists->get_ids(i);

            Tensor cluster_vectors = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);
            Tensor cluster_ids = torch::from_blob((void *) list_ids, {list_size}, torch::kInt64);

            vectors.push_back(cluster_vectors);
            ids.push_back(cluster_ids);
        }
        return std::make_pair(torch::cat(vectors, 0), torch::cat(ids, 0));
    }
}

Tensor DynamicIVF_C::get_ids() {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        auto flat_index = dynamic_cast<faiss::IndexIDMap2 *>(index_);
        // get ids from the flat index
        vector<int64_t> ids = flat_index->id_map;
        return torch::from_blob((void *) ids.data(), {(int64_t) ids.size()}, torch::kInt64).clone();
    } else {
        vector<Tensor> ids;
        ids.reserve(nlist());
        for (int i = 0; i < nlist(); i++) {
            int64_t list_size = casted_index->invlists->list_size(i);
            if (list_size == 0) {
                continue;
            }

            auto list_ids = casted_index->invlists->get_ids(i);
            Tensor cluster_ids = torch::from_blob((void *) list_ids, {list_size}, torch::kInt64).clone();
            ids.push_back(cluster_ids);
        }
        return torch::cat(ids, 0);
    }
}

faiss::DynamicInvertedLists *DynamicIVF_C::get_invlists() {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return nullptr;
    } else {
        return dynamic_cast<faiss::DynamicInvertedLists *>(casted_index->invlists);
    }
}


Tensor DynamicIVF_C::centroids() {
    std::tie(centroids_, std::ignore) = parent_->get_vectors_and_ids();
    return centroids_;
}

std::tuple<Tensor, vector<Tensor>, vector<Tensor> > DynamicIVF_C::split_partitions(Tensor partition_ids) {
    // Ensure partition_ids is contiguous and on CPU
    partition_ids = partition_ids.contiguous();

    // Calculate the number of new partitions after splitting
    int64_t num_partitions_to_split = partition_ids.size(0);
    int64_t num_splits = 2; // Example: splitting each partition into 2
    int64_t total_new_partitions = num_partitions_to_split * num_splits;

    // Initialize tensors to hold new centroids, vectors, and IDs
    Tensor split_centroids = torch::empty({total_new_partitions, d_}, torch::kFloat32);
    vector<Tensor> split_vectors;
    vector<Tensor> split_ids;

    split_vectors.reserve(total_new_partitions);
    split_ids.reserve(total_new_partitions);

    // Retrieve data from the selected partitions
    Tensor initial_centroids;
    vector<Tensor> initial_vectors;
    vector<Tensor> initial_ids;
    std::tie(initial_centroids, initial_vectors, initial_ids) = select_clusters(partition_ids, /*copy=*/false);

    // Perform k-means splitting on each partition
    for (int64_t i = 0; i < partition_ids.size(0); ++i) {
        // Perform k-means with k=2 to split the partition into two
        assert(initial_vectors[i].size(0) >= 4 && "Partition must have at least 8 vectors to split.");

        Tensor new_centroid;
        vector<Tensor> new_split_vectors;
        vector<Tensor> new_split_ids;
        std::tie(new_centroid, new_split_vectors, new_split_ids) = kmeans(
            initial_vectors[i],
            initial_ids[i],
            2,
            index_->metric_type
        );

        // Assign the new centroid and corresponding vectors and IDs
        for (size_t j = 0; j < new_split_vectors.size(); ++j) {
            split_centroids[i * num_splits + j] = new_centroid[j];
            split_vectors.push_back(new_split_vectors[j]);
            split_ids.push_back(new_split_ids[j]);
        }
    }

    return std::make_tuple(split_centroids, split_vectors, split_ids);
}

Tensor DynamicIVF_C::add_partitions(Tensor new_centroids, vector<Tensor> new_cluster_vectors,
                                    vector<Tensor> new_cluster_ids) {
    // Validate input sizes
    int64_t num_new_partitions = new_centroids.size(0);
    assert(
        new_cluster_vectors.size() == num_new_partitions &&
        "Number of new_cluster_vectors must match number of new_centroids.");
    assert(
        new_cluster_ids.size() == num_new_partitions &&
        "Number of new_cluster_ids must match number of new_centroids.");
    assert(new_centroids.size(1) == d_ && "Dimensionality of new_centroids must match existing centroids.");

    // Retrieve the number of existing partitions
    int old_nlist = nlist();
    int new_nlist = old_nlist + num_new_partitions;
    size_t total_existing_vectors = index_->ntotal;

    // Add new centroids to the quantizer
    Tensor new_partition_ids = add_centroids(new_centroids);

    // Assign new vectors to the index
    for (int64_t i = 0; i < num_new_partitions; ++i) {
        int64_t num_vectors = new_cluster_vectors[i].size(0);

        if (num_vectors == 0) {
            continue;
        }

        // Ensure vectors and IDs are contiguous in memory
        Tensor vectors = new_cluster_vectors[i].contiguous();
        Tensor ids = new_cluster_ids[i].contiguous();

        // Add vectors with their IDs to the index
        add(vectors, ids, false);
    }

    // Return the IDs of the newly added partitions
    // Tensor new_partition_ids = torch::arange(old_nlist, new_nlist, torch::kInt64);
    // auto new_partition_ids_accessor = new_partition_ids.accessor<int64_t, 1>();
    // for (int i = 0; i < num_new_partitions; i++) {
    //     maintenance_policy_->add_partition(new_partition_ids_accessor[i]);
    // }
    return new_partition_ids;
}

Tensor DynamicIVF_C::get_partition_ids() const {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kInt64);
    } else {
        auto casted_invlists = dynamic_cast<faiss::DynamicInvertedLists *>(casted_index->invlists);
        Tensor partition_ids = torch::empty({(int64_t) casted_invlists->ids_.size()}, torch::kInt64);

        // Iterate through the map to get the keys
        int i = 0;
        for (auto const &element: casted_invlists->ids_) {
            partition_ids[i] = (int64_t) element.first;
            i++;
        }
        return partition_ids;
    }
}

void DynamicIVF_C::delete_partitions(Tensor partition_ids, bool reassign, Tensor reassignments) {
    if (parent_ != nullptr) {
        auto casted_index = ivf_index_;
        assert(casted_index != nullptr && "Index must be of type IndexIVF.");
        Tensor centroids;
        vector<Tensor> cluster_vectors;
        vector<Tensor> cluster_ids;

        std::tie(centroids, cluster_vectors, cluster_ids) = select_clusters(partition_ids, /*copy=*/true);
        parent_->remove(partition_ids);

        auto invlists = dynamic_cast<faiss::DynamicInvertedLists *>(casted_index->invlists);

        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids[i].item<int64_t>();
            invlists->remove_list(list_no);
            casted_index->nlist--;
        }

        // decrement the ntotal count
        for (int i = 0; i < cluster_ids.size(); i++) {
            casted_index->ntotal -= cluster_ids[i].size(0);
        }

        if (reassign) {
            for (int i = 0; i < partition_ids.size(0); i++) {
                Tensor vectors = cluster_vectors[i];
                Tensor ids = cluster_ids[i];
                if (vectors.size(0) == 0) {
                    continue;
                }
                add(vectors, ids);
            }
        }
    } else {
        throw std::runtime_error("Index is not partitioned");
    }
}

Tensor DynamicIVF_C::add_centroids(Tensor centroids) {
    if (parent_ != nullptr) {
        int64_t current_nlist = nlist();

        auto casted_index = ivf_index_;
        assert(casted_index != nullptr && "Index must be of type IndexIVF.");
        auto invlists = dynamic_cast<faiss::DynamicInvertedLists *>(casted_index->invlists);

        int64_t id_offset = invlists->curr_list_id_;
        Tensor centroid_ids = id_offset + torch::arange(centroids.size(0), torch::kInt64);
        parent_->add(centroids, centroid_ids);

        auto centroid_ids_accessor = centroid_ids.accessor<int64_t, 1>();

        // add empty partitions to the invlists
#ifdef QUAKE_NUMA
        int curr_numa_node = 0;
        int num_numa_nodes = this->get_num_numa_nodes();
#endif
        for (int i = 0; i < centroids.size(0); i++) {
            size_t new_list_id = invlists->get_new_list_id();
            invlists->add_list(new_list_id);
#ifdef QUAKE_NUMA
            if(using_numa_optimizations_) {
                invlists->set_numa_node(new_list_id, curr_numa_node);
                curr_numa_node = (curr_numa_node + 1) % num_numa_nodes;
            }
#endif
        }
        casted_index->nlist += centroids.size(0);

        return centroid_ids;
    } else {
        throw std::runtime_error("Top level does not have a parent");
    }
}

void DynamicIVF_C::add_level(int nlist) {
    if (parent_->parent_ != nullptr) {
        parent_->add_level(nlist);
    } else {
        // Get the centroids from the previous parent
        Tensor centroid_vectors; Tensor centroid_ids;
        std::tie(centroid_vectors, centroid_ids) = parent_->get_vectors_and_ids();

        // Create the parent add in the centroids
        parent_ = std::make_shared<DynamicIVF_C>(d_, nlist, index_->metric_type, num_scan_workers_, num_codebooks_,
            code_size_, using_numa_optimizations_, log_mode_, verify_numa_, same_core_, use_centroid_workers_, use_adpative_nprobe_);
        parent_->curr_depth_ = curr_depth_ + 1;
        parent_->build(centroid_vectors, centroid_ids);
    }
}

void DynamicIVF_C::remove_level() {
    // If we are 1 or 2 level index then do nothing (IndexFlat or IndexIVFFlat -> IndexFlat)
    if(parent_ == nullptr || parent_->parent_ == nullptr) {
        return;
    }

    // Ensure we are in the scenario that our grandparent is a IndexFlat (IndexIVFFlat -> IndexIVFFlat -> IndexFlat)
    auto our_grandparent = parent_->parent_;
    if(our_grandparent->parent_ != nullptr) {
        return parent_->remove_level();
    }

    // Get the vectors and ids of the parent
    Tensor parent_vectors; Tensor parent_ids;
    std::tie(parent_vectors, parent_ids) = parent_->get_vectors_and_ids();

    // Replace the parent having a IndexIVFFlat with a IndexFlat
    parent_ = std::make_shared<DynamicIVF_C>(d_, 0, index_->metric_type, num_scan_workers_, num_codebooks_, code_size_,
            using_numa_optimizations_, log_mode_, verify_numa_, same_core_, use_centroid_workers_, use_adpative_nprobe_);
    parent_->curr_depth_ = curr_depth_ + 1;
    parent_->build(parent_vectors, parent_ids, false, true);
}

void DynamicIVF_C::build_given_centroids(Tensor centroids, Tensor x, Tensor ids) {
    faiss::MetricType metric_type = index_->metric_type;
    int dimension = centroids.size(1);

    // Build the quantizer with the given centroids
    auto quantizer = new faiss::IndexFlat(dimension, metric_type);
    quantizer->add(centroids.size(0), centroids.data_ptr<float>());
    quantizer->is_trained = true;

    if (use_refine_) {
        delete refine_index_;
    }
    delete index_;

    // Depending on use_refine_, create the appropriate index
    if (use_refine_) {
        index_ = new faiss::IndexIDMap2(new faiss::IndexIVFPQ(quantizer, dimension, quantizer->ntotal, num_codebooks_, code_size_));
        ivf_index_ = dynamic_cast<faiss::IndexIVF *>(dynamic_cast<faiss::IndexIDMap2 *>(index_)->index);
        refine_index_ = new faiss::IndexRefineFlat(ivf_index_);
    } else {
        index_ = new faiss::IndexIDMap2(new faiss::IndexIVFFlat(quantizer, dimension, quantizer->ntotal, metric_type));
        ivf_index_ = dynamic_cast<faiss::IndexIVF *>(dynamic_cast<faiss::IndexIDMap2 *>(index_)->index);
    }
    index_->verbose = false;

#ifdef FAISS_ENABLE_GPU
    build_index_on_gpu(x, ids);
#else
    build_index_on_cpu(x, ids);
#endif
}

BuildTimingInfo DynamicIVF_C::build(Tensor x, Tensor ids, bool build_parent, bool launch_workers) {
    x = x.contiguous().clone();
    ids = ids.contiguous();

    BuildTimingInfo timing_info;
    timing_info.n_vectors = x.size(0);
    timing_info.n_clusters = nlist();
    timing_info.d = d_;
    timing_info.num_codebooks = num_codebooks_;
    timing_info.code_size = code_size_;

    auto start_time = std::chrono::high_resolution_clock::now();

#ifdef FAISS_ENABLE_GPU
    timing_info = build_index_on_gpu(x, ids);
#else
    timing_info = build_index_on_cpu(x, ids);
#endif

    // If we have a centroid store then populate it as well
    int curr_nlist = nlist();
    if(centroid_store_ != nullptr) {
        std::tuple<Tensor, Tensor> curr_centroids = get_vectors_and_ids(false);
        Tensor curr_vectors = std::get<0>(curr_centroids).contiguous();
        Tensor curr_ids = std::get<1>(curr_centroids).contiguous();
        centroid_store_->add_centroids(curr_vectors.size(0), curr_vectors.data_ptr<float>(), curr_ids.data_ptr<idx_t>());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    timing_info.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    if (build_parent || curr_nlist > 1) {
        // get centroids from quantizer
        Tensor cents = torch::empty({curr_nlist, d_}, torch::kFloat32);
        ivf_index_->quantizer->reconstruct_n(0, curr_nlist, cents.data_ptr<float>());

        parent_ = std::make_shared<DynamicIVF_C>(d_, 0, index_->metric_type, num_scan_workers_, num_codebooks_, code_size_,
            using_numa_optimizations_, log_mode_, verify_numa_, same_core_, use_centroid_workers_, use_adpative_nprobe_);
        parent_->curr_depth_ = curr_depth_ + 1;
        parent_->build(cents, torch::arange(curr_nlist, torch::kInt64), false, false);
    }

    // create maintenance policy
    maintenance_policy_ = std::make_shared<QueryCostMaintenance>(shared_from_this());
    if(launch_workers) {
        this->launch_cluster_scan_workers();
    }

    return timing_info;
}

void DynamicIVF_C::rebuild(int new_nlist) {
    Tensor vectors;
    Tensor ids;
    std::tie(vectors, ids) = get_vectors_and_ids();

#ifdef FAISS_ENABLE_GPU
    rebuild_index_on_gpu(vectors, ids, new_nlist);
#else
    rebuild_index_on_cpu(vectors, ids, new_nlist);
#endif
}

void DynamicIVF_C::save(const std::string &path) {
    // replace the dynamic inverted lists with on array inverted lists
    auto casted_index = ivf_index_;
    if (casted_index != nullptr) {
        auto invlists = casted_index->invlists;
        std::unordered_map<size_t, size_t> old_to_new_ids;
        auto array_invlists = convert_to_array_invlists(dynamic_cast<faiss::DynamicInvertedLists *>(invlists), old_to_new_ids);
        casted_index->replace_invlists(array_invlists, true);

        // First remove all of the centroids
        auto result = parent_->get_vectors_and_ids();
        Tensor centroid_vectors = std::get<0>(result); Tensor centroid_ids = std::get<1>(result);

        int64_t* ids_ptr = centroid_ids.data_ptr<int64_t>();
        int num_centroids = centroid_ids.size(0);
        parent_->remove(centroid_ids, false);

        // Now get the new ids
        for(int i = 0; i < num_centroids; i++) {
            size_t prev_id = static_cast<size_t>(ids_ptr[i]);
            ids_ptr[i] = static_cast<int64_t>(old_to_new_ids[prev_id]);
        }

        // Reinsert all of the vectors
        parent_->add(centroid_vectors, centroid_ids);
    }

    if (use_refine_) {
        faiss::write_index(refine_index_, path.c_str());
    } else {
        faiss::write_index(index_, path.c_str());
    }

    if (parent_ != nullptr) {
        parent_->save(path + ".parent");
    }
}

void DynamicIVF_C::load(const std::string &path, bool launch_workers) {

    index_ = faiss::read_index(path.c_str());
    // convert array inverted lists to dynamic inverted lists
    ivf_index_ = dynamic_cast<faiss::IndexIVF*>(index_);
    if (ivf_index_ != nullptr) {
        auto invlists = ivf_index_->invlists;
        auto dynamic_invlists = faiss::convert_from_array_invlists(dynamic_cast<faiss::ArrayInvertedLists *>(invlists));
        ivf_index_->replace_invlists(dynamic_invlists, true);
        centroid_store_ = nullptr;
    } else {
        // Populate the dynamic centroid store
        std::tuple<Tensor, Tensor> curr_centroids = get_vectors_and_ids(false);
        Tensor curr_vectors = std::get<0>(curr_centroids).contiguous();
        Tensor curr_ids = std::get<1>(curr_centroids).contiguous();
        centroid_store_->add_centroids(curr_vectors.size(0), curr_vectors.data_ptr<float>(), curr_ids.data_ptr<idx_t>());
    }

    d_ = index_->d;
#ifdef QUAKE_NUMA
    int num_numa_nodes = get_num_numa_nodes();
    curr_queries_per_node_.reserve(num_numa_nodes);
    int total_buffer_size = QUERY_BUFFER_REUSE_THRESHOLD * d_;
    for(int i = 0; i < num_numa_nodes; i++) {
        if(total_buffer_size > 0) {
            curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(total_buffer_size * sizeof(float), i));
            if(curr_queries_per_node_[i] == NULL) {
                throw std::runtime_error("Unable to allocate vector on numa node");
            }
        } else {
            curr_queries_per_node_[i] = nullptr;
        }
    }
#endif

    if (std::filesystem::exists(path + ".parent")) {
        parent_ = std::make_shared<DynamicIVF_C>(d_, 0, index_->metric_type, num_scan_workers_, num_codebooks_, code_size_,
            using_numa_optimizations_, log_mode_, verify_numa_, same_core_, use_centroid_workers_, use_adpative_nprobe_);
        parent_->curr_depth_ = curr_depth_ + 1;
        parent_->load(path + ".parent", false);
    }

    maintenance_policy_ = std::make_shared<QueryCostMaintenance>(shared_from_this());

    if(launch_workers) {
        this->launch_cluster_scan_workers();
    }
}

ModifyTimingInfo DynamicIVF_C::add(Tensor x, Tensor ids, bool call_maintenance) {

    ModifyTimingInfo modify_timing_info;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto casted_index = ivf_index_;

    if (casted_index == nullptr) {
        index_->add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if(centroid_store_ != nullptr) {
            centroid_store_->add_centroids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
        }
        modify_timing_info.modify_time_us = duration;
    } else {
        start_time = std::chrono::high_resolution_clock::now();
        Tensor partition_ids;
        Tensor distances;
        shared_ptr<SearchTimingInfo> timing_info = std::make_shared<SearchTimingInfo>();

        std::tie(partition_ids, distances, timing_info) = search_quantizer(x, 1, 10, 1.0);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        modify_timing_info.find_partition_time_us = duration;

        start_time = std::chrono::high_resolution_clock::now();
        casted_index->add_core(x.size(0),
                               x.data_ptr<float>(),
                               ids.data_ptr<int64_t>(),
                               partition_ids.data_ptr<int64_t>());
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        modify_timing_info.modify_time_us = duration;

        start_time = std::chrono::high_resolution_clock::now();
        if (call_maintenance) {
            maintenance_policy_->maintenance();
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        modify_timing_info.maintenance_time_us = duration;
    }
    modify_timing_info.n_vectors = x.size(0);
    return modify_timing_info;
}

ModifyTimingInfo DynamicIVF_C::remove(Tensor ids, bool call_maintenance) {
    ModifyTimingInfo modify_timing_info;
    auto start_time = std::chrono::high_resolution_clock::now();

    if(centroid_store_ != nullptr) {
        int prev_count = centroid_store_->total_centroids_;
        std::set<idx_t> ids_to_remove;
        int num_ids = ids.size(0);
        int64_t* ids_data = ids.contiguous().data_ptr<int64_t>();
        for(int i = 0; i < num_ids; i++) {
            ids_to_remove.insert(ids_data[i]);
        }
        centroid_store_->remove_centroids(ids_to_remove);
        index_->remove_ids(faiss::IDSelectorArray(ids.size(0), ids.data_ptr<int64_t>()));
    } else {
        auto invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index_->invlists);
        int num_ids = ids.size(0);
        int64_t* ids_data = ids.contiguous().data_ptr<int64_t>();
        std::set<idx_t> ids_to_remove;
        for(int i = 0; i < num_ids; i++) {
            ids_to_remove.insert(static_cast<idx_t>(ids_data[i]));
        }
        invlists->remove_vectors(ids_to_remove);
        index_->ntotal -= num_ids;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    modify_timing_info.modify_time_us = duration;

    start_time = std::chrono::high_resolution_clock::now();
    if (call_maintenance) {
        maintenance_policy_->maintenance();
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    modify_timing_info.maintenance_time_us = duration;
    modify_timing_info.n_vectors = ids.size(0);
    return modify_timing_info;
}

ModifyTimingInfo DynamicIVF_C::modify(Tensor x, Tensor ids, bool call_maintenance) {

    ModifyTimingInfo modify_timing_info;
    auto start_time = std::chrono::high_resolution_clock::now();
    index_->remove_ids(faiss::IDSelectorArray(ids.size(0), ids.data_ptr<int64_t>()));
    add(x, ids, false);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    modify_timing_info.modify_time_us = duration;

    start_time = std::chrono::high_resolution_clock::now();
    if (call_maintenance) {
        maintenance_policy_->maintenance();
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    modify_timing_info.maintenance_time_us = duration;
    modify_timing_info.n_vectors = x.size(0);
    return modify_timing_info;
}

MaintenanceTimingInfo DynamicIVF_C::maintenance() {
    if (parent_ != nullptr) {
        return maintenance_policy_->maintenance();
    } else {
        return MaintenanceTimingInfo();
    }
}


Tensor DynamicIVF_C::get_cluster_sizes() const {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kInt64);
    } else {
        auto invlists = casted_index->invlists;
        Tensor cluster_sizes = torch::empty({nlist()}, torch::kInt64);

        // get partition ids
        Tensor partition_ids = get_partition_ids();
        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids[i].item<int64_t>();
            cluster_sizes[i] = static_cast<int64_t>(invlists->list_size(list_no));
        }
        return cluster_sizes;
    }
}

std::tuple<Tensor, vector<Tensor>, vector<Tensor> > DynamicIVF_C::select_clusters(Tensor select_ids, bool copy) {
    Tensor centroids = parent_->select_vectors(select_ids);
    vector<Tensor> cluster_vectors;
    vector<Tensor> cluster_ids;
    auto casted_index = ivf_index_;

    if (casted_index == nullptr) {
        throw std::runtime_error("Index must be of type IndexIVF.");
    }

    auto invlists = casted_index->invlists;

    for (int i = 0; i < select_ids.size(0); i++) {
        int64_t list_no = select_ids[i].item<int64_t>();

        int64_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            cluster_vectors.push_back(torch::empty({0, d_}, torch::kFloat32));
            cluster_ids.push_back(torch::empty({0}, torch::kInt64));
            continue;
        }

        auto codes = invlists->get_codes(list_no);
        auto ids = invlists->get_ids(list_no);

        Tensor cluster_vectors_i = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);
        Tensor cluster_ids_i = torch::from_blob((void *) ids, {list_size}, torch::kInt64);

        if (copy) {
            cluster_vectors_i = cluster_vectors_i.clone();
            cluster_ids_i = cluster_ids_i.clone();
        }

        cluster_vectors.push_back(cluster_vectors_i);
        cluster_ids.push_back(cluster_ids_i);
    }

    return std::make_tuple(centroids, cluster_vectors, cluster_ids);
}

Tensor DynamicIVF_C::select_vectors(Tensor ids) {
    Tensor vectors = torch::empty({ids.size(0), d_}, torch::kFloat32);
    float *vectors_ptr = vectors.data_ptr<float>();
    auto ids_accessor = ids.contiguous().data_ptr<int64_t>();
    int64_t n = ids.size(0);

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        int64_t search_id = ids_accessor[i];
        float* vector_write_ptr = vectors_ptr + i * d_;
        if(ivf_index_ != nullptr) {
            auto dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index_->invlists);
            if(!dynamic_invlists->get_vector_for_id(search_id, vector_write_ptr)) {
                std::string err_msg = "DynamicInvertedLists at level " + std::to_string(curr_depth_) + " failed to find vector with id " + std::to_string(search_id);
                throw std::runtime_error(err_msg);
            }
        } else {
            index_->reconstruct(search_id, vector_write_ptr);
        }
        // } else if (!centroid_store_->get_vector_for_id(search_id, vector_write_ptr)) {
        //     std::string err_msg = "DynamicCentroidStore at level " + std::to_string(curr_depth_) + " failed to find vector with id " + std::to_string(search_id);
        //     throw std::runtime_error(err_msg);
        // }
    }
    return vectors;
}


void DynamicIVF_C::recompute_centroids(Tensor ids) {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return;
    }

    auto invlists = casted_index->invlists;

    // Cast to IndexFlat
    faiss::IndexFlat *quantizer = dynamic_cast<faiss::IndexFlat *>(casted_index->quantizer);
    Tensor centroids = torch::from_blob((void *) quantizer->get_xb(), {nlist(), d_}, torch::kFloat32);

    if (ids.defined() && ids.numel() > 0) {
        for (int i = 0; i < ids.size(0); i++) {
            int64_t list_no = ids[i].item<int64_t>();
            int64_t list_size = invlists->list_size(list_no);
            if (list_size == 0) {
                continue;
            }
            auto codes = invlists->get_codes(list_no);
            Tensor cluster_vectors_i = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);

            // Compute mean
            Tensor mean = cluster_vectors_i.mean(0);

            // Update centroid in place
            centroids[list_no] = mean;
        }
    } else {
        for (int i = 0; i < nlist(); i++) {
            int64_t list_no = i;
            int64_t list_size = invlists->list_size(list_no);
            if (list_size == 0) {
                continue;
            }
            auto codes = invlists->get_codes(list_no);
            Tensor cluster_vectors_i = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);

            // Compute mean
            Tensor mean = cluster_vectors_i.mean(0);

            // Update centroid in place
            centroids[i] = mean;
        }
    }
}

Tensor DynamicIVF_C::get_cluster_ids() const {
    // For each vector, get the cluster ID
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kInt64);
    }

    auto invlists = casted_index->invlists;

    int64_t total_vectors = invlists->compute_ntotal();
    Tensor cluster_ids = torch::empty({total_vectors}, torch::kInt64);
    int64_t offset = 0;

    for (int i = 0; i < nlist(); i++) {
        int64_t list_size = invlists->list_size(i);
        if (list_size == 0) {
            continue;
        }
        cluster_ids.slice(0, offset, offset + list_size).fill_(i);
        offset += list_size;
    }
    return cluster_ids;
}

int DynamicIVF_C::get_nprobe_for_recall_target(Tensor x, int k, float recall_target) {
    // Ensure input tensor is contiguous
    x = x.contiguous();
    int num_queries = x.size(0);

    // Get ground truth by exhaustive search over quantizer centroids
    Tensor gt_labels = torch::empty({num_queries, k}, torch::kInt64);
    Tensor gt_distances = torch::empty({num_queries, k}, torch::kFloat32);

    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return 1;
    }

    // Perform exhaustive search over quantizer centroids
    casted_index->quantizer->search(num_queries, x.data_ptr<float>(), k, gt_distances.data_ptr<float>(),
                                    gt_labels.data_ptr<int64_t>());

    // Initialize binary search for nprobe
    int min_nprobe = 1;
    int max_nprobe = nlist();

    int best_nprobe = max_nprobe;

    while (min_nprobe <= max_nprobe) {
        int nprobe = (min_nprobe + max_nprobe) / 2;
        Tensor distances;
        Tensor labels;
        shared_ptr<SearchTimingInfo> timing_info;

        std::tie(labels, distances, timing_info) = search(x, nprobe, k);
        Tensor recalls = calculate_recall(labels, gt_labels);
        float recall = recalls.mean().item<float>();

        if (recall >= recall_target) {
            // Achieved desired recall; try to find smaller nprobe
            best_nprobe = nprobe;
            max_nprobe = nprobe - 1;
        } else {
            // Not enough recall; increase nprobe
            min_nprobe = nprobe + 1;
        }
    }

    // Return the minimal nprobe that achieves the recall target
    return best_nprobe;
}

void DynamicIVF_C::search_all_centroids(Tensor x, int k, int64_t* ret_ids, float* ret_dis, shared_ptr<SearchTimingInfo> timing_info) {
    auto func_start_time = std::chrono::high_resolution_clock::now();
    auto search_pre_start = std::chrono::high_resolution_clock::now();

    // Record the query data
    int vector_dimension = d_;
    max_vectors_per_cluster_ = k;
    int prev_num_vectors = num_search_vectors_;
    num_search_vectors_ = x.size(0);
    float* x_ptr = x.contiguous().data_ptr<float>();

    // Create and reset intermediate buffer per query
    bool is_descending = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
    int query_buffer_capacity = max_vectors_per_cluster_ * num_scan_workers_;
    while(final_result_mergers_.size() < num_search_vectors_) {
        final_result_mergers_.emplace_back(max_vectors_per_cluster_, is_descending);
    }
    for(int i = 0; i < num_search_vectors_; i++) {
        final_result_mergers_[i].set_k(max_vectors_per_cluster_);
        final_result_mergers_[i].reset();
    }

    // Create the buffer to hold the job details
    int* prev_job_queries_buffer = this->job_query_id_;
    this->job_query_id_ = new int[num_search_vectors_ * num_scan_workers_];
    if(prev_job_queries_buffer != nullptr) {
        delete[] prev_job_queries_buffer;
    }

    // Initialize the buffer
    std::memset(all_vectors_scanned_ptr_, 0, num_scan_workers_ * sizeof(int));
    if(log_mode_) {
        std::memset(all_counts_ptr_, 0, num_scan_workers_ * sizeof(int));
        std::memset(all_job_times_ptr_, 0, num_scan_workers_ * sizeof(int));
        std::memset(all_scan_times_ptr_, 0, num_scan_workers_ * sizeof(int));
        std::memset(all_throughputs_ptr_, 0, num_scan_workers_ * sizeof(float));
    }

    // Populate the write offsets for the workers
    auto search_pre_end = std::chrono::high_resolution_clock::now();

    // See if we need to create tempory numa buffer
    int temp_buffer_size;
    std::vector<float*> numa_reuse_buffers;
    if(using_numa_optimizations_ && num_search_vectors_ > QUERY_BUFFER_REUSE_THRESHOLD) {
#ifdef QUAKE_NUMA
        int num_numa_nodes = this->get_num_numa_nodes();
        numa_reuse_buffers.reserve(num_numa_nodes);
        temp_buffer_size = num_search_vectors_ * vector_dimension;
        for(int i = 0; i < num_numa_nodes; i++) {
            // Save the pointer to the old buffer
            numa_reuse_buffers[i] = curr_queries_per_node_[i];

            // Allocate the new buffer
            curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(temp_buffer_size * sizeof(float), i));
            if(curr_queries_per_node_[i] == NULL) {
                throw std::runtime_error("Unable to allocate vector on numa node");
            }
        }
#endif
    }
    timing_info->partition_scan_setup_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_pre_end - search_pre_start).count();

    auto search_start_time = std::chrono::high_resolution_clock::now();
    if(using_numa_optimizations_) {
#ifdef QUAKE_NUMA
        // Copy over the vector into each numa node
        timing_info->using_numa = true;
        auto numa_pre_start = std::chrono::high_resolution_clock::now();
        int num_numa_nodes = this->get_num_numa_nodes();
        int prev_vector_buffer_size = prev_num_vectors * vector_dimension;
        int vector_buffer_size = num_search_vectors_ * vector_dimension;
        for(int i = 0; i < num_numa_nodes; i++) {
            // Copy the vector into the buffer
            std::memcpy(curr_queries_per_node_[i], x_ptr, vector_buffer_size * sizeof(float));
        }
        auto numa_pre_end = std::chrono::high_resolution_clock::now();
        timing_info->total_numa_preprocessing_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_pre_end - numa_pre_start).count();

        // Inform the workers that a query is avaliable and wait for them to complete
        auto numa_distribute_start = std::chrono::high_resolution_clock::now();
        int curr_job_id = 0;
        for (int i = 0; i < num_search_vectors_; i++) {
            TopKVectors& curr_query_buffer = final_result_mergers_[i];
            curr_query_buffer.set_processing_query(true);
            curr_query_buffer.set_jobs_left(num_scan_workers_);

            for(int worker_id = 0; worker_id < num_scan_workers_; worker_id++) {
                // Submit the job
                curr_job_id = i * num_scan_workers_ + worker_id;
                this->job_query_id_[curr_job_id] = i;
                jobs_queue_[worker_id].enqueue(curr_job_id);
            }
        }
        auto numa_distribute_end = std::chrono::high_resolution_clock::now();

        // Wait until all threads have processed the query
        auto numa_wait_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_search_vectors_; i++) {
            TopKVectors& curr_query_buffer = final_result_mergers_[i];
            while(!curr_query_buffer.finished_all_jobs()) {
                std::this_thread::yield();
            }
            curr_query_buffer.set_processing_query(false);
        }
        auto numa_wait_end = std::chrono::high_resolution_clock::now();

        // Record the timings
        timing_info->total_job_distribute_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_distribute_end - numa_distribute_start).count();
        timing_info->total_result_wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_wait_end - numa_wait_start).count();
#endif
    } else {
        timing_info->using_numa = false;
        faiss::MetricType metric_type = index_->metric_type;

        for(int i = 0; i < num_search_vectors_; i++) {
            // Get the ptrs to write the intermediate results
            float* query_vector = x_ptr + i * vector_dimension;
            TopKVectors& curr_query_buffer = final_result_mergers_[i];

#pragma omp parallel for num_threads(num_scan_workers_)
            for(int j = 0; j < num_scan_workers_; j++) {
                auto worker_query_start_time = std::chrono::high_resolution_clock::now();

                // Get the thread id
                int thread_id = omp_get_thread_num();
                int worker_num_centroids = centroid_store_->num_centroids_for_worker(thread_id);
                float* worker_centroid_vectors = centroid_store_->get_vectors_for_worker(thread_id);
                idx_t* worker_centroid_ids = centroid_store_->get_ids_for_worker(thread_id);
                TopKVectors& curr_cluster_buffer = intermediate_results_buffer_[thread_id];
                curr_cluster_buffer.set_k(max_vectors_per_cluster_);

                // Perform the actual scan
                auto worker_scan_start_time = std::chrono::high_resolution_clock::now();
                scan_list(query_vector, worker_centroid_vectors, worker_centroid_ids, worker_num_centroids, vector_dimension, curr_cluster_buffer, metric_type);
                auto worker_scan_end_time = std::chrono::high_resolution_clock::now();

                // Write the number of results
                std::vector<int64_t> overall_top_ids = curr_cluster_buffer.get_topk_indices();
                std::vector<float> overall_top_dists = curr_cluster_buffer.get_topk();
                int num_results = std::min((int) overall_top_ids.size(), max_vectors_per_cluster_);

                // Also copy over the distances and ids while properly converting the types
                curr_query_buffer.batch_add(overall_top_dists.data(), overall_top_ids.data(), num_results);
                all_vectors_scanned_ptr_[thread_id] += worker_num_centroids;

                if(log_mode_) {
                    auto worker_query_end_time = std::chrono::high_resolution_clock::now();

                    // Record the worker summary
                    int total_scan_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_scan_end_time - worker_scan_start_time).count();
                    all_scan_times_ptr_[thread_id] += total_scan_time;
                    size_t memory_scanned = ((size_t) worker_num_centroids) * (vector_dimension * sizeof(float) + sizeof(idx_t));
                    all_throughputs_ptr_[thread_id] += ((1.0 * memory_scanned)/pow(10.0, 9))/((1.0 * total_scan_time)/pow(10.0, 6));

                    // Record the end to end time
                    int total_job_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_query_end_time - worker_query_start_time).count();
                    all_job_times_ptr_[thread_id] += total_job_time;
                    all_counts_ptr_[thread_id] += 1;
                }
            }
        }
    }
    auto search_end_time = std::chrono::high_resolution_clock::now();
    timing_info->partition_scan_search_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_end_time - search_start_time).count();
    timing_info->target_vectors_scanned = num_search_vectors_ * centroid_store_->total_centroids_;

    // Write the combined results
    auto search_post_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_search_vectors_; i++) {
        TopKVectors& curr_query_buffer = final_result_mergers_[i];

        // Copy the result to the final result
        float* final_dist_writer = ret_dis + i * max_vectors_per_cluster_;
        int64_t* final_ids_writer = ret_ids + i * max_vectors_per_cluster_;
        std::vector<float> topk_dists = curr_query_buffer.get_topk();
        std::vector<int64_t> topk_ids = curr_query_buffer.get_topk_indices();
        int num_results = std::min((int) topk_ids.size(), max_vectors_per_cluster_);
        std::memcpy(final_dist_writer, topk_dists.data(), num_results * sizeof(float));
        std::memcpy(final_ids_writer, topk_ids.data(), num_results * sizeof(int64_t));
    }

    // See if we need to free up the temporary buffer
    if(using_numa_optimizations_ && num_search_vectors_ > QUERY_BUFFER_REUSE_THRESHOLD) {
#ifdef QUAKE_NUMA
        int num_numa_nodes = this->get_num_numa_nodes();
        for(int i = 0; i < num_numa_nodes; i++) {
            // Free the temporary buffer
            numa_free(curr_queries_per_node_[i], temp_buffer_size * sizeof(float));
            curr_queries_per_node_[i] = numa_reuse_buffers[i];
        }
#endif
    }

    auto search_post_end = std::chrono::high_resolution_clock::now();
    timing_info->partition_scan_post_process_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_post_end - search_post_start).count();

    auto func_end_time = std::chrono::high_resolution_clock::now();
    timing_info->partition_scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(func_end_time - func_start_time).count();

    // Also record the total worker times
    if(log_mode_) {
        int total_records = 0;
        for(int i = 0; i < num_scan_workers_; i++) {
            timing_info->average_worker_job_time_us += this->all_job_times_ptr_[i];
            timing_info->average_worker_scan_time_us += this->all_scan_times_ptr_[i];
            timing_info->average_worker_throughput += this->all_throughputs_ptr_[i];
            total_records += this->all_counts_ptr_[i];
        }

        if(total_records > 0) {
            timing_info->average_worker_job_time_us /= total_records;
            timing_info->average_worker_scan_time_us /= total_records;
            timing_info->average_worker_throughput /= total_records;
        }
    }

    for(int i = 0; i < num_scan_workers_; i++) {
        timing_info->total_vectors_scanned += this->all_vectors_scanned_ptr_[i];
    }
}

std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > DynamicIVF_C::search_one(
    Tensor query, int k, float recall_target, int nprobe, float recompute_threshold, bool use_precomputed) {
    recall_target = recall_target - .001;

    shared_ptr<SearchTimingInfo> timing_info = std::make_shared<SearchTimingInfo>();
    shared_ptr<SearchTimingInfo> parent_timing_info = nullptr;

    timing_info->n_queries = 1;
    timing_info->n_vectors = ntotal();
    timing_info->n_clusters = nlist();
    timing_info->d = d_;
    timing_info->num_codebooks = num_codebooks_;
    timing_info->code_size = code_size_;
    timing_info->k = k;
    timing_info->recall_target = recall_target;

    auto start_time = std::chrono::high_resolution_clock::now();
    bool is_descending = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
    bool euclidean = index_->metric_type == faiss::METRIC_L2;
    auto topk_buffer = TopkBuffer(k, is_descending);
    float *query_vector = query.data_ptr<float>();

    // Perform search if the index is not partitioned
    if (parent_ == nullptr) {
        auto ret_ids = torch::ones({k}, torch::kInt64);
        auto ret_dis = torch::empty({k}, torch::kFloat32);
        index_->search(1, query_vector, k, ret_dis.data_ptr<float>(), ret_ids.data_ptr<int64_t>());
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        timing_info->total_time_us = duration;
        return std::make_tuple(ret_ids, ret_dis, timing_info);
    } else {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto t2 = std::chrono::high_resolution_clock::now();

        int initial_nprobe;

        bool use_recall_target = nprobe == -1;
        if (use_recall_target) {
            float nprobe_factor = .05;
            if (recall_target > .95) {
                nprobe_factor = .1;
            }
            initial_nprobe = std::max((int) (nprobe_factor * nlist()), 1); // TODO automatically determine the initial nprobe
        } else {
            initial_nprobe = nprobe;
        }
        int recall_profile_duration_us = 0;
        int boundary_distance_duration_us = 0;
        int partition_scan_duration_us = 0;

        timing_info->nprobe = initial_nprobe;

        // index is partitioned
        t1 = std::chrono::high_resolution_clock::now();
        Tensor cluster_ret_ids;
        Tensor cluster_ret_dis;
        std::tie(cluster_ret_ids, cluster_ret_dis, parent_timing_info) = parent_->search_one(query,
            initial_nprobe,
            recall_target);
        t2 = std::chrono::high_resolution_clock::now();
        timing_info->quantizer_search_time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        // scan the nearest partition
        t1 = std::chrono::high_resolution_clock::now();
        auto scan_partition_start = std::chrono::high_resolution_clock::now();
        int64_t list_no = cluster_ret_ids[0].item<int64_t>();
        auto casted_index = dynamic_cast<faiss::IndexIVF *>(index_);
        auto invlists = casted_index->invlists;
        int64_t list_size = invlists->list_size(list_no);

        scan_list(query_vector,
                  (float *) invlists->get_codes(list_no),
                  invlists->get_ids(list_no),
                  list_size,
                  d_,
                  topk_buffer,
                  casted_index->metric_type);
        t2 = std::chrono::high_resolution_clock::now();
        partition_scan_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        float query_radius = topk_buffer.get_kth_distance();
        if (casted_index->metric_type == faiss::METRIC_INNER_PRODUCT) {
            query_radius = acos(query_radius);
        } else {
            query_radius = sqrt(query_radius);
        }

        // chop off any -1 cluster ids
        cluster_ret_ids = cluster_ret_ids.masked_select(cluster_ret_ids.ge(0));

        if (use_recall_target) {
            // estimate the recall profile using:
            // 1. the distance to the k-th nearest neighbor (this may change every time a partition is scanned)
            // 2. the distance of the query to the boundaries of the clusters (this is fixed)
            t1 = std::chrono::high_resolution_clock::now();
            Tensor cluster_centroids = parent_->select_vectors(cluster_ret_ids);

            Tensor boundary_distances = compute_boundary_distances(query.flatten(), cluster_centroids, euclidean);

            t2 = std::chrono::high_resolution_clock::now();
            boundary_distance_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            // Reorder clusters by their distance to boundary (seems to make things worse?)
            // Tensor sort_args = torch::argsort(boundary_distances);
            // cluster_ret_ids = cluster_ret_ids.index_select(0, sort_args);
            // boundary_distances = boundary_distances.index_select(0, sort_args);

            // compute recall profile
            t1 = std::chrono::high_resolution_clock::now();

            Tensor partition_probs = compute_recall_profile(boundary_distances, query_radius, d_, {}, use_precomputed, euclidean);
            Tensor recall_profile = torch::cumsum(partition_probs, 0);

            t2 = std::chrono::high_resolution_clock::now();
            recall_profile_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

            int compute_count = 1;
            if (recall_profile[0].item<float>() < recall_target) {
                for (int i = 1; i < cluster_ret_ids.size(0); i++) {
                    t1 = std::chrono::high_resolution_clock::now();
                    list_no = cluster_ret_ids[i].item<int64_t>();
                    list_size = invlists->list_size(list_no);
                    if (list_size == 0) {
                        continue;
                    }

                    scan_list(query_vector,
                              (float *) invlists->get_codes(list_no),
                              invlists->get_ids(list_no),
                              list_size,
                              d_,
                              topk_buffer,
                              casted_index->metric_type);
                    t2 = std::chrono::high_resolution_clock::now();
                    partition_scan_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

                    float curr_radius = sqrt(topk_buffer.get_kth_distance());

                    // compute recall profile if the radius has changed
                    t1 = std::chrono::high_resolution_clock::now();
                    // check if radius has changed beyond the threshold

                    float percent_change = abs(curr_radius - query_radius) / query_radius;

                    if (percent_change > recompute_threshold) {
                        compute_count++;
                        query_radius = topk_buffer.get_kth_distance();

                        // if the metric is inner product then we need to convert the inner product to an l2 distance (assume vectors have been normalized)
                        if (casted_index->metric_type == faiss::METRIC_INNER_PRODUCT) {
                            query_radius = acos(query_radius);
                        } else {
                            query_radius = sqrt(query_radius);
                        }

                        partition_probs = compute_recall_profile(boundary_distances, query_radius, d_, {}, use_precomputed, euclidean);
                        // partition_probs = partition_probs * variance_in_direction_of_query;
                        // partition_probs /= partition_probs.sum();
                        recall_profile = torch::cumsum(partition_probs, 0);
                    }
                    t2 = std::chrono::high_resolution_clock::now();
                    recall_profile_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

                    if (recall_profile[i].item<float>() >= recall_target) {
                        // std::cout << "Recompute count: " << compute_count << " out of " << i + 1 << std::endl;
                        timing_info->nprobe = i + 1;
                        break;
                    }
                }
            } else {
                // std::cout << "Recompute count: " << compute_count << " out of " << 1 << std::endl;
                timing_info->nprobe = 1;
            }
        } else {
            for (int i = 1; i < cluster_ret_ids.size(0); i++) {
                t1 = std::chrono::high_resolution_clock::now();
                list_no = cluster_ret_ids[i].item<int64_t>();
                list_size = invlists->list_size(list_no);
                if (list_size == 0) {
                    continue;
                }

                scan_list(query_vector,
                          (float *) invlists->get_codes(list_no),
                          invlists->get_ids(list_no),
                          list_size,
                          d_,
                          topk_buffer,
                          casted_index->metric_type);
                t2 = std::chrono::high_resolution_clock::now();
                partition_scan_duration_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            }
        }

        // scan the rest of the partitions, early stopping if the recall target is met
        // #pragma omp parallel for

        // print out recall profile
        // std::cout << "Recall profile: " << recall_profile << std::endl;

        vector<int64_t> scanned_partitions = std::vector<int64_t>(timing_info->nprobe);
        auto cluster_ret_ids_accessor = cluster_ret_ids.accessor<int64_t, 1>();
        for (int i = 0; i < timing_info->nprobe; i++) {
            scanned_partitions[i] = cluster_ret_ids_accessor[i];
        }

        if (maintenance_policy_->maintenance_policy_name_ == "query_cost") {
            maintenance_policy_->update_hits(scanned_partitions);
        }

        Tensor ret_ids_tensor = torch::empty({1, k}, torch::kInt64).fill_(-1);
        Tensor ret_dis_tensor = torch::empty({1, k}, torch::kFloat32).fill_(1E9);

        if (index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
            ret_dis_tensor = -ret_dis_tensor;
        }
        int64_t *ret_ids_writer = ret_ids_tensor.data_ptr<int64_t>();
        float *ret_dis_writer = ret_dis_tensor.data_ptr<float>();

        std::vector<float> topk_dists = topk_buffer.get_topk();
        std::vector<int64_t> topk_ids = topk_buffer.get_topk_indices();
        int num_results = std::min((int) topk_ids.size(), k);
        std::memcpy(ret_dis_writer, topk_dists.data(), num_results * sizeof(float));
        std::memcpy(ret_ids_writer, topk_ids.data(), num_results * sizeof(int64_t));

        // Tensor ret_ids_tensor = torch::from_blob(ret_ids_long.data(), {k}, torch::kInt64).clone();
        // Tensor ret_dis_tensor = torch::from_blob(ret_dis.data(), {k}, torch::kFloat64).clone().to(torch::kFloat32);

        timing_info->parent_info = parent_timing_info;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        timing_info->recall_profile_us = recall_profile_duration_us;
        timing_info->boundary_time_us = boundary_distance_duration_us;
        timing_info->partition_scan_time_us = partition_scan_duration_us;
        timing_info->total_time_us = duration;

        return std::make_tuple(ret_ids_tensor.clone(), ret_dis_tensor.clone(), timing_info);
    }
}


std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo>> DynamicIVF_C::search(
    Tensor x, int nprobe, int k, float recall_target, float k_factor, bool use_precomputed) {

    shared_ptr<SearchTimingInfo> quantizer_timing_info;
    shared_ptr<SearchTimingInfo> preassigned_timing_info;
    shared_ptr<SearchTimingInfo> timing_info = std::make_shared<SearchTimingInfo>();

    timing_info->n_queries = x.size(0);
    timing_info->n_vectors = ntotal();
    timing_info->n_clusters = nlist();
    timing_info->d = d_;
    timing_info->num_codebooks = num_codebooks_;
    timing_info->code_size = code_size_;
    timing_info->k = k;
    timing_info->k_factor = k_factor;
    timing_info->recall_target = recall_target;

    auto start_time = std::chrono::high_resolution_clock::now();
    if (parent_ == nullptr) {
        bool using_custom_search = use_centroid_workers_ && workers_initialized_;
        int num_centroids = using_custom_search ? centroid_store_->total_centroids_ : index_->ntotal;
        k = std::min((int) k, num_centroids);
        auto ret_ids = torch::full({x.size(0), k}, -1, torch::kInt64).contiguous();
        auto ret_dis = torch::full({x.size(0), k}, -1.0, torch::kFloat32).contiguous();

        if(using_custom_search) {
            search_all_centroids(x, k, ret_ids.data_ptr<int64_t>(), ret_dis.data_ptr<float>(), timing_info);
        } else {
            index_->search(x.size(0), x.data_ptr<float>(), k, ret_dis.data_ptr<float>(), ret_ids.data_ptr<int64_t>());
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        timing_info->total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        return std::make_tuple(ret_ids, ret_dis, timing_info);
    }

    Tensor cluster_ret_ids;
    Tensor cluster_ret_dis;
    std::tie(cluster_ret_ids, cluster_ret_dis, quantizer_timing_info) = search_quantizer(x, nprobe, nprobe);
    nprobe = cluster_ret_ids.size(1);

    timing_info->parent_info = quantizer_timing_info;
    timing_info->quantizer_search_time_us = quantizer_timing_info->total_time_us;
    timing_info->nprobe = nprobe;

    auto start_metadata_time = std::chrono::high_resolution_clock::now();
    Tensor ret_ids;
    Tensor ret_dis;
    auto cluster_ret_ids_accessor = cluster_ret_ids.contiguous().data_ptr<int64_t>();

    if (maintenance_policy_->maintenance_policy_name_ == "query_cost") {
        for (int i = 0; i < x.size(0); i++) {
            vector<int64_t> scanned_partitions(nprobe);
            int offset = 0;
            for (int j = 0; j < nprobe; j++) {
                int64_t curr_partition = cluster_ret_ids_accessor[i * nprobe + j];

                if(curr_partition != -1) {
                    scanned_partitions[offset++] = curr_partition;
                }
            }
            scanned_partitions.resize(offset);
            maintenance_policy_->update_hits(scanned_partitions);
        }
    }

    auto end_metadata_time = std::chrono::high_resolution_clock::now();
    timing_info->metadata_update_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_metadata_time - start_metadata_time).count();

    auto quantizier_end_time = std::chrono::high_resolution_clock::now();
    int quantizier_duration = std::chrono::duration_cast<std::chrono::microseconds>(quantizier_end_time - start_time).count();

    auto partition_scan_start_time = std::chrono::high_resolution_clock::now();
    int partition_scan_latency = query_latency_target_time_us_ > 0 ? query_latency_target_time_us_ - quantizier_duration : -1;
    std::tie(ret_ids, ret_dis, preassigned_timing_info) = scan_partitions(x, cluster_ret_ids, cluster_ret_dis, k, partition_scan_latency,
        recall_target, k_factor, use_precomputed);
    auto partition_scan_end_time = std::chrono::high_resolution_clock::now();


    // Record the overall time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_info->total_time_us = duration;

    // Copy over the scan timing info
    preassigned_timing_info->copy_partition_scan_info(timing_info);
    timing_info->partition_scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(partition_scan_end_time - partition_scan_start_time).count();
    return std::make_tuple(ret_ids, ret_dis, timing_info);
}

std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > DynamicIVF_C::search_quantizer(
    Tensor x, int k, int nprobe, float recall_target, float k_factor, bool use_gt_to_meet_target, bool use_precomputed) {
    FAISS_ASSERT(k > 0);
    Tensor ret_ids;
    Tensor ret_dis;

    shared_ptr<SearchTimingInfo> timing_info = std::make_shared<SearchTimingInfo>();
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return std::make_tuple(ret_ids, ret_dis, timing_info);
    }

    if (use_gt_to_meet_target && parent_ != nullptr) {
        // Adjust nprobe based on recall target using the parent index
        nprobe = parent_->get_nprobe_for_recall_target(x, k, recall_target) + 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    if (parent_ != nullptr) {
        // Delegate search to the parent index
        std::tie(ret_ids, ret_dis, timing_info) = parent_->search(x, nprobe, k, recall_target, k_factor, use_precomputed);
    } else {
        throw std::runtime_error("Index is not partitioned");
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_info->total_time_us = duration;
    
    return std::make_tuple(ret_ids, ret_dis, timing_info);
}

std::tuple<Tensor, Tensor, shared_ptr<SearchTimingInfo> > DynamicIVF_C::scan_partitions(
    Tensor x, Tensor cluster_ids, Tensor cluster_dists, int k, int target_latency_us,
    float recall_target, float k_factor, int distance_budget, bool use_precomputed) {

    int vector_dimension = d_;
    int prev_num_search_vectors = num_search_vectors_;
    num_search_vectors_ = x.size(0);
    Tensor ret_ids = torch::full({num_search_vectors_, k}, -1, torch::kInt64).contiguous();
    Tensor ret_dis = torch::ones({num_search_vectors_, k}, torch::kFloat32).contiguous();
    const bool use_adaptive_for_search = use_adpative_nprobe_ && recall_target > 0;

    if (index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
        ret_dis.fill_(-1e9);
    } else {
        ret_dis.fill_(1e9);
    }

    shared_ptr<SearchTimingInfo> timing_info = std::make_shared<SearchTimingInfo>();
    timing_info->total_vectors_scanned = 0;
    timing_info->target_vectors_scanned = 0;
    timing_info->using_adpative_nprobe = false;

    /*
    if (distance_budget > 0) {
        // truncate the partitions to scan based on the distance budget
        Tensor partition_sizes = get_partition_sizes(cluster_ids.flatten()).reshape(cluster_ids.sizes());

        // Total scanned vectors as cumulative sum of partition sizes for each query
        Tensor total_scanned_vectors = torch::cumsum(partition_sizes, 1);

        // If the distance budget is less than the total number of vectors in the partition, truncate the partition by setting it to -1
        Tensor partition_mask = total_scanned_vectors <= distance_budget;
        cluster_ids.masked_fill_(~partition_mask, -1);
        cluster_dists.masked_fill_(~partition_mask, -1);
    }
    */

    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        auto timing_info = std::make_shared<SearchTimingInfo>();
        auto x_ptr = x.data_ptr<float>();
        auto ret_ids_ptr = ret_ids.data_ptr<int64_t>();
        auto ret_dis_ptr = ret_dis.data_ptr<float>();
        index_->search(x.size(0), x_ptr, k, ret_dis_ptr, ret_ids_ptr);
        return std::make_tuple(ret_ids, ret_dis, timing_info);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    const bool have_target_latency = target_latency_us > 0;

    if (use_refine_) {
        int n = x.size(0);
        int initial_k = static_cast<int>(k * k_factor);

        auto curr_start_time = std::chrono::high_resolution_clock::now();

        Tensor init_ret_ids = -torch::ones({x.size(0) * initial_k}, torch::kInt64);
        Tensor init_ret_dis = torch::ones({x.size(0) * initial_k}, torch::kFloat32) * 1e9;

        int64_t *cluster_ids_ptr = cluster_ids.data_ptr<int64_t>();
        float *cluster_dists_ptr = cluster_dists.data_ptr<float>();

        float *x_ptr = x.data_ptr<float>();

        // Call search_preassigned on the index_
        casted_index->search_preassigned(
            x.size(0),
            x_ptr,
            initial_k,
            cluster_ids_ptr,
            cluster_dists_ptr,
            init_ret_dis.data_ptr<float>(),
            init_ret_ids.data_ptr<int64_t>(),
            false,
            nullptr);

        auto curr_end_time = std::chrono::high_resolution_clock::now();
        auto duration =
                std::chrono::duration_cast<std::chrono::microseconds>(curr_end_time - curr_start_time)
                .count();

        timing_info->scan_pq_time_us = duration;

        curr_start_time = std::chrono::high_resolution_clock::now();

        auto init_ret_ids_ptr = init_ret_ids.data_ptr<int64_t>();
        auto init_ret_dis_ptr = init_ret_dis.data_ptr<float>();

        // Rerank the results using the refine_index_
#pragma omp parallel if (n > 1)
        {
            std::unique_ptr<faiss::DistanceComputer> dc(
                refine_index_->refine_index->get_distance_computer());
#pragma omp for
            for (int i = 0; i < n; i++) {
                dc->set_query(x_ptr + i * d_);
                int ij = i * initial_k;
                for (int j = 0; j < initial_k; j++) {
                    int64_t idx = init_ret_ids_ptr[ij];
                    if (idx < 0)
                        break;
                    init_ret_dis_ptr[ij] = (*dc)(idx);
                    ij++;
                }
            }
        }

        int64_t *labels = ret_ids.data_ptr<int64_t>();
        float *distances = ret_dis.data_ptr<float>();

        int64_t *base_labels = init_ret_ids.data_ptr<int64_t>();
        float *base_distances = init_ret_dis.data_ptr<float>();

        // Sort and store result
        if (index_->metric_type == faiss::METRIC_L2) {
            typedef faiss::CMax<float, int64_t> Comparator;
            reorder_two_heaps<Comparator>(n, k, labels, distances, initial_k, base_labels, base_distances);
        } else if (index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
            typedef faiss::CMin<float, int64_t> Comparator;
            reorder_two_heaps<Comparator>(n, k, labels, distances, initial_k, base_labels, base_distances);
        } else {
            throw std::runtime_error("Unsupported metric type");
        }

        curr_end_time = std::chrono::high_resolution_clock::now();
        duration =
                std::chrono::duration_cast<std::chrono::microseconds>(curr_end_time - curr_start_time)
                .count();
        timing_info->refine_time_us = duration;
    } else {
        auto search_preprocess_start = std::chrono::high_resolution_clock::now();
        float* ret_dis_ptr = ret_dis.data_ptr<float>();
        int64_t* ret_ids_ptr = ret_ids.data_ptr<int64_t>();
        int64_t* cluster_ids_ptr = cluster_ids.data_ptr<int64_t>();
        float* x_ptr = x.data_ptr<float>();

        auto ret_ids_accessor = ret_ids.accessor<int64_t, 2>();
        auto ret_dis_accessor = ret_dis.accessor<float, 2>();

        auto invlists = casted_index->invlists;
        bool processed_query = false;

        // Set the scan parameters
        max_vectors_per_cluster_ = k;
        num_partitions_to_scan_ = cluster_ids.size(1);

        // Create the buffer for the job details
        int* prev_job_query_id = this->job_query_id_;
        int64_t* prev_cluster_id = this->job_search_cluster_id_;
        this->job_query_id_ = new int[num_search_vectors_ * num_partitions_to_scan_];
        this->job_search_cluster_id_ = new int64_t[num_search_vectors_ * num_partitions_to_scan_];

        // Free up the previous buffers
        if(prev_job_query_id != nullptr) {
            delete[] prev_job_query_id;
        }

        if(prev_cluster_id != nullptr) {
            delete[] prev_cluster_id;
        }

        // Create the intermediate buffer per query
        bool is_descending = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
        int query_buffer_capacity = max_vectors_per_cluster_ * num_partitions_to_scan_;
        while(final_result_mergers_.size() < num_search_vectors_) {
            final_result_mergers_.emplace_back(max_vectors_per_cluster_, is_descending);
        }

        for(int i = 0; i < num_search_vectors_; i++) {
            final_result_mergers_[i].set_k(max_vectors_per_cluster_);
            final_result_mergers_[i].reset();
        }

        // Initialize the buffer
        std::memset(all_vectors_scanned_ptr_, 0, num_scan_workers_ * sizeof(int));
        if(log_mode_) {
            std::memset(all_counts_ptr_, 0, num_scan_workers_ * sizeof(int));
            std::memset(all_job_times_ptr_, 0, num_scan_workers_ * sizeof(int));
            std::memset(all_scan_times_ptr_, 0, num_scan_workers_ * sizeof(int));
            std::memset(all_throughputs_ptr_, 0, num_scan_workers_ * sizeof(float));
        }

        // See if we need to create tempory numa buffer
        int temp_buffer_size;
        std::vector<float*> numa_reuse_buffers;
        if(using_numa_optimizations_ && num_search_vectors_ > QUERY_BUFFER_REUSE_THRESHOLD) {
    #ifdef QUAKE_NUMA
            int num_numa_nodes = this->get_num_numa_nodes();
            numa_reuse_buffers.reserve(num_numa_nodes);
            temp_buffer_size = num_search_vectors_ * vector_dimension;
            for(int i = 0; i < num_numa_nodes; i++) {
                // Save the pointer to the old buffer
                numa_reuse_buffers[i] = curr_queries_per_node_[i];

                // Allocate the new buffer
                curr_queries_per_node_[i] = reinterpret_cast<float*>(numa_alloc_onnode(temp_buffer_size * sizeof(float), i));
                if(curr_queries_per_node_[i] == NULL) {
                    throw std::runtime_error("Unable to allocate vector on numa node");
                }
            }
    #endif
        }

        auto search_preprocess_end = std::chrono::high_resolution_clock::now();
        timing_info->partition_scan_setup_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_preprocess_end - search_preprocess_start).count();
        timing_info->using_numa = false;
        target_latency_us -= timing_info->partition_scan_setup_time_us;

        auto search_scan_start = std::chrono::high_resolution_clock::now();
#ifdef QUAKE_NUMA
        if(using_numa_optimizations_) {
            timing_info->using_numa = true;
            
            // See if we need to preprocessing for adaptive nprobe
            auto numa_adaptive_start = std::chrono::high_resolution_clock::now();
            Tensor partition_probs; Tensor recall_profile;
            const bool is_euclidean = index_->metric_type == faiss::METRIC_L2;
            const bool use_acos = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
            std::vector<Tensor> all_boundary_distances;
            if(use_adaptive_for_search) {
                timing_info->using_adpative_nprobe = true;
                Tensor cluster_centroids = parent_->select_vectors(cluster_ids);
                all_boundary_distances.reserve(num_search_vectors_);
                for(int i = 0; i < num_search_vectors_; i++) {
                    all_boundary_distances.push_back(compute_boundary_distances(x[i].flatten(), cluster_centroids, is_euclidean));
                }
            }
            auto numa_adaptive_end = std::chrono::high_resolution_clock::now();
            timing_info->total_numa_adaptive_preprocess_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_adaptive_end - numa_adaptive_start).count();

            auto numa_preprocess_start_time = std::chrono::high_resolution_clock::now();
            int num_numa_nodes = this->get_num_numa_nodes();
            int prev_vector_buffer_size = prev_num_search_vectors * vector_dimension;
            int vector_buffer_size = num_search_vectors_ * vector_dimension;
            for(int i = 0; i < num_numa_nodes; i++) {
                // Copy the vector into the buffer
                std::memcpy(curr_queries_per_node_[i], x_ptr, vector_buffer_size * sizeof(float));
            }

            // Get the current index
            auto casted_index = ivf_index_;
            faiss::DynamicInvertedLists* dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(casted_index->invlists);

            // Reset the counts
            auto numa_preprocess_end_time = std::chrono::high_resolution_clock::now();
            timing_info->total_numa_preprocessing_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_preprocess_end_time - numa_preprocess_start_time).count();
            if(have_target_latency) {
                target_latency_us -= timing_info->total_numa_preprocessing_time_us;
            }

            int curr_job_id = 0;
            auto numa_job_distribute_start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_search_vectors_; i++) {
                // Send out the requests for the clusters to process
                int64_t* list_no_ptr = cluster_ids_ptr + i * num_partitions_to_scan_;
                int skipped_jobs = 0;
                TopKVectors& vector_merge_buffer = final_result_mergers_[i];
                vector_merge_buffer.set_processing_query(true);
                vector_merge_buffer.set_jobs_left(num_partitions_to_scan_);

                for (int j = 0; j < num_partitions_to_scan_; j++) {
                    curr_job_id = i * num_partitions_to_scan_ + j;
                    int64_t list_no = list_no_ptr[j];
                    if (list_no == -1) {
                        skipped_jobs += 1;
                        continue;
                    }

                    // Write the job details
                    this->job_search_cluster_id_[curr_job_id] = list_no;
                    this->job_query_id_[curr_job_id] = i;
                    timing_info->target_vectors_scanned += dynamic_invlists->list_size(list_no);

                    // Direct the request to approriate numa node
                    int job_queue_id = dynamic_invlists->get_numa_node(list_no);
                    jobs_queue_[job_queue_id].enqueue(curr_job_id);
                }

                vector_merge_buffer.record_skipped_jobs(skipped_jobs);
            }
            auto numa_job_distribution_end_time = std::chrono::high_resolution_clock::now();
            timing_info->total_job_distribute_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_job_distribution_end_time - numa_job_distribute_start_time).count();
            if(have_target_latency) {
                target_latency_us -= timing_info->total_job_distribute_time_us;
            }

            // Wait for all of the jobs to be completed
            int recall_estimate_time = 0;
            int total_flush_time = 0;
            auto numa_wait_start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_search_vectors_; i++) {
                TopKVectors& vector_merge_buffer = final_result_mergers_[i];
                Tensor curr_boundary_distances;
                if(use_adaptive_for_search) {
                    curr_boundary_distances = all_boundary_distances[i];
                }

                // Wait for the query to finish calling flush every so often
                auto query_start_time = std::chrono::high_resolution_clock::now();
                auto last_flush_time = std::chrono::high_resolution_clock::now();
                int scan_time_passed_us = 0;
                // Wait either until all of the jobs have finished or we hit the target latency
                while(!vector_merge_buffer.finished_all_jobs() && (!have_target_latency || scan_time_passed_us < target_latency_us)) {
                    // Determine if we need to flush
                    auto flush_curr_time = std::chrono::high_resolution_clock::now();
                    int time_since_last_flush = std::chrono::duration_cast<std::chrono::microseconds>(flush_curr_time - last_flush_time).count();
                    if(partition_search_flush_gap_us_ > 0 && time_since_last_flush > partition_search_flush_gap_us_) {
                        // Get the query radius
                        auto flush_start_time = std::chrono::high_resolution_clock::now();
                        float query_radius = vector_merge_buffer.flush();
                        auto flush_end_time = std::chrono::high_resolution_clock::now();
                        total_flush_time += std::chrono::duration_cast<std::chrono::microseconds>(flush_end_time - flush_start_time).count();

                        if(use_adaptive_for_search) {
                            // Compute the recall profile from the query profile
                            auto adaptive_start_time = std::chrono::high_resolution_clock::now();
                            query_radius = use_acos ? acos(query_radius) : sqrt(query_radius);
                            partition_probs = compute_recall_profile(curr_boundary_distances, query_radius, d_, {}, use_precomputed, is_euclidean);
                            recall_profile = torch::cumsum(partition_probs, 0);
                            float curr_recall_estimate = recall_profile[i].item<float>();

                            // Record the calculate time
                            auto adaptive_end_time = std::chrono::high_resolution_clock::now();
                            recall_estimate_time += std::chrono::duration_cast<std::chrono::microseconds>(adaptive_end_time - adaptive_start_time).count();

                            // See if we need to early exit
                            if (curr_recall_estimate >= recall_target) {
                                break;
                            }
                        }

                        last_flush_time = std::chrono::high_resolution_clock::now();
                    }

                    // Update scan time passed
                    auto scan_curr_time = std::chrono::high_resolution_clock::now();
                    scan_time_passed_us = std::chrono::duration_cast<std::chrono::microseconds>(scan_curr_time - query_start_time).count();
                }

                // Record that we are done processing this query
                vector_merge_buffer.set_processing_query(false);
            }
            auto numa_wait_end_time = std::chrono::high_resolution_clock::now();

            // Write the timings for the above steps
            timing_info->total_result_wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_wait_end_time - numa_wait_start_time).count();
            timing_info->total_adaptive_calculate_time_us += recall_estimate_time;
            timing_info->total_shared_flush_time += total_flush_time;
            processed_query = true;
        }
#endif

        if(!processed_query) {
            faiss::MetricType metric_type = index_->metric_type;

            for (int i = 0; i < num_search_vectors_; i++) {
                float* curr_query = x_ptr + i * d_;
                float* curr_dist = ret_dis_ptr + i * k;
                int64_t* curr_ids = ret_ids_ptr + i * k;
                int64_t* list_no_ptr = cluster_ids_ptr + i * cluster_ids.size(1);
                int nprobe = std::min((int) cluster_ids.size(1), (int) num_partitions_to_scan_);
                TopKVectors& curr_query_buffer = final_result_mergers_[i];

                int total_vectors = 0;
#pragma omp parallel for schedule(dynamic) num_threads(num_scan_workers_) reduction(+:total_vectors)
                for (int j = 0; j < nprobe; j++) {
                    int64_t list_no = list_no_ptr[j];
                    if (list_no == -1) {
                        continue;
                    }

                    int thread_id = omp_get_thread_num();
                    auto worker_query_start_time = std::chrono::high_resolution_clock::now();
                    TopKVectors& curr_cluster_buffer = intermediate_results_buffer_[thread_id];
                    curr_cluster_buffer.set_k(max_vectors_per_cluster_);

                    // Load the scan values
                    const float* list_vecs = (float*) invlists->get_codes(list_no);
                    const int64_t* list_ids = invlists->get_ids(list_no);
                    int list_size = invlists->list_size(list_no);
                    if(list_size == 0) {
                        curr_query_buffer.record_empty_job();
                        continue;
                    }

                    total_vectors += list_size;

                    // Perform the actual scan
                    auto worker_scan_start_time = std::chrono::high_resolution_clock::now();
                    scan_list(curr_query, list_vecs, list_ids, list_size, vector_dimension, curr_cluster_buffer, metric_type);
                    auto worker_scan_end_time = std::chrono::high_resolution_clock::now();

                    // Write the results out for this cluster
                    vector<float> overall_top_dists = curr_cluster_buffer.get_topk();
                    vector<int64_t> overall_top_ids = curr_cluster_buffer.get_topk_indices();
                    int num_results = std::min(max_vectors_per_cluster_, (int) overall_top_ids.size());
                    curr_query_buffer.batch_add(overall_top_dists.data(), overall_top_ids.data(), num_results);
                    all_vectors_scanned_ptr_[thread_id] += list_size;

                    // Log the times
                    if(log_mode_) {
                        auto worker_query_end_time = std::chrono::high_resolution_clock::now();

                        // Record the worker scan time
                        int total_scan_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_scan_end_time - worker_scan_start_time).count();
                        all_scan_times_ptr_[thread_id] += total_scan_time;

                        // Record the worker job time
                        int total_job_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_query_end_time - worker_query_start_time).count();
                        all_job_times_ptr_[thread_id] += total_job_time;
                        all_counts_ptr_[thread_id] += 1;

                        // Record worker scan throughput
                        size_t memory_scanned = ((size_t) list_size) * (vector_dimension * sizeof(float) + sizeof(idx_t));
                        float scan_throughput = ((1.0 * memory_scanned)/pow(10.0, 9))/((1.0 * total_scan_time)/pow(10.0, 6));
                        all_throughputs_ptr_[thread_id] += scan_throughput;
                    }
                }
                timing_info->target_vectors_scanned += total_vectors;
            }
        }

        // Record that we finished scanning
        auto search_scan_end = std::chrono::high_resolution_clock::now();
        timing_info->partition_scan_search_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_scan_end - search_scan_start).count();

        // Now perform the merge for each query
        auto search_post_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_search_vectors_; i++) {
            // Now write the top k for this query
            TopKVectors& vector_merge_buffer = final_result_mergers_[i];
            int write_offset = i * max_vectors_per_cluster_;
            float* final_dist_writer = ret_dis_ptr + write_offset;
            int64_t* final_ids_writer = ret_ids_ptr + write_offset;
            std::vector<float> topk_dists = vector_merge_buffer.get_topk();

            std::vector<int64_t> topk_ids = vector_merge_buffer.get_topk_indices();
            int num_results = std::min((int) topk_ids.size(), max_vectors_per_cluster_);
            std::memcpy(final_dist_writer, topk_dists.data(), num_results * sizeof(float));
            std::memcpy(final_ids_writer, topk_ids.data(), num_results * sizeof(int64_t));
        }

        // Wait for the queues to be drained
        if(using_numa_optimizations_) {
            auto numa_post_process_start_time = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < jobs_queue_.size(); i++) {
                moodycamel::BlockingConcurrentQueue<int>& curr_queue = jobs_queue_[i];
                while(curr_queue.size_approx() > 0) {
                    std::this_thread::yield();
                }
            }
            auto numa_post_process_end_time = std::chrono::high_resolution_clock::now();
            timing_info->total_numa_postprocessing_time_us += std::chrono::duration_cast<std::chrono::microseconds>(numa_post_process_end_time - numa_post_process_start_time).count();
        }

        // See if we need to free up the temporary buffer
        if(using_numa_optimizations_ && num_search_vectors_ > QUERY_BUFFER_REUSE_THRESHOLD) {
#ifdef QUAKE_NUMA
            int num_numa_nodes = this->get_num_numa_nodes();
            for(int i = 0; i < num_numa_nodes; i++) {
                // Free the temporary buffer
                numa_free(curr_queries_per_node_[i], temp_buffer_size * sizeof(float));
                curr_queries_per_node_[i] = numa_reuse_buffers[i];
            }
#endif
        }

        auto search_post_end = std::chrono::high_resolution_clock::now();
        timing_info->partition_scan_post_process_time_us += std::chrono::duration_cast<std::chrono::microseconds>(search_post_end - search_post_start).count();

        // Also record the total worker times if we are in log mode
        if(log_mode_) {
            int num_records = 0;
            for(int worker_id = 0; worker_id < num_scan_workers_; worker_id++) {
                // Record the timings
                timing_info->average_worker_job_time_us += this->all_job_times_ptr_[worker_id];
                timing_info->average_worker_scan_time_us += this->all_scan_times_ptr_[worker_id];
                timing_info->average_worker_throughput += this->all_throughputs_ptr_[worker_id];
                num_records += this->all_counts_ptr_[worker_id];
            }

            if(num_records > 0) {
                timing_info->average_worker_job_time_us /= num_records;
                timing_info->average_worker_scan_time_us /= num_records;
                timing_info->average_worker_throughput /= num_records;
            }
        }

        for(int worker_id = 0; worker_id < num_scan_workers_; worker_id++) {
            timing_info->total_vectors_scanned += this->all_vectors_scanned_ptr_[worker_id];
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    timing_info->partition_scan_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    return std::make_tuple(ret_ids, ret_dis, timing_info);
}

void DynamicIVF_C::refine_clusters(Tensor cluster_ids, int max_iterations) {
    // Cast the index to IndexIVF
    auto ivf_index = ivf_index_;
    if (!ivf_index) {
        std::cerr << "Index is not of type faiss::IndexIVF." << std::endl;
        return;
    }

    if (cluster_ids.size(0) == 0) {
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get the metric type from the index
    faiss::MetricType metric = ivf_index->metric_type;

    // Select initial clusters
    Tensor centroids;
    std::vector<Tensor> cluster_vectors;
    std::vector<Tensor> cluster_vec_ids;
    std::tie(centroids, cluster_vectors, cluster_vec_ids) = select_clusters(cluster_ids, true);

    int num_clusters = centroids.size(0);
    const float convergence_threshold = 1e-4;

    // Initialize assignments to what they are currently
    std::vector<Tensor> assignments(cluster_vectors.size(),
                                    -torch::ones({cluster_vectors[0].size(0)}, torch::kInt64));


    for (int iter = 0; iter < (max_iterations + 1); iter++) {
        // Prepare new centroids and counts
        torch::Tensor new_centroids = torch::zeros_like(centroids);
        std::vector<int64_t> counts(num_clusters, 0);

        // Iterate over each cluster's vectors in parallel
        #pragma omp parallel for
        for (size_t i = 0; i < cluster_vectors.size(); ++i) {
            if (cluster_vectors[i].size(0) == 0) continue;

            torch::Tensor distances_or_similarities;
            torch::Tensor assignment_indices;

            if (metric == faiss::METRIC_INNER_PRODUCT) {
                // Compute similarities using inner products
                distances_or_similarities = torch::matmul(cluster_vectors[i], centroids.transpose(0, 1));
                // Find the maximum similarities
                torch::Tensor max_vals, max_indices;
                std::tie(max_vals, max_indices) = distances_or_similarities.max(1);
                assignment_indices = max_indices;
            } else {
                // Compute Euclidean distances
                torch::Tensor distances = torch::cdist(cluster_vectors[i], centroids, /*p=*/2);
                // Find the minimum distances
                torch::Tensor min_vals, min_indices;
                std::tie(min_vals, min_indices) = distances.min(1);
                assignment_indices = min_indices;
            }

            assignments[i] = assignment_indices;

            // Accumulate new centroids using thread-local storage
            if (iter < max_iterations) {
                torch::Tensor assignment = assignments[i];
                torch::Tensor vecs = cluster_vectors[i];

                // Use thread-local storage for centroids and counts
                torch::Tensor new_centroids_local = torch::zeros_like(new_centroids);
                std::vector<int64_t> counts_local(num_clusters, 0);

                auto assignment_accessor = assignment.accessor<int64_t, 1>();
                auto vecs_accessor = vecs.accessor<float, 2>();

                for (int64_t j = 0; j < vecs.size(0); ++j) {
                    int64_t cluster_idx = assignment_accessor[j];
                    new_centroids_local[cluster_idx] += vecs[j];
                    counts_local[cluster_idx]++;
                }

                // Reduce the local accumulations into the global ones
                #pragma omp critical
                {
                    new_centroids += new_centroids_local;
                    for (int64_t k = 0; k < num_clusters; ++k) {
                        counts[k] += counts_local[k];
                    }
                }
            }
        }

        // Update centroids
        if (iter < max_iterations) {
            for (int64_t i = 0; i < num_clusters; ++i) {
                if (counts[i] > 0) {
                    centroids[i] = new_centroids[i] / counts[i];
                    if (metric == faiss::METRIC_INNER_PRODUCT) {
                        // Normalize centroid to unit length for spherical k-means
                        float norm = centroids[i].norm().item<float>();
                        if (norm > 0) {
                            centroids[i] = centroids[i] / norm;
                        }
                    }
                }
            }
        }
        if (iter == max_iterations) {
            std::cout << "Reached maximum iterations." << std::endl;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Refined centroids in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();

    if (max_iterations > 0) {
        parent_->modify(centroids, cluster_ids, false);
    }

    // Reassign vectors to new clusters
    auto dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index->invlists);
    if (!dynamic_invlists) {
        std::cerr << "Failed to cast invlists to faiss::DynamicInvertedLists." << std::endl;
        return;
    }

    int64_t* cluster_ids_accessor = cluster_ids.data_ptr<int64_t>();
    int64_t num_reassignments = 0;
    int64_t n_total = 0;
    int num_cluster_vectors = cluster_vectors.size();
    float total_delete_time = 0.0; float total_update_time = 0.0; float total_set_build_time = 0.0; int count = 0;

    for (int i = 0; i < cluster_vectors.size(); i++) {
        Tensor assignment = assignments[i];
        Tensor vecs = cluster_vectors[i];
        Tensor ids = cluster_vec_ids[i];
        if (vecs.size(0) == 0) continue;

        // Get the new cluster IDs based on assignments
        Tensor new_cluster_ids = cluster_ids.index_select(0, assignment);
        int64_t* new_cluster_ids_accessor = new_cluster_ids.data_ptr<int64_t>();
        int64_t* ids_accessor = ids.data_ptr<int64_t>();
        uint8_t* codes = reinterpret_cast<uint8_t*>(vecs.data_ptr<float>());
        int64_t old_cluster = cluster_ids_accessor[i];

        // First do pass batching up the requests
        auto set_start_time = std::chrono::high_resolution_clock::now();
        std::set<idx_t> vectors_to_remove;
        size_t vector_code_size = d_ * sizeof(float);
        int total_vectors = vecs.size(0);
        for (int64_t j = 0; j < total_vectors; j++) {
            if (new_cluster_ids_accessor[j] != old_cluster) {
                idx_t vec_id = static_cast<idx_t>(ids_accessor[j]);
                vectors_to_remove.insert(vec_id);
                num_reassignments++;
            }
            n_total++;
        }
        auto set_end_time = std::chrono::high_resolution_clock::now();
        total_set_build_time += std::chrono::duration_cast<std::chrono::microseconds>(set_end_time - set_start_time).count()/1000.0;

        // Now make the requests to first remove these vectors and then reinsert them
        size_t casted_old_cluster = static_cast<size_t>(old_cluster);
        auto remove_start_time = std::chrono::high_resolution_clock::now();
        dynamic_invlists->remove_entries_from_partition(casted_old_cluster, vectors_to_remove);
        auto remove_end_time = std::chrono::high_resolution_clock::now();
        total_delete_time += std::chrono::duration_cast<std::chrono::microseconds>(remove_end_time - remove_start_time).count()/1000.0;

        auto update_start_time = std::chrono::high_resolution_clock::now();
        dynamic_invlists->batch_update_entries(casted_old_cluster, new_cluster_ids_accessor, codes, ids_accessor, total_vectors);
        auto update_end_time = std::chrono::high_resolution_clock::now();
        total_update_time += std::chrono::duration_cast<std::chrono::microseconds>(update_end_time - update_start_time).count()/1000.0;
        count += 1;
    }
    end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Reassingment breakdown: Set Build Time - " << total_set_build_time << ", Remove - " << total_delete_time << ", Update - " << total_update_time << std::endl;
    std::cout << "Reassignment took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms." << std::endl;
    std::cout << "Reassigned " << num_reassignments << " vectors out of " << n_total << " total vectors." << std::endl;
}

void DynamicIVF_C::selective_merge(Tensor selected_clusters, DynamicIVF_C &other, Tensor other_selected_clusters) {
    // Merge selected clusters from another index into the current index

    auto casted_index = ivf_index_;
    auto casted_other_index = dynamic_cast<faiss::IndexIVF *>(other.index_);

    if (casted_index == nullptr || casted_other_index == nullptr) {
        return;
    }

    faiss::IndexIVF *new_index = merge_faiss_ivf(casted_index, selected_clusters, casted_other_index,
                                                 other_selected_clusters);
    delete index_;
}

Tensor DynamicIVF_C::add_centroids_and_reassign_existing(Tensor new_centroids, Tensor reassign_cluster_ids,
                                                         Tensor kept_cluster_ids) {
    // Implement logic to add new centroids and reassign existing vectors
    // This involves updating the quantizer, reassigning vectors, and updating the inverted lists

    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kInt64);
    }

    // Get existing centroids
    Tensor existing_centroids = centroids();

    // Concatenate new centroids
    Tensor all_centroids = torch::cat({existing_centroids.index_select(0, kept_cluster_ids), new_centroids}, 0);

    // Build new quantizer
    auto quantizer = new faiss::IndexFlat(d_, index_->metric_type);
    quantizer->add(all_centroids.size(0), all_centroids.data_ptr<float>());

    // Create new index
    faiss::IndexIVF *new_index = nullptr;
    if (use_refine_) {
        new_index = new faiss::IndexIVFPQ(quantizer, d_, all_centroids.size(0), num_codebooks_, code_size_);
    } else {
        new_index = new faiss::IndexIVFFlat(quantizer, d_, all_centroids.size(0), index_->metric_type);
    }

    // Reassign vectors from reassign_cluster_ids
    for (int i = 0; i < reassign_cluster_ids.size(0); i++) {
        int64_t cluster_id = reassign_cluster_ids[i].item<int64_t>();
        int64_t list_size = casted_index->invlists->list_size(cluster_id);
        if (list_size == 0) {
            continue;
        }
        auto codes = casted_index->invlists->get_codes(cluster_id);
        auto ids = casted_index->invlists->get_ids(cluster_id);

        Tensor vectors = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);
        Tensor vector_ids = torch::from_blob((void *) ids, {list_size}, torch::kInt64);

        // Assign vectors to new clusters
        Tensor distances;
        Tensor new_cluster_ids;
        quantizer->assign(list_size, vectors.data_ptr<float>(), new_cluster_ids.data_ptr<int64_t>());

        // Add vectors to new index
        new_index->add_with_ids(list_size, vectors.data_ptr<float>(), vector_ids.data_ptr<int64_t>());
    }

    // Copy kept clusters
    for (int i = 0; i < kept_cluster_ids.size(0); i++) {
        int64_t cluster_id = kept_cluster_ids[i].item<int64_t>();
        int64_t list_size = casted_index->invlists->list_size(cluster_id);
        if (list_size == 0) {
            continue;
        }
        auto codes = casted_index->invlists->get_codes(cluster_id);
        auto ids = casted_index->invlists->get_ids(cluster_id);

        Tensor vectors = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);
        Tensor vector_ids = torch::from_blob((void *) ids, {list_size}, torch::kInt64);

        // Add vectors to new index
        new_index->add_with_ids(list_size, vectors.data_ptr<float>(), vector_ids.data_ptr<int64_t>());
    }

    delete index_;

    // Update centroids
    centroids_ = all_centroids;

    return torch::arange(all_centroids.size(0), torch::kInt64);
}

Tensor DynamicIVF_C::compute_quantization_error() {
    Tensor partition_ids = get_partition_ids();
    auto clusters = select_clusters(partition_ids, false);

    Tensor centroids = std::get<0>(clusters);
    vector<Tensor> cluster_vectors = std::get<1>(clusters);

    Tensor errors = torch::zeros({centroids.size(0)}, torch::kFloat32);
    for (int i = 0; i < centroids.size(0); i++) {
        // Calculate the differences from the centroid
        Tensor diffs = cluster_vectors[i] - centroids[i];

        // Calculate the mean distance
        errors[i] = diffs.pow(2).sum(1).sqrt().mean();
    }

    return errors;
}

Tensor DynamicIVF_C::compute_cluster_sums(bool squared) const {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kFloat32);
    }
    auto invlists = casted_index->invlists;
    Tensor cluster_sums = torch::zeros({nlist(), d_}, torch::kFloat32);

    for (int i = 0; i < nlist(); i++) {
        int64_t list_size = invlists->list_size(i);
        if (list_size == 0) {
            continue;
        }

        auto codes = invlists->get_codes(i);
        Tensor vectors = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);

        if (squared) {
            cluster_sums[i] = vectors.pow(2).sum(0);
        } else {
            cluster_sums[i] = vectors.sum(0);
        }
    }

    return cluster_sums;
}

Tensor DynamicIVF_C::compute_cluster_covariance(int cluster_id) const {
    auto casted_index = ivf_index_;
    if (casted_index == nullptr) {
        return torch::empty({0}, torch::kFloat32);
    }
    auto invlists = casted_index->invlists;
    int64_t list_size = invlists->list_size(cluster_id);
    if (list_size == 0) {
        return torch::zeros({d_, d_}, torch::kFloat32);
    }
    auto codes = invlists->get_codes(cluster_id);
    Tensor vectors = torch::from_blob((void *) codes, {list_size, d_}, torch::kFloat32);

    Tensor mean = vectors.mean(0);
    Tensor diffs = vectors - mean;
    Tensor covariance = diffs.t().mm(diffs) / (list_size - 1);

    return covariance;
}

vector<Tensor> DynamicIVF_C::compute_partition_variances(Tensor partition_ids) {
    auto clusters = select_clusters(partition_ids, false);

    Tensor centroids = std::get<0>(clusters);
    vector<Tensor> cluster_vectors = std::get<1>(clusters);

    vector<Tensor> variances;
    for (int i = 0; i < centroids.size(0); i++) {
        // Calculate the differences from the centroid
        Tensor diffs = cluster_vectors[i] - centroids[i];

        // Calculate the variance
        Tensor variance_per_dim = diffs.var(0, /*unbiased=*/false);

        variances.push_back(variance_per_dim);
    }
    return variances;
}

vector<Tensor> DynamicIVF_C::compute_partition_covariances() {
    vector<Tensor> covariances;
    Tensor partition_ids = get_partition_ids();
    Tensor centroids;
    vector<Tensor> vectors;
    vector<Tensor> ids;

    std::tie(centroids, vectors, ids) = select_clusters(partition_ids, false);

    for (int i = 0; i < partition_ids.size(0); i++) {
        // use torch::cov
        Tensor curr_vectors = vectors[i];
        Tensor cov = torch::cov(curr_vectors.t(), 0);
        covariances.push_back(cov);
    }

    return covariances;
}

faiss::InvertedLists* merge_invlists(faiss::DynamicInvertedLists* array_lists1, Tensor selected_clusters1,
                                     faiss::DynamicInvertedLists* array_lists2, Tensor selected_clusters2) {
    int64_t nlist = selected_clusters1.size(0) + selected_clusters2.size(0);
    auto code_size = array_lists1->code_size;

    auto ret_invlists = new faiss::DynamicInvertedLists(nlist, code_size);

    return ret_invlists;
}

faiss::IndexIVF* merge_faiss_ivf(faiss::IndexIVF* index1, Tensor selected_clusters1, faiss::IndexIVF* index2, Tensor selected_clusters2) {
    // Extract centroids
    Tensor centroids1 = torch::empty({(int64_t)index1->nlist, index1->d}, torch::kFloat32);
    index1->quantizer->reconstruct_n(0, index1->nlist, centroids1.data_ptr<float>());
    centroids1 = centroids1.index_select(0, selected_clusters1.to(torch::kLong));

    Tensor centroids2 = torch::empty({(int64_t)index2->nlist, index2->d}, torch::kFloat32);
    index2->quantizer->reconstruct_n(0, index2->nlist, centroids2.data_ptr<float>());
    centroids2 = centroids2.index_select(0, selected_clusters2.to(torch::kLong));

    // Merge centroids
    Tensor all_centroids = torch::cat({centroids1, centroids2}, 0);

    // Build new quantizer
    auto quantizer = new faiss::IndexFlat(index1->d, index1->metric_type);
    quantizer->add(all_centroids.size(0), all_centroids.data_ptr<float>());

    // Merge inverted lists
    auto invlists1 = dynamic_cast<faiss::DynamicInvertedLists*>(index1->invlists);
    auto invlists2 = dynamic_cast<faiss::DynamicInvertedLists*>(index2->invlists);

    auto new_invlists = merge_invlists(invlists1, selected_clusters1, invlists2, selected_clusters2);

    // Create new index
    faiss::IndexIVF* new_index = nullptr;
    if (dynamic_cast<faiss::IndexIVFPQ*>(index1)) {
        auto index1_pq = dynamic_cast<faiss::IndexIVFPQ*>(index1);
        new_index = new faiss::IndexIVFPQ(quantizer, index1->d, quantizer->ntotal, index1_pq->pq.M, index1_pq->pq.nbits);
    } else {
        new_index = new faiss::IndexIVFFlat(quantizer, index1->d, quantizer->ntotal, index1->metric_type);
    }

    new_index->own_invlists = true;
    new_index->replace_invlists(new_invlists);

    return new_index;
}

#ifdef FAISS_ENABLE_GPU
BuildTimingInfo DynamicIVF_C::build_index_on_gpu(Tensor x, Tensor ids) {
    BuildTimingInfo timing_info;

    faiss::gpu::StandardGpuResources res;

    if (use_refine_) {
        auto train_start = std::chrono::high_resolution_clock::now();

        faiss::gpu::GpuIndexIVFPQConfig config;
        config.device = 0;  // Set appropriate device if needed
        faiss::gpu::GpuIndexIVFPQ gpu_index(&res, dynamic_cast<faiss::IndexIVFPQ*>(index_), config);

        gpu_index.train(x.size(0), x.data_ptr<float>());
        auto train_end = std::chrono::high_resolution_clock::now();

        auto add_start = std::chrono::high_resolution_clock::now();
        gpu_index.add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
        auto add_end = std::chrono::high_resolution_clock::now();

        faiss::IndexIVFPQ* cpu_index = dynamic_cast<faiss::IndexIVFPQ*>(faiss::gpu::index_gpu_to_cpu(&gpu_index));

        delete index_;
        index_ = cpu_index;

        delete refine_index_;
        refine_index_ = new faiss::IndexRefineFlat(index_);

        index_->make_direct_map();

        timing_info.train_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();
        timing_info.assign_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(add_end - add_start).count();

    } else {
        auto train_start = std::chrono::high_resolution_clock::now();

        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;  // Set appropriate device if needed
        faiss::gpu::GpuIndexIVFFlat gpu_index(&res, dynamic_cast<faiss::IndexIVFFlat*>(index_), config);

        gpu_index.train(x.size(0), x.data_ptr<float>());
        auto train_end = std::chrono::high_resolution_clock::now();

        auto add_start = std::chrono::high_resolution_clock::now();
        gpu_index.add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
        auto add_end = std::chrono::high_resolution_clock::now();

        faiss::IndexIVFFlat* cpu_index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::gpu::index_gpu_to_cpu(&gpu_index));

        delete index_;
        index_ = cpu_index;

        index_->make_direct_map();

        timing_info.train_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();
        timing_info.assign_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(add_end - add_start).count();
    }

    return timing_info;
}

void DynamicIVF_C::rebuild_index_on_gpu(Tensor x, Tensor ids, int new_nlist) {
    int dimension = index_->d;
    faiss::gpu::StandardGpuResources res;

    if (use_refine_) {
        faiss::gpu::GpuIndexIVFPQConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexIVFPQ gpu_index(&res, dimension, new_nlist, num_codebooks_, code_size_, index_->metric_type, config);

        gpu_index.train(x.size(0), x.data_ptr<float>());
        gpu_index.add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());

        faiss::IndexIVFPQ* new_index = dynamic_cast<faiss::IndexIVFPQ*>(faiss::gpu::index_gpu_to_cpu(&gpu_index));

        delete index_;
        index_ = new_index;

        delete refine_index_;
        refine_index_ = new faiss::IndexRefineFlat(index_);

        index_->make_direct_map();

    } else {
        faiss::gpu::GpuIndexIVFFlatConfig config;
        config.device = 0;
        faiss::gpu::GpuIndexIVFFlat gpu_index(&res, dimension, new_nlist, index_->metric_type, config);

        gpu_index.train(x.size(0), x.data_ptr<float>());
        gpu_index.add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());

        faiss::IndexIVFFlat* new_index = dynamic_cast<faiss::IndexIVFFlat*>(faiss::gpu::index_gpu_to_cpu(&gpu_index));

        delete index_;
        index_ = new_index;

        index_->make_direct_map();
    }
}
#endif  // FAISS_ENABLE_GPU

BuildTimingInfo DynamicIVF_C::build_index_on_cpu(Tensor x, Tensor ids) {
    BuildTimingInfo timing_info;

    auto train_start = std::chrono::high_resolution_clock::now();
    if (use_refine_) {
        refine_index_->train(x.size(0), x.data_ptr<float>());
    } else {
        index_->train(x.size(0), x.data_ptr<float>());
    }

    auto train_end = std::chrono::high_resolution_clock::now();

    auto add_start = std::chrono::high_resolution_clock::now();
    if (use_refine_) {
        refine_index_->add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
    } else {
        index_->add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());
    }
    auto add_end = std::chrono::high_resolution_clock::now();

    // index_->make_direct_map();

    timing_info.train_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();
    timing_info.assign_time_us =
            std::chrono::duration_cast<std::chrono::microseconds>(add_end - add_start).count();

    return timing_info;
}

void DynamicIVF_C::rebuild_index_on_cpu(Tensor x, Tensor ids, int new_nlist) {
    int dimension = index_->d;

    // Build quantizer
    auto quantizer = new faiss::IndexFlat(dimension, index_->metric_type);

    // Build index
    if (use_refine_) {
        faiss::IndexIVFPQ *new_index = new faiss::IndexIVFPQ(quantizer, dimension, new_nlist, num_codebooks_,
                                                             code_size_);

        new_index->train(x.size(0), x.data_ptr<float>());
        new_index->add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());

        delete index_;
        index_ = new faiss::IndexIDMap2(new_index);
        ivf_index_ = new_index;

        delete refine_index_;
        refine_index_ = new faiss::IndexRefineFlat(index_);
    } else {
        faiss::IndexIVFFlat *new_index = new faiss::IndexIVFFlat(quantizer, dimension, new_nlist, index_->metric_type);

        new_index->train(x.size(0), x.data_ptr<float>());
        new_index->add_with_ids(x.size(0), x.data_ptr<float>(), ids.data_ptr<int64_t>());

        delete index_;
        index_ = new faiss::IndexIDMap2(new_index);
        ivf_index_ = new_index;
    }
}

Tensor DynamicIVF_C::compute_partition_boundary_distances(Tensor query, Tensor partition_ids) {
    if (parent_ == nullptr) {
        throw std::runtime_error("Index is not partitioned");
    }
    // filter out the nearest centroid
    Tensor centroids = parent_->select_vectors(partition_ids);
    Tensor boundary_distances = compute_boundary_distances(query.flatten(), centroids);
    return boundary_distances;
}

Tensor DynamicIVF_C::get_partition_ids_for_vector_ids(Tensor vector_ids) {
    auto invlists = get_invlists();

    // for each vector id get the corresponding list id
    // we don't have a map currently to get the list id for a vector id so we have to scan all lists

    // iterate over the ids
    auto ids = invlists->ids_;

    Tensor partition_ids = torch::full({vector_ids.size(0)}, -1, torch::kInt64);
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    auto vector_ids_accessor = vector_ids.accessor<int64_t, 1>();

    for (int i = 0; i < vector_ids.size(0); i++) {
        int64_t vector_id = vector_ids_accessor[i];
        for (auto item: ids) {
            auto list_no = item.first;
            if (invlists->id_in_list(list_no, vector_id)) {
                partition_ids_accessor[i] = list_no;
                break;
            }
        }
    }

    return partition_ids;
}

float DynamicIVF_C::compute_kth_nearest_neighbor_distance(Tensor query, int k) {
    auto res = search(query, 100, k, 1.1);
    auto ret_dis = std::get<1>(res).flatten();

    float kth_dist = sqrt(ret_dis[k - 1].item<float>());

    return kth_dist;
}

Tensor DynamicIVF_C::compute_partition_probabilities(Tensor query, int k, Tensor partition_ids, bool use_size) {
    if (!partition_ids.defined()) {
        partition_ids = get_partition_ids();
    }

    if (parent_ == nullptr) {
        throw std::runtime_error("Index is not partitioned");
    }

    float query_radius = compute_kth_nearest_neighbor_distance(query, k);
    Tensor boundary_distances = compute_partition_boundary_distances(query, partition_ids);

    Tensor cluster_sizes;
    if (use_size) {
        cluster_sizes = get_partition_sizes(partition_ids);
    }

    Tensor partition_probs = compute_recall_profile(boundary_distances, query_radius, query.size(0), cluster_sizes);
    return partition_probs;
}

Tensor DynamicIVF_C::compute_partition_intersection_volumes(Tensor query, Tensor partition_ids, int k) {
    if (!partition_ids.defined()) {
        partition_ids = get_partition_ids();
    }

    if (parent_ == nullptr) {
        throw std::runtime_error("Index is not partitioned");
    }

    float query_radius = compute_kth_nearest_neighbor_distance(query, k);
    Tensor boundary_distances = compute_partition_boundary_distances(query, partition_ids);


    vector<float> intersection_volume = compute_intersection_volume(boundary_distances, query_radius, d_);

    Tensor partition_volumes = torch::from_blob(intersection_volume.data(), {partition_ids.size(0)}, torch::kFloat32).
            clone();

    return partition_volumes;
}

Tensor DynamicIVF_C::compute_partition_distances(Tensor query, Tensor partition_ids) {
    if (!partition_ids.defined()) {
        partition_ids = get_partition_ids();
    }

    if (parent_ == nullptr) {
        throw std::runtime_error("Index is not partitioned");
    }

    Tensor centroids = parent_->select_vectors(partition_ids);
    Tensor distances = torch::cdist(query.unsqueeze(0), centroids, /*p=*/2).squeeze(0);

    return distances;
}

Tensor DynamicIVF_C::compute_partition_volume(Tensor partition_ids) {
    if (!partition_ids.defined()) {
        partition_ids = get_partition_ids();
    }

    Tensor cluster_radii = torch::norm(compute_quantization_error().index_select(0, partition_ids), 2, 1);
    Tensor partition_volumes = torch::empty({partition_ids.size(0)}, torch::kFloat32);

    auto cluster_radii_accessor = cluster_radii.accessor<float, 1>();
    auto partition_volumes_accessor = partition_volumes.accessor<float, 1>();
    for (int i = 0; i < cluster_radii.size(0); i++) {
        partition_volumes_accessor[i] = log_hypersphere_volume(cluster_radii_accessor[i], d_);
    }

    return partition_volumes;
}

Tensor DynamicIVF_C::compute_partition_density(Tensor partition_ids) {
    if (!partition_ids.defined()) {
    partition_ids = get_partition_ids();
    }

    Tensor partition_volumes = compute_partition_volume(partition_ids);
    Tensor partition_sizes = torch::log(get_partition_sizes(partition_ids));

    return partition_sizes - partition_volumes;
}

Tensor DynamicIVF_C::get_partition_sizes(Tensor partition_ids) {
    auto dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index_->invlists);
    Tensor partition_sizes = torch::empty({partition_ids.size(0)}, torch::kInt64);

    auto partition_ids_accessor = partition_ids.contiguous().data_ptr<int64_t>();
    auto partition_sizes_accessor = partition_sizes.contiguous().data_ptr<int64_t>();
    for (int i = 0; i < partition_ids.size(0); i++) {
        partition_sizes_accessor[i] = dynamic_invlists->list_size(partition_ids_accessor[i]);
    }

    return partition_sizes;
}

void DynamicIVF_C::set_maintenance_policy_params(MaintenancePolicyParams params) {

    // create the maintenance policy based on the parameters
    if (params.maintenance_policy == "query_cost") {
        maintenance_policy_ = std::make_shared<QueryCostMaintenance>(shared_from_this(), params);
    } else if (params.maintenance_policy == "lire") {
        maintenance_policy_ = std::make_shared<LireMaintenance>(shared_from_this(), params.target_partition_size, params.max_partition_ratio, params.min_partition_size);
    } else if (params.maintenance_policy == "dedrift") {
        maintenance_policy_ = std::make_shared<DeDriftMaintenance>(shared_from_this(), params.k_large, params.k_small, params.modify_centroids);
    } else {
        throw std::runtime_error("Invalid maintenance policy");
    }
}


vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> DynamicIVF_C::get_split_history() {
    return maintenance_policy_->get_split_history();
}

bool DynamicIVF_C::index_ready() {
    bool is_ready = workers_ready_.load() == num_scan_workers_;
    if(is_ready && parent_ != nullptr) {
        is_ready = parent_->index_ready();
    }
    return is_ready;
}

void DynamicIVF_C::launch_cluster_scan_workers(bool only_current_level) {
    // Only the first launch call should be considered
    if(workers_initialized_) {
        return;
    }

    // Also ensure at least one worker is specified
    if(num_scan_workers_ == 0) {
        return;
    }

    if(!clusters_distributed_) {
        distribute_clusters(only_current_level);
    }

#ifdef QUAKE_NUMA
    // Ensure the cluster has been distributed

    if(using_numa_optimizations_) {
        // Create the job queues
        bool is_top_level = parent_ == nullptr;
        int num_numa_nodes = get_num_numa_nodes();
        int num_job_queues = is_top_level ? num_scan_workers_ : num_numa_nodes;
        for(int i = 0; i < num_job_queues; i++) {
            jobs_queue_.emplace_back();
        }

        // Launch the workers
        workers_ready_.store(0);
        for (int tid = 0; tid < num_scan_workers_; tid++) {
            if(is_top_level) {
                if(use_centroid_workers_) {
                    scan_workers_.emplace_back(&DynamicIVF_C::centroids_scan_worker_function, this, tid);
                } else {
                    workers_ready_.fetch_add(1);
                }
            } else {
                scan_workers_.emplace_back(&DynamicIVF_C::partition_scan_worker_function, this, tid);
            }
        }
    } else {
        // Create the intermediate buffers for the workers
        bool is_descending = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
        for(int i = 0; i < num_scan_workers_; i++) {
            intermediate_results_buffer_.emplace_back(max_vectors_per_cluster_, is_descending);
        }
        workers_ready_.store(num_scan_workers_);
    }

#else
    bool is_descending = index_->metric_type == faiss::METRIC_INNER_PRODUCT;
    for(int i = 0; i < num_scan_workers_; i++) {
        intermediate_results_buffer_.emplace_back(max_vectors_per_cluster_, is_descending);
    }
    workers_ready_.store(num_scan_workers_);
#endif

    if (!only_current_level && parent_ != nullptr) {
        parent_->launch_cluster_scan_workers();
    }

    workers_initialized_ = true;
}


void DynamicIVF_C::distribute_clusters(bool only_current_level) {
    bool is_top_level = parent_ == nullptr;

#ifdef QUAKE_NUMA
    int num_numa_nodes = this->get_num_numa_nodes();
    if(!is_top_level && using_numa_optimizations_) {
        // Get the unassigned inverted list from the index and sequentially assign each unassigned to a different numa node
        // TODO: This is a relatively naive policy but this will ensure all unassigned clusters are placed to a numa node
        auto dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index_->invlists);
        std::set<size_t> current_unassigned_clusters = dynamic_invlists->get_unassigned_clusters();
        int curr_numa_node = 0;
        for(auto current_cluster : current_unassigned_clusters) {
            dynamic_invlists->set_numa_node(current_cluster, curr_numa_node);
            curr_numa_node = (curr_numa_node + 1) % num_numa_nodes;
        }
    } else if(is_top_level) {
        centroid_store_->distribute_centroids(num_scan_workers_, workers_numa_nodes_.data(), using_numa_optimizations_);
    }
#else
    if (is_top_level) {
        centroid_store_->distribute_centroids(num_scan_workers_, workers_numa_nodes_.data(), using_numa_optimizations_);
    }
#endif
    clusters_distributed_ = true;

    if(!only_current_level && parent_ != nullptr) {
        parent_->distribute_clusters();
    }
}

#ifdef QUAKE_NUMA
void DynamicIVF_C::centroids_scan_worker_function(int thread_id) {
    int thread_numa_node = workers_numa_nodes_[thread_id];
    int num_numa_nodes = this->get_num_numa_nodes();
    int expected_numa_node_id = thread_numa_node;
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu = thread_id % num_cpus;

    // Pin thread to a CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }

    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }

    // Set the preferred NUMA node for memory allocations
    int numa_node_id = numa_node_of_cpu(sched_getcpu());
    if(numa_node_id != expected_numa_node_id) {
        throw std::runtime_error("Thread not pinned to the expected numa node");
    }
    numa_set_preferred(numa_node_id);
    numa_set_strict(1);

    // Get the intermediate results
    int vector_dimension = d_;
    moodycamel::BlockingConcurrentQueue<int>& requests_queue = jobs_queue_[thread_id];
    faiss::MetricType metric_type = index_->metric_type;
    bool is_descending = metric_type == faiss::METRIC_INNER_PRODUCT;
    TopKVectors curr_cluster_buffer(max_vectors_per_cluster_, is_descending);

    // Mark that this worker is ready
    workers_ready_.fetch_add(1);

    int total_scan_time = 0; int total_job_time = 0;
    int job_id;
    while (true) {
        // Wait for a new query
        requests_queue.wait_dequeue(job_id);

        // See if is the shutdown query
        auto worker_query_start_time = std::chrono::high_resolution_clock::now();
        if(job_id == -1) {
            break;
        }

        // Load the scan values
        int query_id = job_query_id_[job_id];
        float* query_vector = curr_queries_per_node_[numa_node_id] + query_id * vector_dimension;
        float* local_centroid_vectors = centroid_store_->get_vectors_for_worker(thread_id);
        idx_t* local_centroid_ids = centroid_store_->get_ids_for_worker(thread_id);
        int curr_num_centroids = centroid_store_->num_centroids_for_worker(thread_id);
        curr_cluster_buffer.set_k(max_vectors_per_cluster_);

        // Perform the numa verificiation on the centroid data
        if(verify_numa_) {
            int curr_worker_numa_node = numa_node_of_cpu(sched_getcpu());
            if(curr_worker_numa_node != expected_numa_node_id) {
                throw std::runtime_error("Centroid Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but cluster on node " + std::to_string(expected_numa_node_id));
            }

            int codes_numa_node = -1;
            get_mempolicy(&codes_numa_node, NULL, 0, (void*) local_centroid_vectors, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != codes_numa_node) {
                throw std::runtime_error("Centroid Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but codes on node " + std::to_string(codes_numa_node));
            }

            int ids_numa_node = -1;
            get_mempolicy(&ids_numa_node, NULL, 0, (void*) local_centroid_ids, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != ids_numa_node) {
                throw std::runtime_error("Centroid Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but ids on node " + std::to_string(ids_numa_node));
            }

            int query_numa_node = -1;
            get_mempolicy(&query_numa_node, NULL, 0, (void*) query_vector, MPOL_F_NODE | MPOL_F_ADDR);
            if(expected_numa_node_id != query_numa_node) {
                throw std::runtime_error("Centroid Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(expected_numa_node_id) + " but query on " + std::to_string(query_numa_node));
            }
        }

        // Perform the actual scan
        auto worker_scan_start_time = std::chrono::high_resolution_clock::now();
        scan_list(query_vector, local_centroid_vectors, local_centroid_ids, curr_num_centroids, vector_dimension, curr_cluster_buffer, metric_type);
        auto worker_scan_end_time = std::chrono::high_resolution_clock::now();

        // Write the number of results
        std::vector<int64_t> overall_top_ids = curr_cluster_buffer.get_topk_indices();
        std::vector<float> overall_top_dists = curr_cluster_buffer.get_topk();
        int num_results = std::min((int) overall_top_ids.size(), max_vectors_per_cluster_);

        // Insert the records into the buffer
        final_result_mergers_[query_id].batch_add(overall_top_dists.data(), overall_top_ids.data(), num_results);
        all_vectors_scanned_ptr_[thread_id] += curr_num_centroids;

        if(log_mode_) {
            auto worker_query_end_time = std::chrono::high_resolution_clock::now();

            // Record the worker summary
            int total_scan_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_scan_end_time - worker_scan_start_time).count();
            all_scan_times_ptr_[thread_id] += total_scan_time;
            size_t memory_scanned = ((size_t) curr_num_centroids) * (vector_dimension * sizeof(float) + sizeof(idx_t));
            all_throughputs_ptr_[thread_id] += ((1.0 * memory_scanned)/pow(10.0, 9))/((1.0 * total_scan_time)/pow(10.0, 6));

            // Record the end to end time
            int total_job_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_query_end_time - worker_query_start_time).count();
            all_job_times_ptr_[thread_id] += total_job_time;
            all_counts_ptr_[thread_id] += 1;
        }
    }
}

void DynamicIVF_C::partition_scan_worker_function(int thread_id) {
    int thread_numa_node = workers_numa_nodes_[thread_id];
    int num_numa_nodes = this->get_num_numa_nodes();
    int expected_numa_node_id = thread_numa_node;
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu;
    if(same_core_) {
        worker_cpu = thread_id % num_cpus;
    } else {
        int total_numa_nodes = numa_max_node() + 1;
        int worker_shift = num_scan_workers_ + total_numa_nodes - (num_scan_workers_ % total_numa_nodes);
        worker_cpu = (thread_id + worker_shift) % num_cpus;
    }

    // Pin thread to a CPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }

    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }

    // Set the preferred NUMA node for memory allocations
    int numa_node_id = numa_node_of_cpu(sched_getcpu());
    if(numa_node_id != expected_numa_node_id) {
        throw std::runtime_error("Thread not pinned to the expected numa node");
    }
    numa_set_preferred(numa_node_id);
    numa_set_strict(1);

    // Allocate the TopK buffer on this numa node
    moodycamel::BlockingConcurrentQueue<int>& requests_queue = jobs_queue_[numa_node_id];
    int vector_dimension = d_;
    faiss::MetricType metric_type = index_->metric_type;
    bool is_descending = metric_type == faiss::METRIC_INNER_PRODUCT;
    TopKVectors curr_cluster_buffer(max_vectors_per_cluster_, is_descending);

    // Load the index and the buffer
    auto dynamic_invlists = dynamic_cast<faiss::DynamicInvertedLists*>(ivf_index_->invlists);
    int job_id;

    // Mark that this worker is ready
    workers_ready_.fetch_add(1);

    while(true) {
        // Block till we have a request to process
        requests_queue.wait_dequeue(job_id);

        // See if this is a stop worker request
        auto worker_query_start_time = std::chrono::high_resolution_clock::now();
        if(job_id == -1) {
            break;
        }

        // Load the query details
        int query_id = job_query_id_[job_id];
        float* query_vector = curr_queries_per_node_[numa_node_id] + query_id * vector_dimension;
        size_t curr_cluster_id = job_search_cluster_id_[job_id];

        // Load the scan variables
        const float* list_vecs = (float*) dynamic_invlists->get_codes(curr_cluster_id);
        const idx_t* list_ids = dynamic_invlists->get_ids(curr_cluster_id);
        int list_size = dynamic_invlists->list_size(curr_cluster_id);
        if(list_size == 0) {
            final_result_mergers_[query_id].record_empty_job();
            continue;
        }

        if(!final_result_mergers_[query_id].currently_processing_query()) {
            continue;
        }

        curr_cluster_buffer.set_k(max_vectors_per_cluster_);

        // Verify that we are doing a read within the same numa node
        if(verify_numa_) {
            int curr_worker_numa_node = numa_node_of_cpu(sched_getcpu());
            int cluster_expected_node = dynamic_invlists->get_numa_node(curr_cluster_id);
            if(curr_worker_numa_node != cluster_expected_node) {
                throw std::runtime_error("Partition Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but cluster on node " + std::to_string(cluster_expected_node));
            }

            int codes_numa_node = -1;
            get_mempolicy(&codes_numa_node, NULL, 0, (void*) list_vecs, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != codes_numa_node) {
                throw std::runtime_error("Partition Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but codes on node " + std::to_string(codes_numa_node));
            }

            int ids_numa_node = -1;
            get_mempolicy(&ids_numa_node, NULL, 0, (void*) list_ids, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != ids_numa_node) {
                throw std::runtime_error("Partition Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but ids on node " + std::to_string(ids_numa_node));
            }

            int query_numa_node = -1;
            get_mempolicy(&query_numa_node, NULL, 0, (void*) query_vector, MPOL_F_NODE | MPOL_F_ADDR);
            if(curr_worker_numa_node != query_numa_node) {
                throw std::runtime_error("Partition Thread " + std::to_string(thread_id) + " running on numa node " + std::to_string(curr_worker_numa_node) + " but query on " + std::to_string(query_numa_node));
            }
        }

        // Perform the actual scan
        auto worker_scan_start_time = std::chrono::high_resolution_clock::now();
        scan_list(query_vector, list_vecs, list_ids, list_size, vector_dimension, curr_cluster_buffer, metric_type);
        auto worker_scan_end_time = std::chrono::high_resolution_clock::now();

        // Write the number of results
        std::vector<int64_t> overall_top_ids = curr_cluster_buffer.get_topk_indices();
        std::vector<float> overall_top_dists = curr_cluster_buffer.get_topk();
        int num_results = std::min((int) overall_top_ids.size(), max_vectors_per_cluster_);

        // Record the result for the query
        final_result_mergers_[query_id].batch_add(overall_top_dists.data(), overall_top_ids.data(), num_results);
        all_vectors_scanned_ptr_[thread_id] += list_size;

        // Log the times
        if(log_mode_) {
            auto worker_query_end_time = std::chrono::high_resolution_clock::now();

            // Record the worker summary
            int total_scan_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_scan_end_time - worker_scan_start_time).count();
            all_scan_times_ptr_[thread_id] += total_scan_time;
            size_t memory_scanned = ((size_t) list_size) * (vector_dimension * sizeof(float) + sizeof(idx_t));
            all_throughputs_ptr_[thread_id] += ((1.0 * memory_scanned)/pow(10.0, 9))/((1.0 * total_scan_time)/pow(10.0, 6));

            // Record the end to end time
            int total_job_time = std::chrono::duration_cast<std::chrono::microseconds>(worker_query_end_time - worker_query_start_time).count();
            all_job_times_ptr_[thread_id] += total_job_time;
            all_counts_ptr_[thread_id] += 1;
        }
    }
}
#endif