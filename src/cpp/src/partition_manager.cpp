//
// partition_manager.cpp
// Created by Jason on 12/22/24
// Prompt for GitHub Copilot:
// - Conform to the Google style guide
// - Use descriptive variable names
//

#include "partition_manager.h"
#include "clustering.h"
#include <stdexcept>
#include <iostream>
#include "quake_index.h"

using std::runtime_error;

/**
 * @brief Helper: interpret float32 data as a uint8_t* (for storing in InvertedLists).
 */
static inline const uint8_t *as_uint8_ptr(const Tensor &float_tensor) {
    return reinterpret_cast<const uint8_t *>(float_tensor.data_ptr<float>());
}

PartitionManager::PartitionManager() {
    parent_ = nullptr;
    partitions_ = nullptr;
}

PartitionManager::~PartitionManager() {
    // no special cleanup
}

void PartitionManager::init_partitions(
    shared_ptr<QuakeIndex> parent,
    shared_ptr<Clustering> clustering
) {
    parent_ = parent;
    int64_t nlist = clustering->nlist();
    int64_t ntotal = clustering->ntotal();
    int64_t dim = clustering->dim();

    if (nlist <= 0 || ntotal <= 0) {
        throw runtime_error("[PartitionManager] init_partitions: nlist or ntotal is <= 0.");
    }

    // if parent is null make sure there is only a single partition
    if (parent_ && nlist != parent_->ntotal()) {
        throw runtime_error(
            "[PartitionManager] init_partitions: parent's ntotal does not match partition_ids.size(0).");
    }

    // Create the local partitions_:
    size_t code_size_bytes = static_cast<size_t>(dim * sizeof(float));
    partitions_ = std::make_shared<faiss::DynamicInvertedLists>(
        0,
        code_size_bytes
    );

    // partition ids are [0, 1, 2, ..., nlist-1]
    clustering->partition_ids = torch::arange(nlist, torch::kInt64);
    curr_partition_id_ = nlist;

    // Add an empty list for each partition ID
    auto partition_ids_accessor = clustering->partition_ids.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nlist; i++) {
        partitions_->add_list(partition_ids_accessor[i]);
    }

    for (int64_t i = 0; i < nlist; i++) {
        Tensor v = clustering->vectors[i];
        Tensor id = clustering->vector_ids[i];
        if (v.size(0) != id.size(0)) {
            throw runtime_error("[PartitionManager] init_partitions: mismatch in v.size(0) vs id.size(0).");
        }

        // Insert them
        size_t count = v.size(0);

        if (count == 0) {
            // nothing to add
            continue;
        } else {
            partitions_->add_entries(
                partition_ids_accessor[i],
                count,
                id.data_ptr<int64_t>(),
                as_uint8_ptr(v)
            );
        }
    }

    std::cout << "[PartitionManager] init_partitions: Created " << nlist
            << " partitions, dimension=" << dim << "\n";
}

void PartitionManager::add(
    const Tensor &vectors,
    const Tensor &vector_ids,
    const Tensor &assignments
) {
    if (!partitions_) {
        throw runtime_error("[PartitionManager] add: partitions_ is null. Did you call init_partitions?");
    }

    if (!vectors.defined() || !vector_ids.defined()) {
        throw runtime_error("[PartitionManager] add: vectors or vector_ids is undefined.");
    }
    if (vectors.size(0) != vector_ids.size(0)) {
        throw runtime_error("[PartitionManager] add: mismatch in vectors.size(0) and vector_ids.size(0).");
    }
    int64_t n = vectors.size(0);
    if (n == 0) {
        // nothing to add
        return;
    }

    // Infer dimension from vectors
    if (vectors.dim() != 2) {
        throw runtime_error("[PartitionManager] add: 'vectors' must be 2D [N, dim].");
    }
    int64_t dim = vectors.size(1);

    // If assignments is provided, use it. Otherwise, do centroid search via parent's IVF.
    vector<int64_t> partition_ids_for_each(n, -1);

    if (parent_ == nullptr) {
        // If parent_ is null, we assume there is only one partition and assign all vectors to it.
        partition_ids_for_each.assign(n, 0);
    } else {
        if (assignments.defined() && assignments.numel() > 0) {
            if (assignments.size(0) != n) {
                throw runtime_error("[PartitionManager] add: assignments.size(0) != vectors.size(0).");
            }
            // Copy assignments into vector<int64_t>
            auto a_ptr = assignments.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = a_ptr[i];
            }
        } else {
            auto search_params = make_shared<SearchParams>();
            search_params->k = 1;
            search_params->nprobe = parent_->nlist();
            search_params->recall_target = 1.0;
            auto parent_search_result = parent_->search(vectors, search_params);

            Tensor label_out = parent_search_result->ids;

            auto lbl_ptr = label_out.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = lbl_ptr[i];
            }
        }
    }

    // Now we insert each vector into partitions_ accordingly
    size_t code_size_bytes = partitions_->code_size;
    auto id_ptr = vector_ids.data_ptr<int64_t>();
    const uint8_t *code_ptr = as_uint8_ptr(vectors);

    for (int64_t i = 0; i < n; i++) {
        int64_t pid = partition_ids_for_each[i];
        // Insert one by one (TODO batched approach).
        partitions_->add_entries(
            pid,
            /*n_entry=*/1,
            &id_ptr[i],
            code_ptr + i * code_size_bytes
        );
    }
}

void PartitionManager::remove(const Tensor &ids) {
    if (!partitions_) {
        throw runtime_error("[PartitionManager] remove: partitions_ is null.");
    }
    if (!ids.defined() || ids.size(0) == 0) {
        return; // nothing to remove
    }

    std::set<faiss::idx_t> to_remove;
    auto ptr = ids.data_ptr<int64_t>();
    for (int64_t i = 0; i < ids.size(0); i++) {
        to_remove.insert(static_cast<faiss::idx_t>(ptr[i])); // todo slow
    }
    partitions_->remove_vectors(to_remove);
}

Tensor PartitionManager::get(const Tensor &ids) {
    auto ids_accessor = ids.accessor<int64_t, 1>();
    Tensor vectors = torch::empty({ids.size(0), partitions_->d_}, torch::kFloat32);
    auto vectors_ptr = vectors.data_ptr<float>();

    for (int64_t i = 0; i < ids.size(0); i++) {
        partitions_->get_vector_for_id(ids_accessor[i], vectors_ptr + i * partitions_->d_);
    }

    return vectors;
}

shared_ptr<Clustering> PartitionManager::select_partitions(const Tensor &select_ids, bool copy) {
    Tensor centroids = parent_->get(select_ids);
    vector<Tensor> cluster_vectors;
    vector<Tensor> cluster_ids;
    int d = (int) partitions_->d_;

    auto selected_ids_accessor = select_ids.accessor<int64_t, 1>();
    for (int i = 0; i < select_ids.size(0); i++) {
        int64_t list_no = selected_ids_accessor[i];

        int64_t list_size = partitions_->list_size(list_no);
        if (list_size == 0) {
            cluster_vectors.push_back(torch::empty({0, d}, torch::kFloat32));
            cluster_ids.push_back(torch::empty({0}, torch::kInt64));
            continue;
        }

        auto codes = partitions_->get_codes(list_no);
        auto ids = partitions_->get_ids(list_no);

        Tensor cluster_vectors_i = torch::from_blob((void *) codes, {list_size, d}, torch::kFloat32);
        Tensor cluster_ids_i = torch::from_blob((void *) ids, {list_size}, torch::kInt64);

        if (copy) {
            cluster_vectors_i = cluster_vectors_i.clone();
            cluster_ids_i = cluster_ids_i.clone();
        }

        cluster_vectors.push_back(cluster_vectors_i);
        cluster_ids.push_back(cluster_ids_i);
    }

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = select_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    return clustering;
}

shared_ptr<Clustering> PartitionManager::split_partitions(const Tensor &partition_ids) {
    // Calculate the number of new partitions after splitting
    int64_t num_partitions_to_split = partition_ids.size(0);
    int64_t num_splits = 2; // Example: splitting each partition into 2
    int64_t total_new_partitions = num_partitions_to_split * num_splits;
    int d = partitions_->d_;

    // Initialize tensors to hold new centroids, vectors, and IDs
    Tensor split_centroids = torch::empty({total_new_partitions, d}, torch::kFloat32);
    vector<Tensor> split_vectors;
    vector<Tensor> split_ids;

    split_vectors.reserve(total_new_partitions);
    split_ids.reserve(total_new_partitions);

    // Retrieve data from the selected partitions
    shared_ptr<Clustering> clustering = select_partitions(partition_ids);

    // Perform k-means splitting on each partition
    for (int64_t i = 0; i < partition_ids.size(0); ++i) {
        // Perform k-means with k=2 to split the partition into two
        assert(clustering->cluster_size(i) >= 4 && "Partition must have at least 8 vectors to split.");

        shared_ptr<Clustering> curr_split_clustering = kmeans(
            clustering->vectors[i],
            clustering->vector_ids[i],
            num_splits,
            parent_->metric_
        );

        // Assign the new centroid and corresponding vectors and IDs
        for (size_t j = 0; j < curr_split_clustering->nlist(); ++j) {
            split_centroids[i * num_splits + j] = curr_split_clustering->centroids[j];
            split_vectors.push_back(curr_split_clustering->vectors[j]);
            split_ids.push_back(curr_split_clustering->vector_ids[j]);
        }
    }

    shared_ptr<Clustering> split_clustering = std::make_shared<Clustering>();
    split_clustering->centroids = split_centroids;
    split_clustering->partition_ids = partition_ids;
    split_clustering->vectors = split_vectors;
    split_clustering->vector_ids = split_ids;

    return split_clustering;
}

void PartitionManager::refine_partitions(const Tensor &partition_ids, int iterations) {
    Tensor pids = partition_ids.defined() && partition_ids.size(0) > 0
        ? partition_ids
        : get_partition_ids();
    if (!pids.size(0) || !partitions_ || !parent_) {
        throw std::runtime_error("[PartitionManager] refine_partitions: no partitions to refine.");
    }

    faiss::MetricType mt = parent_->metric_;
    shared_ptr<Clustering> selected_parts = select_partitions(pids);
    Tensor centroids = selected_parts->centroids;
    int64_t nclusters = pids.size(0);
    int64_t d = centroids.size(1);

    // Precompute whether we do negative matmul (inner product) or cdist (L2).
    bool isIP = (mt == faiss::METRIC_INNER_PRODUCT);

    auto pids_accessor = pids.accessor<int64_t, 1>();

    for (int iter = 0; iter < iterations; iter++) {
        Tensor new_centroids = torch::zeros_like(centroids);  // [nclusters, d]
        Tensor counts = torch::zeros({nclusters}, torch::kLong);

        // Single parallel block for partial sums
        #pragma omp parallel
        {
            // Allocate local buffers once per thread
            Tensor local_centroids = torch::zeros_like(centroids);
            Tensor local_counts = torch::zeros({nclusters}, torch::kLong);

            // Distribute the cluster loop across threads
            #pragma omp for nowait
            for (int64_t i = 0; i < nclusters; i++) {
                Tensor vecs = selected_parts->vectors[i];  // shape [n_i, d]
                if (!vecs.defined() || !vecs.size(0)) continue;

                Tensor dist = isIP
                    ? -torch::mm(vecs, centroids.t())    // negative => min => argmax IP
                    : torch::cdist(vecs, centroids);

                // each row => best centroid
                auto min_res = dist.min(/*dim=*/1, /*keepdim=*/false);
                Tensor labels = std::get<1>(min_res);  // shape [n_i]

                auto lbl_acc = labels.accessor<int64_t, 1>();

                for (int64_t row = 0; row < vecs.size(0); row++) {
                    int64_t c = lbl_acc[row];
                    local_centroids[c] += vecs[row];
                    local_counts[c] += 1;
                }
            }

            // Merge thread-local buffers into global
            #pragma omp critical
            {
                new_centroids += local_centroids;
                counts += local_counts;
            }
        } // end parallel block

        // Update centroids (except after the last iteration, we just keep them)
        if (iter < iterations) {
            auto counts_acc = counts.accessor<int64_t, 1>();
            for (int64_t c = 0; c < nclusters; c++) {
                int64_t n = counts_acc[c];
                if (n > 0) {
                    centroids[c] = new_centroids[c] / (float)n;
                    if (isIP) {
                        float norm = centroids[c].norm().item<float>();
                        if (norm > 1e-12f) {
                            centroids[c] /= norm;
                        }
                    }
                }
            }
        }
    }

    // Final single-pass reassignment
    // #pragma omp parallel for
    for (int64_t i = 0; i < nclusters; i++) {
        int64_t pid = pids_accessor[i];
        Tensor vecs = selected_parts->vectors[i];
        Tensor ids = selected_parts->vector_ids[i];
        if (!vecs.defined() || !vecs.size(0)) continue;

        Tensor dist = isIP
            ? -torch::mm(vecs, centroids.t())
            : torch::cdist(vecs, centroids);
        auto min_res = dist.min(/*dim=*/1, /*keepdim=*/false);
        Tensor labels = std::get<1>(min_res);  // shape [n_i]

        std::vector<idx_t> to_remove;
        to_remove.reserve(ids.size(0));
        std::vector<int64_t> new_pids_array(ids.size(0));

        auto lbl_acc = labels.accessor<int64_t, 1>();
        auto ids_acc = ids.accessor<int64_t, 1>();
        for (int64_t row = 0; row < vecs.size(0); row++) {
            int64_t c = lbl_acc[row];
            if (c != i) {
                to_remove.push_back((idx_t)ids_acc[row]);
            }
            new_pids_array[row] = pids_accessor[c];
        }

        // We assume we store vectors as float => reinterpret cast to uint8
        partitions_->batch_update_entries(
            pid,
            new_pids_array.data(),
            (u_int8_t *) vecs.data_ptr(),
            ids_acc.data(),
            vecs.size(0)
        );

        std::cout << "Ntotal after update: " << partitions_->compute_ntotal() << std::endl;
        std::cout << "List size after update: " << partitions_->list_size(pid) << std::endl;
    }

    // Update parent index with refined centroids
    parent_->modify(pids, centroids);
}


void PartitionManager::add_partitions(shared_ptr<Clustering> partitions) {
    int64_t nlist = partitions->nlist();

    // generate new partition ids
    partitions->partition_ids = torch::arange(curr_partition_id_, curr_partition_id_ + nlist, torch::kInt64);
    curr_partition_id_ += nlist;

    auto p_ids_accessor = partitions->partition_ids.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nlist; i++) {
        int64_t list_no = p_ids_accessor[i];
        partitions_->add_list(list_no);
        partitions_->add_entries(
            list_no,
            partitions->vectors[i].size(0),
            partitions->vector_ids[i].data_ptr<int64_t>(),
            as_uint8_ptr(partitions->vectors[i])
        );
    }

    parent_->add(partitions->centroids, partitions->partition_ids);
}


void PartitionManager::delete_partitions(const Tensor &partition_ids, bool reassign) {
    if (parent_ != nullptr) {
        shared_ptr<Clustering> partitions = select_partitions(partition_ids);
        parent_->remove(partition_ids);

        auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();

        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids_accessor[i];
            partitions_->remove_list(list_no);
        }

        if (reassign) {
            for (int i = 0; i < partition_ids.size(0); i++) {
                Tensor vectors = partitions->vectors[i];
                Tensor ids = partitions->vector_ids[i];
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

void PartitionManager::distribute_flat(int n_partitions) {
    if (parent_ != nullptr) {
        throw std::runtime_error("Index is not flat");
    } else {
        auto codes = (float *) partitions_->get_codes(0);
        auto ids = (int64_t *) partitions_->get_ids(0);
        int64_t ntotal = partitions_->list_size(0);
        Tensor vectors = torch::from_blob(codes, {ntotal, d()}, torch::kFloat32);
        Tensor vector_ids = torch::from_blob(ids, {ntotal}, torch::kInt64);

        // randomly assign vectors to new partitions
        Tensor partition_assignments = torch::randint(n_partitions, {vectors.size(0)}, torch::kInt64);

        Tensor partition_ids = torch::arange(n_partitions, torch::kInt64);
        Tensor centroids = torch::empty({n_partitions, d()}, torch::kFloat32);
        vector<Tensor> new_vectors(n_partitions);
        vector<Tensor> new_ids(n_partitions);

        for (int i = 0; i < n_partitions; i++) {
            Tensor ids = torch::nonzero(partition_assignments == i).squeeze(1);
            new_vectors[i] = vectors.index_select(0, ids);
            new_ids[i] = vector_ids.index_select(0, ids);
            centroids[i] = new_vectors[i].mean(0);
        }

        shared_ptr<Clustering> new_partitions = std::make_shared<Clustering>();
        new_partitions->centroids = centroids;
        new_partitions->partition_ids = partition_ids;
        new_partitions->vectors = new_vectors;
        new_partitions->vector_ids = new_ids;

        init_partitions(nullptr, new_partitions);
    }
}

void PartitionManager::distribute_partitions(int num_workers) {
    std::cout << "Distributing partitions across " << num_workers << " workers." << std::endl;
    if (parent_ == nullptr) {
        std::cout << "Index is flat." << std::endl;
        throw std::runtime_error("Index is not partitioned");
    } else {

        int num_numa_nodes = std::min(get_num_numa_nodes(), num_workers);

        if (num_numa_nodes == -1) {
            std::cout << "No NUMA nodes detected." << std::endl;
            return;
        }

        // round robin assignment of partitions to numa nodes
        int next_numa_node = 0;
        for (auto partition : partitions_->partitions_) {
            int numa_node = next_numa_node;
            next_numa_node = (next_numa_node + 1) % num_numa_nodes;
            partition.second->set_numa_node(numa_node);
        }
    }
    std::cout << "Distributing partitions across " << num_workers << " workers." << std::endl;
}

int PartitionManager::get_num_numa_nodes() {
#ifdef QUAKE_USE_NUMA
    return numa_max_node() + 1;
#else
    return -1;
#endif
}


int64_t PartitionManager::ntotal() const {
    if (!partitions_) {
        return 0;
    }
    return partitions_->compute_ntotal();
}

int64_t PartitionManager::nlist() const {
    if (!partitions_) {
        return 0;
    }
    return partitions_->nlist;
}

int PartitionManager::d() const {
    if (!partitions_) {
        return 0;
    }
    return partitions_->d_;
}

Tensor PartitionManager::get_partition_ids() {
    return partitions_->get_partition_ids();
}


Tensor PartitionManager::get_partition_sizes(Tensor partition_ids) {

    if (!partitions_) {
        throw std::runtime_error("[PartitionManager] get_partition_sizes: partitions_ is null.");
    }

    if (!partition_ids.defined() || partition_ids.size(0) == 0) {
        partition_ids = get_partition_ids();
    }

    Tensor partition_sizes = torch::empty({partition_ids.size(0)}, torch::kInt64);
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    auto partition_sizes_accessor = partition_sizes.accessor<int64_t, 1>();
    for (int i = 0; i < partition_ids.size(0); i++) {
        int64_t list_no = partition_ids_accessor[i];
        partition_sizes_accessor[i] = partitions_->list_size(list_no);
    }

    return partition_sizes;
}


void PartitionManager::save(const string &path) {
    if (!partitions_) {
        throw std::runtime_error("No partitions to save");
    }
    partitions_->save(path);
}

void PartitionManager::load(const string &path) {
    if (!partitions_) {
        partitions_ = std::make_shared<faiss::DynamicInvertedLists>(0, 0);
    }
    partitions_->load(path);
}
