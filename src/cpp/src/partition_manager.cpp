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
    if (!parent_ && nlist != 1) {
        throw runtime_error("[PartitionManager] init_partitions: parent_ is null, so nlist must be 1.");
    } else if (parent_ && nlist != parent_->ntotal()) {
        throw runtime_error(
            "[PartitionManager] init_partitions: parent's ntotal does not match partition_ids.size(0).");
    }

    // Create the local partitions_:
    size_t code_size_bytes = static_cast<size_t>(dim * sizeof(float));
    partitions_ = std::make_shared<faiss::DynamicInvertedLists>(
        0,
        code_size_bytes
    );

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
        partitions_->add_entries(
            partition_ids_accessor[i],
            count,
            id.data_ptr<int64_t>(),
            as_uint8_ptr(v)
        );
    }

    std::cout << "[PartitionManager] init_partitions: Created " << nlist
            << " partitions, dimension=" << dim << "\n";
}

void PartitionManager::add(
    const Tensor &vectors,
    const Tensor &vector_ids,
    const Tensor &assignments
) {
    if (!parent_) {
        throw runtime_error("[PartitionManager] add: parent_ is null.");
    }
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
    if (!parent_) {
        throw runtime_error("[PartitionManager] remove: parent_ is null.");
    }
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
    int d = partitions_->d_;

    for (int i = 0; i < select_ids.size(0); i++) {
        int64_t list_no = select_ids[i].item<int64_t>();

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

void PartitionManager::add_partitions(shared_ptr<Clustering> partitions) {
    int64_t nlist = partitions->nlist();
    for (int64_t i = 0; i < nlist; i++) {
        int64_t list_no = partitions->partition_ids[i].item<int64_t>();
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


        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids[i].item<int64_t>();
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
    // Return how many lists are recognized
    return static_cast<int>(partitions_->nlist);
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
