//
// partition_manager.cpp
// Created by Jason on 12/22/24
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
//

#include "partition_manager.h"
#include "clustering.h"
#include <stdexcept>
#include <iostream>
#include "quake_index.h"
#include <arrow/api.h>
#include <arrow/compute/api_vector.h>
#include <arrow/compute/api.h>

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
    shared_ptr<Clustering> clustering,
    bool check_uniques
) {
    if (debug_) {
        std::cout << "[PartitionManager] init_partitions: Entered." << std::endl;
    }
    parent_ = parent;
    int64_t nlist = clustering->nlist();
    int64_t ntotal = clustering->ntotal();
    int64_t dim = clustering->dim();

    if (nlist <= 0 && ntotal <= 0) {
        throw runtime_error("[PartitionManager] init_partitions: nlist and ntotal is <= 0.");
    }

    // if parent is not null, ensure consistency with parent's ntotal
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

    // Set partition ids as [0, 1, 2, ..., nlist-1]
    clustering->partition_ids = torch::arange(nlist, torch::kInt64);
    curr_partition_id_ = nlist;

    // Add an empty list for each partition ID
    auto partition_ids_accessor = clustering->partition_ids.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nlist; i++) {
        partitions_->add_list(partition_ids_accessor[i]);
        if (debug_) {
            std::cout << "[PartitionManager] init_partitions: Added empty list for partition " << i << std::endl;
        }
    }

    // Now insert the vectors into each partition
    for (int64_t i = 0; i < nlist; i++) {
        Tensor v = clustering->vectors[i];
        Tensor id = clustering->vector_ids[i];
        std::shared_ptr<arrow::Table> attributes_table = clustering->attributes_tables[i];
        if (v.size(0) != id.size(0)) {
            throw runtime_error("[PartitionManager] init_partitions: mismatch in v.size(0) vs id.size(0).");
        }

        size_t count = v.size(0);
        if (count == 0) {
            if (debug_) {
                std::cout << "[PartitionManager] init_partitions: Partition " << i << " is empty." << std::endl;
            }
            continue;
        } else {
            if (check_uniques_ && check_uniques) {
                // for each id insert into resident_ids_, if the id already exists, throw an error
                auto id_ptr = id.data_ptr<int64_t>();
                for (int64_t j = 0; j < count; j++) {
                    int64_t id_val = id_ptr[j];
                    if (resident_ids_.find(id_val) != resident_ids_.end()) {
                        throw runtime_error("[PartitionManager] init_partitions: vector ID already exists in the index.");
                    }
                    resident_ids_.insert(id_val);
                }
            }
            partitions_->add_entries(
                partition_ids_accessor[i],
                count,
                id.data_ptr<int64_t>(),
                as_uint8_ptr(v),
                attributes_table
            );
            if (debug_) {
                std::cout << "[PartitionManager] init_partitions: Added " << count
                          << " entries to partition " << partition_ids_accessor[i] << std::endl;
            }
        }
    }

    if (debug_) {
        std::cout << "[PartitionManager] init_partitions: Created " << nlist
                  << " partitions, dimension=" << dim << std::endl;
    } else {
        std::cout << "[PartitionManager] init_partitions: Created " << nlist << " partitions." << std::endl;
    }
}

std::shared_ptr<arrow::Table> PartitionManager::filterRowById(
    std::shared_ptr<arrow::Table> table, 
    int64_t target_id
) { 
    auto id_column = table->GetColumnByName("id");
    if (!id_column) {
        std::cerr << "Column 'id' not found in table." << std::endl;
        return nullptr;
    }
    
    // Create a filter expression (id == target_id)
    arrow::Datum column_data = id_column->chunk(0);
    arrow::Datum scalar_value = arrow::MakeScalar(target_id);
    auto filter_expr = arrow::compute::CallFunction("equal", {column_data, scalar_value});
    
    if (!filter_expr.ok()) {
        std::cerr << "Error creating filter expression: " << filter_expr.status().ToString() << std::endl;
        return nullptr;
    }
    
    // Apply the filter
    auto result = arrow::compute::Filter(table, filter_expr.ValueOrDie());
    if (!result.ok()) {
        std::cerr << "Error filtering table: " << result.status().ToString() << std::endl;
        return nullptr;
    }
    
    return result.ValueOrDie().table();
}


shared_ptr<ModifyTimingInfo> PartitionManager::add(
    const Tensor &vectors,
    const Tensor &vector_ids,
    const Tensor &assignments,
    bool check_uniques,
    std::shared_ptr<arrow::Table> attributes_table
) {

    auto timing_info = std::make_shared<ModifyTimingInfo>();

    if (debug_) {
        std::cout << "[PartitionManager] add: Received " << vectors.size(0)
                  << " vectors to add." << std::endl;
    }

    //////////////////////////////////////////
    /// Input validation
    //////////////////////////////////////////
    auto s1 = std::chrono::high_resolution_clock::now();
    if (!partitions_) {
        throw runtime_error("[PartitionManager] add: partitions_ is null. Did you call init_partitions?");
    }

    if(!attributes_table){
        throw runtime_error("[PartitionManager] add: attributes_table is null. Please add attributes for the vectors");
    }

    if (!vectors.defined() || !vector_ids.defined()) {
        throw runtime_error("[PartitionManager] add: vectors or vector_ids is undefined.");
    }

    if (vectors.size(0) != vector_ids.size(0)) {
        throw runtime_error("[PartitionManager] add: mismatch in vectors.size(0) and vector_ids.size(0).");
    }

    if(attributes_table->num_rows()!= vector_ids.size(0)){
        throw runtime_error("[PartitionManager] add: mismatch in attributes_table and vector_ids size.");
    }

    if(!attributes_table->GetColumnByName("id")){
        throw runtime_error("[PartitionManager] add: No vector_id column in attributes_table");
    }

    int64_t n = vectors.size(0);
    if (n == 0) {
        if (debug_) {
            std::cout << "[PartitionManager] add: No vectors to add. Exiting." << std::endl;
        }
        return timing_info;
    }
    if (vectors.dim() != 2) {
        throw runtime_error("[PartitionManager] add: 'vectors' must be 2D [N, dim].");
    }

    // check ids are below max id
    if ((vector_ids > std::numeric_limits<int32_t>::max()).any().item<bool>()) {
        throw runtime_error("[PartitionManager] add: vector_ids must be less than INT_MAX.");
    }

    // check ids are unique
    int64_t num_unique_ids = std::get<0>(torch::_unique(vector_ids)).size(0);
    if (num_unique_ids != n) {
        std::cout << std::get<0>(torch::sort(vector_ids)) << std::endl;
        throw runtime_error("[PartitionManager] add: vector_ids must be unique.");
    }

    if (check_uniques_ && check_uniques) {
        // for each id insert into resident_ids_, if the id already exists, throw an error
        auto id_ptr = vector_ids.data_ptr<int64_t>();
        for (int64_t j = 0; j < n; j++) {
            int64_t id_val = id_ptr[j];
            if (resident_ids_.find(id_val) != resident_ids_.end()) {
                throw runtime_error("[PartitionManager] init_partitions: vector ID already exists in the index.");
            }
            resident_ids_.insert(id_val);
        }
    }

    // checks assignments are less than partitions_->curr_list_id_
    if (assignments.defined() && (assignments >= curr_partition_id_).any().item<bool>()) {
        throw runtime_error("[PartitionManager] add: assignments must be less than partitions_->curr_list_id_.");
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    timing_info->input_validation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();


    //////////////////////////////////////////
    /// Determine partition assignments
    //////////////////////////////////////////
    auto s2 = std::chrono::high_resolution_clock::now();
    int64_t dim = vectors.size(1);
    // Determine partition assignments for each vector.
    vector<int64_t> partition_ids_for_each(n, -1);
    if (parent_ == nullptr) {
        partition_ids_for_each.assign(n, 0);
        if (debug_) {
            std::cout << "[PartitionManager] add: No parent index; assigning all vectors to partition 0." << std::endl;
        }
    } else {
        if (assignments.defined() && assignments.numel() > 0) {
            if (assignments.size(0) != n) {
                throw runtime_error("[PartitionManager] add: assignments.size(0) != vectors.size(0).");
            }
            auto a_ptr = assignments.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = a_ptr[i];
            }
        } else {
            if (debug_) {
                std::cout << "[PartitionManager] add: No assignments provided; performing parent search." << std::endl;
            }
            auto search_params = make_shared<SearchParams>();
            search_params->k = 1;
            search_params->nprobe = parent_->nlist();
            if (n > 10) {
                search_params->batched_scan = true;
            }
            auto parent_search_result = parent_->search(vectors, search_params);
            Tensor label_out = parent_search_result->ids;
            auto lbl_ptr = label_out.data_ptr<int64_t>();
            for (int64_t i = 0; i < n; i++) {
                partition_ids_for_each[i] = lbl_ptr[i];
            }
        }
    }
    auto e2 = std::chrono::high_resolution_clock::now();
    timing_info->find_partition_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count();

    //////////////////////////////////////////
    /// Add vectors to partitions
    //////////////////////////////////////////
    auto s3 = std::chrono::high_resolution_clock::now();
    size_t code_size_bytes = partitions_->code_size;
    auto id_ptr = vector_ids.data_ptr<int64_t>();
    auto id_accessor = vector_ids.accessor<int64_t, 1>();
    const uint8_t *code_ptr = as_uint8_ptr(vectors);

    for (int64_t i = 0; i < n; i++) {
        int64_t pid = partition_ids_for_each[i];
        if (debug_) {
            std::cout << "[PartitionManager] add: Inserting vector " << i << " with id " << id_accessor[i]
                      << " into partition " << pid << std::endl;
        }

        std::shared_ptr<arrow::Table> filtered_table_result = filterRowById(attributes_table, id_accessor[i]);

        partitions_->add_entries(
            pid,
            /*n_entry=*/1,
            id_ptr + i,
            code_ptr + i * code_size_bytes,
            filtered_table_result
        );

    }
    auto e3 = std::chrono::high_resolution_clock::now();
    timing_info->modify_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e3 - s3).count();
    return timing_info;
}

shared_ptr<ModifyTimingInfo> PartitionManager::remove(const Tensor &ids) {

    shared_ptr<ModifyTimingInfo> timing_info = std::make_shared<ModifyTimingInfo>();
    auto s1 = std::chrono::high_resolution_clock::now();
    if (debug_) {
        std::cout << "[PartitionManager] remove: Removing " << ids.size(0) << " ids." << std::endl;
    }
    if (!partitions_) {
        throw runtime_error("[PartitionManager] remove: partitions_ is null.");
    }
    if (!ids.defined() || ids.size(0) == 0) {
        if (debug_) {
            std::cout << "[PartitionManager] remove: No ids provided. Exiting." << std::endl;
        }
        return timing_info;
    }

    if (check_uniques_) {
        // ids must be in resident_ids_
        auto id_ptr = ids.data_ptr<int64_t>();
        for (int64_t i = 0; i < ids.size(0); i++) {
            int64_t id_val = id_ptr[i];
            if (resident_ids_.find(id_val) == resident_ids_.end()) {
                // print out op ids
                std::cout << ids << std::endl;
                // print out ids in the index
                for (auto &id : resident_ids_) {
                    std::cout << id << " ";
                }
                std::cout << resident_ids_.size() << std::endl;
                throw runtime_error("[PartitionManager] remove: vector ID does not exist in the index.");
            }
            resident_ids_.erase(id_val);
        }
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    timing_info->input_validation_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();

    auto s2 = std::chrono::high_resolution_clock::now();
    std::set<faiss::idx_t> to_remove;
    auto ptr = ids.data_ptr<int64_t>();
    for (int64_t i = 0; i < ids.size(0); i++) {
        to_remove.insert(static_cast<faiss::idx_t>(ptr[i]));
    }
    auto e2 = std::chrono::high_resolution_clock::now();
    timing_info->find_partition_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count();

    auto s3 = std::chrono::high_resolution_clock::now();
    partitions_->remove_vectors(to_remove);
    // TODO: Remove associated attribute data as well??? 
    if (debug_) {
        std::cout << "[PartitionManager] remove: Completed removal." << std::endl;
    }
    auto e3 = std::chrono::high_resolution_clock::now();
    timing_info->modify_time_us = std::chrono::duration_cast<std::chrono::microseconds>(e3 - s3).count();

    return timing_info;
}

Tensor PartitionManager::get(const Tensor &ids) {
    if (debug_) {
        std::cout << "[PartitionManager] get: Retrieving vectors for " << ids.size(0) << " ids." << std::endl;
    }
    auto ids_accessor = ids.accessor<int64_t, 1>();
    Tensor vectors = torch::empty({ids.size(0), partitions_->d_}, torch::kFloat32);
    auto vectors_ptr = vectors.data_ptr<float>();

    for (int64_t i = 0; i < ids.size(0); i++) {
        partitions_->get_vector_for_id(ids_accessor[i], vectors_ptr + i * partitions_->d_);
    }
    if (debug_) {
        std::cout << "[PartitionManager] get: Retrieval complete." << std::endl;
    }
    return vectors;
}

shared_ptr<Clustering> PartitionManager::select_partitions(const Tensor &select_ids, bool copy) {
    if (debug_) {
        std::cout << "[PartitionManager] select_partitions: Selecting partitions from provided ids." << std::endl;
    }
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
            if (debug_) {
                std::cout << "[PartitionManager] select_partitions: Partition " << list_no << " is empty." << std::endl;
            }
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
        if (debug_) {
            std::cout << "[PartitionManager] select_partitions: Selected partition " << list_no
                      << " with " << list_size << " entries." << std::endl;
        }
    }

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = select_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    if (debug_) {
        std::cout << "[PartitionManager] select_partitions: Completed selection." << std::endl;
    }
    return clustering;
}

shared_ptr<Clustering> PartitionManager::split_partitions(const Tensor &partition_ids) {
    if (debug_) {
        std::cout << "[PartitionManager] split_partitions: Splitting " << partition_ids.size(0)
                  << " partitions." << std::endl;
    }
    int64_t num_partitions_to_split = partition_ids.size(0);
    int64_t num_splits = 2;
    int64_t total_new_partitions = num_partitions_to_split * num_splits;
    int d = partitions_->d_;

    Tensor split_centroids = torch::empty({total_new_partitions, d}, torch::kFloat32);
    vector<Tensor> split_vectors;
    vector<Tensor> split_ids;

    split_vectors.reserve(total_new_partitions);
    split_ids.reserve(total_new_partitions);

    shared_ptr<Clustering> clustering = select_partitions(partition_ids);

    for (int64_t i = 0; i < partition_ids.size(0); ++i) {
        // Ensure enough vectors to split
        assert(clustering->cluster_size(i) >= 4 && "Partition must have at least 8 vectors to split.");
        shared_ptr<Clustering> curr_split_clustering = kmeans(
            clustering->vectors[i],
            clustering->vector_ids[i],
            num_splits,
            parent_->metric_
        );

        for (size_t j = 0; j < curr_split_clustering->nlist(); ++j) {
            split_centroids[i * num_splits + j] = curr_split_clustering->centroids[j];
            split_vectors.push_back(curr_split_clustering->vectors[j]);
            split_ids.push_back(curr_split_clustering->vector_ids[j]);
            if (debug_) {
                std::cout << "[PartitionManager] split_partitions: Partition "
                          << clustering->partition_ids[i].item<int64_t>()
                          << " split: created new partition with centroid index "
                          << (i * num_splits + j) << std::endl;
            }
        }
    }

    shared_ptr<Clustering> split_clustering = std::make_shared<Clustering>();
    split_clustering->centroids = split_centroids;
    split_clustering->partition_ids = partition_ids;
    split_clustering->vectors = split_vectors;
    split_clustering->vector_ids = split_ids;

    if (debug_) {
        std::cout << "[PartitionManager] split_partitions: Completed splitting." << std::endl;
    }
    return split_clustering;
}

void PartitionManager::refine_partitions(const Tensor &partition_ids, int iterations) {
    if (debug_) {
        std::cout << "[PartitionManager] refine_partitions: Refining partitions with iterations = "
                  << iterations << std::endl;
    }
    Tensor pids = partition_ids.defined() && partition_ids.size(0) > 0
        ? partition_ids
        : get_partition_ids();
    if (!pids.size(0) || !partitions_ || !parent_) {
        throw runtime_error("[PartitionManager] refine_partitions: no partitions to refine.");
    }

    faiss::MetricType mt = parent_->metric_;
    shared_ptr<Clustering> selected_parts = select_partitions(pids);
    Tensor centroids = selected_parts->centroids;
    int64_t nclusters = pids.size(0);
    int64_t d = centroids.size(1);
    bool isIP = (mt == faiss::METRIC_INNER_PRODUCT);

    auto pids_accessor = pids.accessor<int64_t, 1>();

    for (int iter = 0; iter < iterations; iter++) {
        Tensor new_centroids = torch::zeros_like(centroids);
        Tensor counts = torch::zeros({nclusters}, torch::kLong);

        // #pragma omp parallel
        {
            Tensor local_centroids = torch::zeros_like(centroids);
            Tensor local_counts = torch::zeros({nclusters}, torch::kLong);

            // #pragma omp for nowait
            for (int64_t i = 0; i < nclusters; i++) {
                Tensor vecs = selected_parts->vectors[i];
                if (!vecs.defined() || !vecs.size(0)) continue;

                Tensor dist = isIP
                    ? -torch::mm(vecs, centroids.t())
                    : torch::cdist(vecs, centroids);

                auto min_res = dist.min(/*dim=*/1, /*keepdim=*/false);
                Tensor labels = std::get<1>(min_res);
                auto lbl_acc = labels.accessor<int64_t, 1>();

                for (int64_t row = 0; row < vecs.size(0); row++) {
                    int64_t c = lbl_acc[row];
                    local_centroids[c] += vecs[row];
                    local_counts[c] += 1;
                }
            }
            // #pragma omp critical
            {
                new_centroids += local_centroids;
                counts += local_counts;
            }
        }

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

    // Final assignment
    for (int64_t i = 0; i < nclusters; i++) {
        int64_t pid = pids_accessor[i];
        Tensor vecs = selected_parts->vectors[i];
        Tensor ids = selected_parts->vector_ids[i];
        if (!vecs.defined() || !vecs.size(0)) continue;

        Tensor dist = isIP
            ? -torch::mm(vecs, centroids.t())
            : torch::cdist(vecs, centroids);
        auto min_res = dist.min(/*dim=*/1, /*keepdim=*/false);
        Tensor labels = std::get<1>(min_res);

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

        partitions_->batch_update_entries(
            pid,
            new_pids_array.data(),
            (u_int8_t *) vecs.data_ptr(),
            ids_acc.data(),
            vecs.size(0)
        );
        if (debug_) {
            std::cout << "[PartitionManager] refine_partitions: After updating partition "
                      << pid << ", new size: " << partitions_->list_size(pid) << std::endl;
        }
    }

    parent_->modify(pids, centroids);
    if (debug_) {
        std::cout << "[PartitionManager] refine_partitions: Completed refinement." << std::endl;
    }
}

void PartitionManager::add_partitions(shared_ptr<Clustering> partitions) {
    int64_t nlist = partitions->nlist();
    partitions->partition_ids = torch::arange(curr_partition_id_, curr_partition_id_ + nlist, torch::kInt64);
    curr_partition_id_ += nlist;

    if (debug_) {
        std::cout << "[PartitionManager] add_partitions: Adding " << nlist << " partitions." << std::endl;
        std::cout << "[PartitionManager] add_partitions: New partition IDs: " << partitions->partition_ids << std::endl;
        std::cout << "[PartitionManager] add_partitions: Current partition ID: " << curr_partition_id_ << std::endl;
        std::cout << "[PartitionManager] add_partitions: Nlist: " << nlist << std::endl;
    }

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
        if (debug_) {
            std::cout << "[PartitionManager] add_partitions: Added partition " << list_no
                      << " with " << partitions->vectors[i].size(0) << " vectors." << std::endl;
        }
    }

    parent_->add(partitions->centroids, partitions->partition_ids);
    if (debug_) {
        std::cout << "[PartitionManager] add_partitions: Completed adding partitions." << std::endl;
    }
}

void PartitionManager::delete_partitions(const Tensor &partition_ids, bool reassign) {
    if (parent_ != nullptr) {
        shared_ptr<Clustering> partitions = select_partitions(partition_ids, true);
        parent_->remove(partition_ids);

        auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
        for (int i = 0; i < partition_ids.size(0); i++) {
            int64_t list_no = partition_ids_accessor[i];
            partitions_->remove_list(list_no);
            if (debug_) {
                std::cout << "[PartitionManager] delete_partitions: Removed partition " << list_no << std::endl;
            }
        }

        if (reassign) {
            if (debug_) {
                std::cout << "[PartitionManager] delete_partitions: Reassigning vectors from deleted partitions." << std::endl;
            }
            for (int i = 0; i < partition_ids.size(0); i++) {
                Tensor vectors = partitions->vectors[i];
                Tensor ids = partitions->vector_ids[i];
                if (vectors.size(0) == 0) {
                    continue;
                }
                add(vectors, ids, Tensor(), false);
            }
        }
    } else {
        throw runtime_error("Index is not partitioned");
    }
}

void PartitionManager::distribute_flat(int n_partitions) {
    if (debug_) {
        std::cout << "[PartitionManager] distribute_flat: Distributing flat index into " << n_partitions << " partitions." << std::endl;
    }
    if (parent_ != nullptr) {
        throw runtime_error("Index is not flat");
    } else {
        auto codes = (float *) partitions_->get_codes(0);
        auto ids = (int64_t *) partitions_->get_ids(0);
        int64_t ntotal = partitions_->list_size(0);
        Tensor vectors = torch::from_blob(codes, {ntotal, d()}, torch::kFloat32);
        Tensor vector_ids = torch::from_blob(ids, {ntotal}, torch::kInt64);

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
            if (debug_) {
                std::cout << "[PartitionManager] distribute_flat: Partition " << i
                          << " assigned " << new_vectors[i].size(0) << " vectors." << std::endl;
            }
        }

        shared_ptr<Clustering> new_partitions = std::make_shared<Clustering>();
        new_partitions->centroids = centroids;
        new_partitions->partition_ids = partition_ids;
        new_partitions->vectors = new_vectors;
        new_partitions->vector_ids = new_ids;

        init_partitions(nullptr, new_partitions, false);
        if (debug_) {
            std::cout << "[PartitionManager] distribute_flat: Distribution complete." << std::endl;
        }
    }
}

void PartitionManager::distribute_partitions(int num_workers) {
    if (debug_) {
        std::cout << "[PartitionManager] distribute_partitions: Attempting to distribute partitions across "
                  << num_workers << " workers." << std::endl;
    }
    if (parent_ == nullptr) {
        if (debug_) {
            std::cout << "[PartitionManager] distribute_partitions: Index is flat." << std::endl;
        }
        throw runtime_error("Index is not partitioned");
    } else {
        // TODO: Implement distribute_partitions with logging as needed.
        if (debug_) {
            std::cout << "[PartitionManager] distribute_partitions: (Not yet implemented)" << std::endl;
        }
    }
}

int64_t PartitionManager::ntotal() const {
    if (!partitions_) {
        return 0;
    }
    return partitions_->ntotal();
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
    if (debug_) {
        std::cout << "[PartitionManager] get_partition_ids: Retrieving partition ids." << std::endl;
    }
    return partitions_->get_partition_ids();
}

Tensor PartitionManager::get_ids() {
    Tensor partition_ids = get_partition_ids();
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    vector<Tensor> ids;

    for (int i = 0; i < partition_ids.size(0); i++) {
        int64_t list_no = partition_ids_accessor[i];
        Tensor curr_ids = torch::from_blob((void *) partitions_->get_ids(list_no),
            {(int64_t) partitions_->list_size(list_no)}, torch::kInt64);
        ids.push_back(curr_ids);
    }

    return torch::cat(ids, 0);
}

Tensor PartitionManager::get_partition_sizes(Tensor partition_ids) {
    if (debug_) {
        std::cout << "[PartitionManager] get_partition_sizes: Getting sizes for partitions." << std::endl;
    }
    if (!partitions_) {
        throw runtime_error("[PartitionManager] get_partition_sizes: partitions_ is null.");
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
        if (debug_) {
            std::cout << "[PartitionManager] get_partition_sizes: Partition " << list_no
                      << " size: " << partition_sizes_accessor[i] << std::endl;
        }
    }
    return partition_sizes;
}

bool PartitionManager::validate() {
    if (debug_) {
        std::cout << "[PartitionManager] validate: Validating partitions." << std::endl;
    }
    if (!partitions_) {
        throw runtime_error("[PartitionManager] validate: partitions_ is null.");
    }
    return true;
}


void PartitionManager::save(const string &path) {
    if (debug_) {
        std::cout << "[PartitionManagerPartitionManager] save: Saving partitions to " << path << std::endl;
    }
    if (!partitions_) {
        throw runtime_error("No partitions to save");
    }
    partitions_->save(path);
    if (debug_) {
        std::cout << "[PartitionManager] save: Save complete." << std::endl;
    }
}

void PartitionManager::load(const string &path) {
    if (debug_) {
        std::cout << "[PartitionManager] load: Loading partitions from " << path << std::endl;
    }
    if (!partitions_) {
        partitions_ = std::make_shared<faiss::DynamicInvertedLists>(0, 0);
    }
    partitions_->load(path);
    curr_partition_id_ = partitions_->nlist;

    if (check_uniques_) {
        // add ids into resident set
        Tensor ids = get_ids();
        auto ids_a = ids.accessor<int64_t, 1>();
        for (int i = 0; i < ids.size(0); i++) {
            resident_ids_.insert(ids_a[i]);
        }
    }

    if (debug_) {
        std::cout << "[PartitionManager] load: Load complete." << std::endl;
    }
}