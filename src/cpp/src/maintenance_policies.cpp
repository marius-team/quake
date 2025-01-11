//
// Created by Jason on 10/7/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <maintenance_policies.h>
#include <geometry.h>
#include <list_scanning.h>
#include <quake_index.h>

vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> > MaintenancePolicy::get_split_history() {
    vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> > split_history;

    // iterate over all splits in the split records, record the hit count of the deleted partition and the two split partitions
    for (auto tup: split_records_) {
        int64_t parent_id = std::get<0>(tup);
        int64_t left_id = std::get<1>(tup).first;
        int64_t right_id = std::get<1>(tup).second;

        // get hits for parent
        int64_t parent_hits = deleted_partition_hit_rate_[parent_id];
        int64_t left_hits;
        int64_t right_hits;

        // get the hits for the left_id and right_id
        if (deleted_partition_hit_rate_.find(left_id) == deleted_partition_hit_rate_.end()) {
            // partition has not been deleted
            left_hits = per_partition_hits_[left_id];
        } else {
            left_hits = deleted_partition_hit_rate_[left_id];
        }

        if (deleted_partition_hit_rate_.find(right_id) == deleted_partition_hit_rate_.end()) {
            right_hits = per_partition_hits_[right_id];
        } else {
            right_hits = deleted_partition_hit_rate_[right_id];
        }

        split_history.push_back(std::make_tuple(parent_id, parent_hits, left_id, left_hits, right_id, right_hits));
    }

    return split_history;
}

shared_ptr<PartitionState> MaintenancePolicy::get_partition_state(bool only_modified) {
    vector<int64_t> partition_ids;
    vector<int64_t> partition_sizes;
    vector<float> partition_hit_rate;

    if (only_modified) {
        partition_ids = vector<int64_t>(modified_partitions_.begin(), modified_partitions_.end());
    } else {
        Tensor p_ids = partition_manager_->partitions_->get_partition_ids();
        auto p_ids_accessor = p_ids.accessor<int64_t, 1>();
        partition_ids = vector<int64_t>(p_ids.size(0));
        for (int i = 0; i < p_ids.size(0); i++) {
            partition_ids[i] = p_ids_accessor[i];
        }
    }

    // for each partition id get the size and compute the hit rate
    for (int64_t partition_id: partition_ids) {
        int64_t partition_size = partition_manager_->partitions_->list_size(partition_id);
        int64_t hits = per_partition_hits_[partition_id];
        int curr_window_size = std::max(std::min(window_size_, curr_query_id_), 1);
        float hit_rate = hits / (float) curr_window_size;
        partition_sizes.push_back(partition_size);
        partition_hit_rate.push_back(hit_rate);
    }

    shared_ptr<PartitionState> state = std::make_shared<PartitionState>();
    state->partition_ids = partition_ids;
    state->partition_sizes = partition_sizes;
    state->partition_hit_rate = partition_hit_rate;

    return state;
}

void MaintenancePolicy::set_partition_modified(int64_t partition_id) {
    modified_partitions_.insert(partition_id);
}

void MaintenancePolicy::set_partition_unmodified(int64_t partition_id) {
    modified_partitions_.erase(partition_id);
}

void MaintenancePolicy::decrement_hit_count(int64_t partition_id) {
    // check if id is in the partition hits
    if (per_partition_hits_.find(partition_id) == per_partition_hits_.end()) {
        // find it in the removed partition hits
        if (ancestor_partition_hits_.find(partition_id) == ancestor_partition_hits_.end()) {
            throw std::runtime_error("Partition not found");
        }

        // don't go past zero
        if (ancestor_partition_hits_[partition_id] > 0) {
            int64_t left_partition_id;
            int64_t right_partition_id;
            std::tie(left_partition_id, right_partition_id) = split_records_[partition_id];
            decrement_hit_count(left_partition_id);
            decrement_hit_count(right_partition_id);
            ancestor_partition_hits_[partition_id]--;
        }
    } else {
        if (per_partition_hits_[partition_id] > 0) {
            per_partition_hits_[partition_id]--;
        }
    }
}

void MaintenancePolicy::increment_hit_count(vector<int64_t> hit_partition_ids) {
    // Ensure ntotal is not zero to avoid division by zero
    const int64_t total_vectors = partition_manager_->ntotal();
    if (total_vectors == 0) {
        throw std::runtime_error("Error: index_->ntotal() is zero.");
    }

    // Calculate the scan fraction for the new query
    int vectors_scanned_new = 0;
    for (const auto &hit: hit_partition_ids) {
        per_partition_hits_[hit]++;
        vectors_scanned_new += partition_manager_->partitions_->list_size(hit);
        modified_partitions_.insert(hit);
    }
    float new_scan_fraction = static_cast<float>(vectors_scanned_new) / total_vectors;

    // Determine the current query index in the circular buffer
    int current_query_index = curr_query_id_ % window_size_;

    // If the window is full, remove the oldest query's contribution
    if (curr_query_id_ >= window_size_) {
        const auto &oldest_hits = per_query_hits_[current_query_index];
        const auto &oldest_sizes = per_query_scanned_partitions_sizes_[current_query_index];

        // Decrement per_partition_hits_ based on the oldest hits
        for (const auto &hit: oldest_hits) {
            decrement_hit_count(hit);
        }

        // Calculate the scan fraction of the oldest query
        int vectors_scanned_old = 0;
        for (const auto &size: oldest_sizes) {
            vectors_scanned_old += size;
        }
        float oldest_scan_fraction = static_cast<float>(vectors_scanned_old) / total_vectors;

        // Update the running sum by removing the oldest scan fraction
        running_sum_scan_fraction_ -= oldest_scan_fraction;
    }

    // Update the circular buffers with the new query's data
    per_query_hits_[current_query_index] = hit_partition_ids;

    // Calculate and store the sizes for the new hits
    std::vector<int64_t> hits_sizes;
    hits_sizes.reserve(hit_partition_ids.size());
    for (const auto &hit: hit_partition_ids) {
        hits_sizes.push_back(partition_manager_->partitions_->list_size(hit));
    }
    per_query_scanned_partitions_sizes_[current_query_index] = hits_sizes;

    // Add the new scan fraction to the running sum
    running_sum_scan_fraction_ += new_scan_fraction;

    // Calculate the number of queries in the window
    int current_window_size = std::min(static_cast<int>(curr_query_id_) + 1, window_size_);

    // Update the running average scan fraction
    current_scan_fraction_ = running_sum_scan_fraction_ / static_cast<float>(current_window_size);

    if (current_scan_fraction_ == 0.0) {
        current_scan_fraction_ = 1.0;
    }

    // Increment the query ID for the next update
    curr_query_id_++;
}

vector<float> MaintenancePolicy::estimate_split_delta(shared_ptr<PartitionState> state) {
    vector<float> deltas;
    int n_partitions = partition_manager_->nlist();
    int k = 10;
    float alpha = alpha_;

    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions + 1, k) - latency_estimator_->
                            estimate_scan_latency(n_partitions, k));

    for (int i = 0; i < state->partition_ids.size(); i++) {
        int64_t partition_id = state->partition_ids[i];
        int64_t partition_size = state->partition_sizes[i];
        float hit_rate = state->partition_hit_rate[i];

        // delta reassign overhead
        float old_cost = latency_estimator_->estimate_scan_latency(partition_size, k) * hit_rate;
        float new_cost = latency_estimator_->estimate_scan_latency(partition_size / 2, k) * hit_rate * (2 * alpha);
        float delta = delta_overhead + new_cost - old_cost;
        deltas.push_back(delta);
    }

    return deltas;
}

vector<float> MaintenancePolicy::estimate_delete_delta(shared_ptr<PartitionState> state) {
    vector<float> deltas;
    int n_partitions = partition_manager_->nlist();
    int k = 10;
    float alpha = alpha_;

    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions - 1, k) - latency_estimator_->
                            estimate_scan_latency(n_partitions, k));

    for (int i = 0; i < state->partition_ids.size(); i++) {
        int64_t partition_id = state->partition_ids[i];
        int64_t partition_size = state->partition_sizes[i];
        float hit_rate = state->partition_hit_rate[i];
        // float old_cost = latency_estimator_->estimate_scan_latency(partition_size, k) * hit_rate;

        // increase in cost due to reassigning vectors to neighboring
        float delta_reassign = current_scan_fraction_ * latency_estimator_->estimate_scan_latency(partition_size, k);

        // increase in cost due to increase in number of partiti
        float delta = delta_overhead + delta_reassign;
        deltas.push_back(delta);
    }

    return deltas;
}

float MaintenancePolicy::estimate_add_level_delta() {
    return 0.0;
}

float MaintenancePolicy::estimate_remove_level_delta() {
    return 0.0;
}

shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::maintenance() {
    shared_ptr<MaintenanceTimingInfo> timing_info = std::make_shared<MaintenanceTimingInfo>();

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    Tensor delete_ids = check_and_delete_partitions();
    auto end = std::chrono::high_resolution_clock::now();
    timing_info->delete_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    timing_info->n_deletes = delete_ids.size(0);

    // Todo refine deletes
    // start = std::chrono::high_resolution_clock::now();
    // if (delete_ids.size(0) > 0) {
    //     refine_delete(delete_centroids);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // timing_info.delete_refine_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    Tensor split_ids;
    Tensor old_centroids;
    Tensor old_ids;
    std::tie(split_ids, old_centroids, old_ids) = check_and_split_partitions();
    end = std::chrono::high_resolution_clock::now();
    timing_info->split_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    timing_info->n_splits = old_ids.size(0);

    start = std::chrono::high_resolution_clock::now();
    if (split_ids.size(0) > 0) {
        local_refinement(split_ids, refinement_radius_);
    }
    end = std::chrono::high_resolution_clock::now();
    timing_info->split_refine_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info->total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    return timing_info;
}

void MaintenancePolicy::add_split(int64_t old_partition_id, int64_t left_partition_id, int64_t right_partition_id) {
    int64_t num_queries = std::max(std::min(curr_query_id_, window_size_), 1);
    split_records_[old_partition_id] = std::make_pair(left_partition_id, right_partition_id);
    per_partition_hits_[left_partition_id] = per_partition_hits_[old_partition_id];
    per_partition_hits_[right_partition_id] = per_partition_hits_[old_partition_id];
    ancestor_partition_hits_[old_partition_id] = per_partition_hits_[old_partition_id];
    deleted_partition_hit_rate_[old_partition_id] = (float) per_partition_hits_[old_partition_id] / num_queries; // TODO is this needed?
    per_partition_hits_.erase(old_partition_id);
}

void MaintenancePolicy::add_partition(int64_t partition_id, int64_t hits) {
    per_partition_hits_[partition_id] = hits;
}

void MaintenancePolicy::remove_partition(int64_t partition_id) {
    int64_t num_queries = std::max(std::min(curr_query_id_, window_size_), 1);
    ancestor_partition_hits_[partition_id] = per_partition_hits_[partition_id];
    deleted_partition_hit_rate_[partition_id] = per_partition_hits_[partition_id] / num_queries;
    per_partition_hits_.erase(partition_id);
}

void MaintenancePolicy::refine_partitions(Tensor partition_ids, int refinement_iterations) {
    // TODO
}

QueryCostMaintenance::QueryCostMaintenance(std::shared_ptr<PartitionManager> partition_manager,
                                           shared_ptr<MaintenancePolicyParams> params) {
    maintenance_policy_name_ = "query_cost";
    window_size_ = params->window_size;
    refinement_radius_ = params->refinement_radius;
    min_partition_size_ = params->min_partition_size;
    alpha_ = params->alpha;
    enable_split_rejection_ = params->enable_split_rejection;
    enable_delete_rejection_ = params->enable_delete_rejection;

    curr_query_id_ = 0;
    per_query_hits_ = vector<vector<int64_t> >(window_size_);
    per_query_scanned_partitions_sizes_ = vector<vector<int64_t> >(window_size_);
    partition_manager_ = partition_manager;
    current_scan_fraction_ = 1.0;

    // Specify the file where you want to store or load the profile
    std::string profile_filename = "latency_profile.csv";

    // Create the latency estimator
    latency_estimator_ = std::make_shared<ListScanLatencyEstimator>(
        partition_manager_->d(),
        latency_grid_n_values_,
        latency_grid_k_values_,
        n_trials_,
        false,
        profile_filename);
}

float QueryCostMaintenance::compute_alpha_for_window() {
    // go over the split history and compute the alpha
    // alpha is the fraction of decrease in hit rate resulting from a split
    // go over the split history and compute the alpha

    if (split_records_.empty()) {
        return 0;
    }

    float total_alpha = 0.0;
    for (const auto &split: split_records_) {
        int64_t parent_id = std::get<0>(split);
        int64_t left_id = std::get<1>(split).first;
        int64_t right_id = std::get<1>(split).second;

        // check if children are in the deleted partition hits or the per partition hits
        float left_hit_rate;
        float right_hit_rate;
        if (deleted_partition_hit_rate_.find(left_id) != deleted_partition_hit_rate_.end()) {
            left_hit_rate = deleted_partition_hit_rate_[left_id];
        } else {
            left_hit_rate = (float) per_partition_hits_[left_id] / window_size_;
        }

        if (deleted_partition_hit_rate_.find(right_id) != deleted_partition_hit_rate_.end()) {
            right_hit_rate = deleted_partition_hit_rate_[right_id];
        } else {
            right_hit_rate = (float) per_partition_hits_[right_id] / window_size_;
        }

        // parent is in the deleted partition hits
        float parent_hit_rate = deleted_partition_hit_rate_[parent_id];

        // compute the alpha
        float curr_alpha = (float) (left_hit_rate + right_hit_rate) / (2 * parent_hit_rate);
        total_alpha += curr_alpha;
    }
    return total_alpha / split_records_.size();
}


Tensor QueryCostMaintenance::check_and_delete_partitions() {
    if (partition_manager_->parent_ == nullptr) {
        return {};
    }

    int64_t n_partitions = partition_manager_->parent_->ntotal();
    int64_t num_queries = std::min(curr_query_id_, window_size_);
    shared_ptr<PartitionState> state = get_partition_state(true);
    vector<float> delete_delta = estimate_delete_delta(state);

    Tensor delete_delta_tensor =
            torch::from_blob(delete_delta.data(), {(int64_t) delete_delta.size()}, torch::kFloat32).clone();

    vector<int64_t> partitions_to_delete;
    vector<int64_t> deleted_partition_ids;
    vector<float> partition_to_delete_delta;
    vector<float> partition_to_delete_hit_rate;
    for (int i = 0; i < delete_delta.size(); i++) {
        if (delete_delta[i] < -delete_threshold_ns_) {
            int64_t curr_partition_id = state->partition_ids[i];
            partitions_to_delete.push_back(curr_partition_id);
            partition_to_delete_delta.push_back(delete_delta[i]);
            partition_to_delete_hit_rate.push_back(state->partition_hit_rate[i]);
            remove_partition(curr_partition_id);
        }
    }

    // delete the partitions
    Tensor partition_ids_tensor = torch::from_blob(partitions_to_delete.data(),
                                                   {(int64_t) partitions_to_delete.size()}, torch::kInt64).clone();

    shared_ptr<Clustering> clustering = partition_manager_->select_partitions(partition_ids_tensor, true);

    std::cout << "Deleted " << partitions_to_delete.size() << " partitions." << std::endl;

    partition_manager_->delete_partitions(partition_ids_tensor, true);

    return partition_ids_tensor;
}

std::tuple<Tensor, Tensor, Tensor> QueryCostMaintenance::check_and_split_partitions() {
    if (partition_manager_->parent_ == nullptr) {
        return {};
    }

    int64_t n_partitions = partition_manager_->nlist();
    shared_ptr<PartitionState> state = get_partition_state(true);
    vector<float> split_deltas = estimate_split_delta(state);

    // get the partitions that exceed the threshold
    vector<int64_t> partitions_to_split;
    for (int i = 0; i < split_deltas.size(); i++) {
        int64_t partition_size = state->partition_sizes[i];
        if (split_deltas[i] < -split_threshold_ns_ && partition_size > 2 * min_partition_size_) {
            partitions_to_split.push_back(state->partition_ids[i]);
        }
    }

    // split the partitions
    Tensor partitions_to_split_tensor = torch::from_blob(partitions_to_split.data(),
                                                         {(int64_t) partitions_to_split.size()}, torch::kInt64);
    shared_ptr<Clustering> split_partitions = partition_manager_->split_partitions(
        partitions_to_split_tensor);

    vector<int64_t> removed_partitions;
    vector<int64_t> kept_splits;
    vector<Tensor> split_vectors;
    vector<Tensor> split_ids;

    // rejection mechanism
    for (int i = 0; i < partitions_to_split.size(); i++) {
        int64_t left_split_size = split_partitions->cluster_size(i * 2);
        int64_t right_split_size = split_partitions->cluster_size(i * 2 + 1);

        if (left_split_size > min_partition_size_ && right_split_size > min_partition_size_) {
            removed_partitions.push_back(partitions_to_split[i]);
            kept_splits.push_back(i * 2);
            kept_splits.push_back(i * 2 + 1);
            split_vectors.push_back(split_partitions->vectors[i * 2]);
            split_vectors.push_back(split_partitions->vectors[i * 2 + 1]);
            split_ids.push_back(split_partitions->vector_ids[i * 2]);
            split_ids.push_back(split_partitions->vector_ids[i * 2 + 1]);
        }
    }

    // print out how many splits were kept
    std::cout << "Kept " << kept_splits.size() << " splits out of " << 2 * partitions_to_split.size() << std::endl;

    if (kept_splits.size() == 0) {
        return {};
    }

    Tensor kept_splits_tensor = torch::from_blob(kept_splits.data(), {(int64_t) kept_splits.size()}, torch::kInt64);
    Tensor split_centroids = split_partitions->centroids.index_select(0, kept_splits_tensor);

    shared_ptr<Clustering> new_partitions = make_shared<Clustering>();
    new_partitions->centroids = split_centroids;
    for (int i = 0; i < kept_splits.size(); i++) {
        new_partitions->vectors.push_back(split_vectors[i]);
        new_partitions->vector_ids.push_back(split_ids[i]);
    }

    // Delete the old partitions and add the new ones
    Tensor removed_partitions_tensor = torch::from_blob(removed_partitions.data(),
                                                        {(int64_t) removed_partitions.size()}, torch::kInt64);

    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor old_centroids = partition_manager_->parent_->get(removed_partitions_tensor);
    partition_manager_->delete_partitions(removed_partitions_tensor);
    partition_manager_->add_partitions(new_partitions);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Delete and Add of partitions took " << std::chrono::duration_cast<
        std::chrono::milliseconds>(end_time - start_time).count() << " ms." << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    auto new_ids_accessor = new_partitions->partition_ids.accessor<int64_t, 1>();
    auto split_ids_accessor = removed_partitions_tensor.accessor<int64_t, 1>();
    for (int i = 0; i < removed_partitions_tensor.size(0); i++) {
        int64_t old_partition_id = split_ids_accessor[i];
        int64_t left_partition_id = new_ids_accessor[i * 2];
        int64_t right_partition_id = new_ids_accessor[i * 2 + 1];
        add_split(old_partition_id, left_partition_id, right_partition_id);
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Add splits took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).
            count() << " ms." << std::endl;

    return {new_partitions->partition_ids, old_centroids, removed_partitions_tensor};
}

void QueryCostMaintenance::local_refinement(Tensor partition_ids, int refinement_radius) {
    Tensor split_centroids = partition_manager_->parent_->get(partition_ids);

    auto search_params = std::make_shared<SearchParams>();
    search_params->nprobe = 1000;
    search_params->k = refinement_radius;

    auto result = partition_manager_->parent_->search(split_centroids, search_params);
    Tensor refine_ids = std::get<0>(torch::_unique(result->ids));

    // remove any -1
    refine_ids = refine_ids.masked_select(refine_ids != -1);
    refine_partitions(refine_ids, refinement_iterations_);
}

Tensor LireMaintenance::check_and_delete_partitions() {
    // check for partitions less than min_partition_size_
    if (partition_manager_->parent_ == nullptr) {
        return {};
    }

    Tensor partition_ids = partition_manager_->get_partition_ids();
    Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids);

    Tensor delete_ids = partition_ids.masked_select(partition_sizes < min_partition_size_);

    partition_manager_->delete_partitions(delete_ids);

    return delete_ids;
}

std::tuple<Tensor, Tensor, Tensor> LireMaintenance::check_and_split_partitions() {
    // check for partitions greater than 2 * min_partition_size_
    if (partition_manager_->parent_ == nullptr) {
        return {};
    }

    Tensor partition_ids = partition_manager_->get_partition_ids();
    Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids);

    int max_partition_size = (int) (max_partition_ratio_ * target_partition_size_);
    Tensor old_ids = partition_ids.masked_select(partition_sizes > max_partition_size);


    Tensor old_centroids = partition_manager_->parent_->get(old_ids);

    // perform the split
    shared_ptr<Clustering> split_partitions = partition_manager_->split_partitions(old_ids);
    partition_manager_->delete_partitions(old_ids);
    partition_manager_->add_partitions(split_partitions);
    Tensor new_ids = split_partitions->partition_ids;

    return {new_ids, old_centroids, old_ids};
}

shared_ptr<MaintenanceTimingInfo> DeDriftMaintenance::maintenance() {
    shared_ptr<MaintenanceTimingInfo> timing_info;

    auto total_start = std::chrono::high_resolution_clock::now();

    // recompute the centroids
    // recompute_centroids();

    // select the top small and top large partitions
    Tensor partition_ids = partition_manager_->get_partition_ids();
    Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids);

    // sort the partition size
    Tensor sort_args = partition_sizes.argsort(0, true);

    // select the top small and top large partitions
    Tensor small_partition_ids = partition_ids.index_select(0, sort_args.narrow(0, 0, k_small_));
    Tensor large_partition_ids = partition_ids.index_select(
        0, sort_args.narrow(0, partition_ids.size(0) - k_large_, k_large_));

    // run refinement on the small and large partitions
    Tensor all_partition_ids = torch::cat({small_partition_ids, large_partition_ids}, 0);
    refine_partitions(all_partition_ids, refinement_iterations_);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info->total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    return timing_info;
}
