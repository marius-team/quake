//
// Created by Jason on 10/7/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <maintenance_policies.h>
#include <geometry.h>
#include <list_scanning.h>
#include <quake_index.h>

vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> >
MaintenancePolicy::get_split_history() {
    if (debug_) {
        std::cout << "[MaintenancePolicy] get_split_history: Entered." << std::endl;
    }
    vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> > split_history;

    // iterate over all splits in the split records, record the hit count of the deleted partition and the two split partitions
    for (auto tup : split_records_) {
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

        if (debug_) {
            std::cout << "[MaintenancePolicy] get_split_history: Parent " << parent_id
                      << " (hits=" << parent_hits << ") split into ("
                      << left_id << " with hits=" << left_hits << ", "
                      << right_id << " with hits=" << right_hits << ")." << std::endl;
        }
        split_history.push_back(std::make_tuple(parent_id, parent_hits, left_id, left_hits, right_id, right_hits));
    }

    if (debug_) {
        std::cout << "[MaintenancePolicy] get_split_history: Returning " << split_history.size() << " records." << std::endl;
    }
    return split_history;
}

shared_ptr<PartitionState> MaintenancePolicy::get_partition_state(bool only_modified) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] get_partition_state: Entered with only_modified="
                  << (only_modified ? "true" : "false") << std::endl;
    }
    vector<int64_t> partition_ids;
    vector<int64_t> partition_sizes;
    vector<float> partition_hit_rate;

    if (only_modified) {
        partition_ids = vector<int64_t>(modified_partitions_.begin(), modified_partitions_.end());
    } else {
        Tensor p_ids = partition_manager_->partition_store_->get_partition_ids();
        auto p_ids_accessor = p_ids.accessor<int64_t, 1>();
        partition_ids = vector<int64_t>(p_ids.size(0));
        for (int i = 0; i < p_ids.size(0); i++) {
            partition_ids[i] = p_ids_accessor[i];
        }
    }

    // For each partition, get the size and compute the hit rate.
    for (int64_t partition_id : partition_ids) {
        int64_t partition_size = partition_manager_->partition_store_->list_size(partition_id);
        int64_t hits = per_partition_hits_[partition_id];
        int curr_window_size = std::max(std::min(window_size_, curr_query_id_), 1);
        float hit_rate = hits / (float)curr_window_size;
        partition_sizes.push_back(partition_size);
        partition_hit_rate.push_back(hit_rate);
        if (debug_) {
            std::cout << "[MaintenancePolicy] Partition " << partition_id << ": size=" << partition_size
                      << ", hits=" << hits << ", window=" << curr_window_size
                      << ", hit_rate=" << hit_rate << std::endl;
        }
    }

    shared_ptr<PartitionState> state = std::make_shared<PartitionState>();
    state->partition_ids = partition_ids;
    state->partition_sizes = partition_sizes;
    state->partition_hit_rate = partition_hit_rate;

    if (debug_) {
        std::cout << "[MaintenancePolicy] get_partition_state: Returning state with "
                  << partition_ids.size() << " partitions." << std::endl;
    }
    return state;
}

void MaintenancePolicy::set_partition_modified(int64_t partition_id) {
    modified_partitions_.insert(partition_id);
    if (debug_) {
        std::cout << "[MaintenancePolicy] set_partition_modified: Marked partition "
                  << partition_id << " as modified." << std::endl;
    }
}

void MaintenancePolicy::set_partition_unmodified(int64_t partition_id) {
    modified_partitions_.erase(partition_id);
    if (debug_) {
        std::cout << "[MaintenancePolicy] set_partition_unmodified: Unmarked partition "
                  << partition_id << " as modified." << std::endl;
    }
}

void MaintenancePolicy::decrement_hit_count(int64_t partition_id) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] decrement_hit_count: Called for partition "
                  << partition_id << std::endl;
    }
    // check if id is in the partition hits
    if (per_partition_hits_.find(partition_id) == per_partition_hits_.end()) {
        // find it in the removed partition hits
        if (ancestor_partition_hits_.find(partition_id) == ancestor_partition_hits_.end()) {
            throw std::runtime_error("Partition not found in decrement_hit_count");
        }
        if (ancestor_partition_hits_[partition_id] > 0) {
            int64_t left_partition_id, right_partition_id;
            std::tie(left_partition_id, right_partition_id) = split_records_[partition_id];
            if (debug_) {
                std::cout << "[MaintenancePolicy] decrement_hit_count: Partition " << partition_id
                          << " not in per_partition_hits_. Delegating decrement to children: "
                          << left_partition_id << ", " << right_partition_id << std::endl;
            }
            decrement_hit_count(left_partition_id);
            decrement_hit_count(right_partition_id);
            ancestor_partition_hits_[partition_id]--;
        }
    } else {
        if (per_partition_hits_[partition_id] > 0) {
            per_partition_hits_[partition_id]--;
            if (debug_) {
                std::cout << "[MaintenancePolicy] decrement_hit_count: Decremented partition "
                          << partition_id << " to " << per_partition_hits_[partition_id] << std::endl;
            }
        }
    }
}

void MaintenancePolicy::increment_hit_count(vector<int64_t> hit_partition_ids) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] increment_hit_count: Processing partitions:";
        for (auto id : hit_partition_ids) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
    }

    const int64_t total_vectors = partition_manager_->ntotal();
    if (total_vectors == 0) {
        throw std::runtime_error("Error: index_->ntotal() is zero in increment_hit_count.");
    }

    int vectors_scanned_new = 0;
    for (const auto &hit : hit_partition_ids) {
        per_partition_hits_[hit]++;
        int size = partition_manager_->partition_store_->list_size(hit);
        vectors_scanned_new += size;
        modified_partitions_.insert(hit);
        if (debug_) {
            std::cout << "[MaintenancePolicy] increment_hit_count: Partition " << hit
                      << " new hit count: " << per_partition_hits_[hit] << ", size: " << size << std::endl;
        }
    }
    float new_scan_fraction = static_cast<float>(vectors_scanned_new) / total_vectors;

    int current_query_index = curr_query_id_ % window_size_;
    if (curr_query_id_ >= window_size_) {
        const auto &oldest_hits = per_query_hits_[current_query_index];
        const auto &oldest_sizes = per_query_scanned_partitions_sizes_[current_query_index];

        if (debug_) {
            std::cout << "[MaintenancePolicy] increment_hit_count: Removing oldest query data at index "
                      << current_query_index << std::endl;
        }
        for (const auto &hit : oldest_hits) {
            decrement_hit_count(hit);
        }
        int vectors_scanned_old = 0;
        for (const auto &size : oldest_sizes) {
            vectors_scanned_old += size;
        }
        float oldest_scan_fraction = static_cast<float>(vectors_scanned_old) / total_vectors;
        running_sum_scan_fraction_ -= oldest_scan_fraction;
        if (debug_) {
            std::cout << "[MaintenancePolicy] increment_hit_count: Old scan fraction "
                      << oldest_scan_fraction << " removed." << std::endl;
        }
    }
    per_query_hits_[current_query_index] = hit_partition_ids;
    std::vector<int64_t> hits_sizes;
    hits_sizes.reserve(hit_partition_ids.size());
    for (const auto &hit : hit_partition_ids) {
        hits_sizes.push_back(partition_manager_->partition_store_->list_size(hit));
    }
    per_query_scanned_partitions_sizes_[current_query_index] = hits_sizes;
    running_sum_scan_fraction_ += new_scan_fraction;
    int current_window_size = std::min(static_cast<int>(curr_query_id_) + 1, window_size_);
    current_scan_fraction_ = running_sum_scan_fraction_ / static_cast<float>(current_window_size);
    if (current_scan_fraction_ == 0.0) {
        current_scan_fraction_ = 1.0;
    }
    if (debug_) {
        std::cout << "[MaintenancePolicy] increment_hit_count: Query " << curr_query_id_
                  << " added with scan fraction " << new_scan_fraction
                  << ", new current_scan_fraction: " << current_scan_fraction_ << std::endl;
    }
    curr_query_id_++;
}

vector<float> MaintenancePolicy::estimate_split_delta(shared_ptr<PartitionState> state) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_split_delta: Calculating split deltas." << std::endl;
    }
    vector<float> deltas;
    int n_partitions = partition_manager_->nlist();
    int k = 10;
    float alpha = alpha_;
    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions + 1, k) -
                            latency_estimator_->estimate_scan_latency(n_partitions, k));
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_split_delta: delta_overhead = " << delta_overhead << std::endl;
    }
    for (int i = 0; i < state->partition_ids.size(); i++) {
        int64_t partition_id = state->partition_ids[i];
        int64_t partition_size = state->partition_sizes[i];
        float hit_rate = state->partition_hit_rate[i];

        float old_cost = latency_estimator_->estimate_scan_latency(partition_size, k) * hit_rate;
        float new_cost = latency_estimator_->estimate_scan_latency(partition_size / 2, k) * hit_rate * (2 * alpha);
        float delta = delta_overhead + new_cost - old_cost;
        deltas.push_back(delta);
        if (debug_) {
            std::cout << "[MaintenancePolicy] estimate_split_delta: Partition " << partition_id
                      << " (size=" << partition_size << ", hit_rate=" << hit_rate
                      << ") => old_cost=" << old_cost << ", new_cost=" << new_cost
                      << ", delta=" << delta << std::endl;
        }
    }
    return deltas;
}

vector<float> MaintenancePolicy::estimate_delete_delta(shared_ptr<PartitionState> state) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_delete_delta: Calculating delete deltas." << std::endl;
    }
    vector<float> deltas;
    int n_partitions = partition_manager_->nlist();
    int k = 10;
    float alpha = alpha_;
    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions - 1, k) -
                            latency_estimator_->estimate_scan_latency(n_partitions, k));
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_delete_delta: delta_overhead = " << delta_overhead << std::endl;
    }
    for (int i = 0; i < state->partition_ids.size(); i++) {
        int64_t partition_id = state->partition_ids[i];
        int64_t partition_size = state->partition_sizes[i];
        float hit_rate = state->partition_hit_rate[i];
        float delta_reassign = current_scan_fraction_ * latency_estimator_->estimate_scan_latency(partition_size, k);
        float delta = delta_overhead + delta_reassign;
        deltas.push_back(delta);
        if (debug_) {
            std::cout << "[MaintenancePolicy] estimate_delete_delta: Partition " << partition_id
                      << " (size=" << partition_size << ", hit_rate=" << hit_rate
                      << ") => delta_reassign=" << delta_reassign << ", delta=" << delta << std::endl;
        }
    }
    return deltas;
}

float MaintenancePolicy::estimate_add_level_delta() {
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_add_level_delta: Returning 0.0" << std::endl;
    }
    return 0.0;
}

float MaintenancePolicy::estimate_remove_level_delta() {
    if (debug_) {
        std::cout << "[MaintenancePolicy] estimate_remove_level_delta: Returning 0.0" << std::endl;
    }
    return 0.0;
}

shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::maintenance() {
    if (debug_) {
        std::cout << "[MaintenancePolicy] maintenance: Starting maintenance." << std::endl;
    }
    shared_ptr<MaintenanceTimingInfo> timing_info = std::make_shared<MaintenanceTimingInfo>();

    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    Tensor delete_ids = check_and_delete_partitions();
    auto end = std::chrono::high_resolution_clock::now();
    timing_info->delete_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    timing_info->n_deletes = delete_ids.size(0);

    if (debug_) {
        std::cout << "[MaintenancePolicy] maintenance: Deleted " << timing_info->n_deletes
                  << " partitions." << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    Tensor split_ids;
    Tensor old_centroids;
    Tensor old_ids;
    std::tie(split_ids, old_centroids, old_ids) = check_and_split_partitions();
    end = std::chrono::high_resolution_clock::now();
    timing_info->split_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    timing_info->n_splits = old_ids.size(0);

    if (debug_) {
        std::cout << "[MaintenancePolicy] maintenance: " << timing_info->n_splits
                  << " splits detected." << std::endl;
    }

    start = std::chrono::high_resolution_clock::now();
    if (split_ids.size(0) > 0) {
        local_refinement(split_ids, refinement_radius_);
        if (debug_) {
            std::cout << "[MaintenancePolicy] maintenance: Performed local refinement on splits." << std::endl;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    timing_info->split_refine_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info->total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    if (debug_) {
        std::cout << "[MaintenancePolicy] maintenance: Completed in "
                  << timing_info->total_time_us << " microseconds." << std::endl;
    }
    return timing_info;
}

void MaintenancePolicy::add_split(int64_t old_partition_id, int64_t left_partition_id, int64_t right_partition_id) {
    int64_t num_queries = std::max(std::min(curr_query_id_, window_size_), 1);
    split_records_[old_partition_id] = std::make_pair(left_partition_id, right_partition_id);
    per_partition_hits_[left_partition_id] = per_partition_hits_[old_partition_id];
    per_partition_hits_[right_partition_id] = per_partition_hits_[old_partition_id];
    ancestor_partition_hits_[old_partition_id] = per_partition_hits_[old_partition_id];
    deleted_partition_hit_rate_[old_partition_id] = (float) per_partition_hits_[old_partition_id] / num_queries;
    per_partition_hits_.erase(old_partition_id);
    if (debug_) {
        std::cout << "[MaintenancePolicy] add_split: Partition " << old_partition_id
                  << " split into " << left_partition_id << " and " << right_partition_id
                  << " with original hits " << ancestor_partition_hits_[old_partition_id] << std::endl;
    }
}

void MaintenancePolicy::add_partition(int64_t partition_id, int64_t hits) {
    per_partition_hits_[partition_id] = hits;
    if (debug_) {
        std::cout << "[MaintenancePolicy] add_partition: Added partition " << partition_id
                  << " with hit count " << hits << std::endl;
    }
}

void MaintenancePolicy::remove_partition(int64_t partition_id) {
    int64_t num_queries = std::max(std::min(curr_query_id_, window_size_), 1);
    ancestor_partition_hits_[partition_id] = per_partition_hits_[partition_id];
    deleted_partition_hit_rate_[partition_id] = per_partition_hits_[partition_id] / num_queries;
    per_partition_hits_.erase(partition_id);
    if (debug_) {
        std::cout << "[MaintenancePolicy] remove_partition: Removed partition " << partition_id
                  << ". Final hit rate: " << deleted_partition_hit_rate_[partition_id] << std::endl;
    }
}

void MaintenancePolicy::refine_partitions(Tensor partition_ids, int refinement_iterations) {
    if (debug_) {
        std::cout << "[MaintenancePolicy] refine_partitions: Called with "
                  << (partition_ids.defined() ? std::to_string(partition_ids.numel()) : "all partitions")
                  << " and iterations=" << refinement_iterations << std::endl;
    }
    // TODO: Add detailed logging when refinement is implemented.
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

    std::string profile_filename = "latency_profile.csv";

    latency_estimator_ = std::make_shared<ListScanLatencyEstimator>(
        partition_manager_->d(),
        latency_grid_n_values_,
        latency_grid_k_values_,
        n_trials_,
        false,
        profile_filename);

    if (debug_) {
        std::cout << "[QueryCostMaintenance] Constructor: Initialized with window_size=" << window_size_
                  << ", refinement_radius=" << refinement_radius_
                  << ", min_partition_size=" << min_partition_size_
                  << ", alpha=" << alpha_ << std::endl;
    }
}

float QueryCostMaintenance::compute_alpha_for_window() {
    if (debug_) {
        std::cout << "[QueryCostMaintenance] compute_alpha_for_window: Computing alpha from split history." << std::endl;
    }
    if (split_records_.empty()) {
        if (debug_) {
            std::cout << "[QueryCostMaintenance] compute_alpha_for_window: No split records found." << std::endl;
        }
        return 0;
    }

    float total_alpha = 0.0;
    for (const auto &split: split_records_) {
        int64_t parent_id = std::get<0>(split);
        int64_t left_id = std::get<1>(split).first;
        int64_t right_id = std::get<1>(split).second;

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

        float parent_hit_rate = deleted_partition_hit_rate_[parent_id];
        float curr_alpha = (left_hit_rate + right_hit_rate) / (2 * parent_hit_rate);
        if (debug_) {
            std::cout << "[QueryCostMaintenance] compute_alpha_for_window: For split from partition " << parent_id
                      << ", left hit_rate=" << left_hit_rate << ", right hit_rate=" << right_hit_rate
                      << ", parent hit_rate=" << parent_hit_rate << ", alpha=" << curr_alpha << std::endl;
        }
        total_alpha += curr_alpha;
    }
    float computed_alpha = total_alpha / split_records_.size();
    if (debug_) {
        std::cout << "[QueryCostMaintenance] compute_alpha_for_window: Computed alpha = " << computed_alpha << std::endl;
    }
    return computed_alpha;
}

void QueryCostMaintenance::local_refinement(Tensor partition_ids, int refinement_radius) {
    if (debug_) {
        std::cout << "[QueryCostMaintenance] local_refinement: Refining partitions: " << std::to_string(partition_ids.numel())
                  << " with radius " << refinement_radius << std::endl;
    }
    Tensor split_centroids = partition_manager_->parent_->get(partition_ids);
    auto search_params = std::make_shared<SearchParams>();
    search_params->nprobe = 1000;
    search_params->k = refinement_radius;

    auto result = partition_manager_->parent_->search(split_centroids, search_params);
    Tensor refine_ids = std::get<0>(torch::_unique(result->ids));
    refine_ids = refine_ids.masked_select(refine_ids != -1);
    refine_partitions(refine_ids, refinement_iterations_);
    if (debug_) {
        std::cout << "[QueryCostMaintenance] local_refinement: Completed refinement." << std::endl;
    }
}

Tensor QueryCostMaintenance::check_and_delete_partitions() {
    if (debug_) {
        std::cout << "[QueryCostMaintenance] check_and_delete_partitions: Starting deletion check." << std::endl;
    }
    if (partition_manager_->parent_ == nullptr) {
        if (debug_) {
            std::cout << "[QueryCostMaintenance] check_and_delete_partitions: No parent index; skipping deletion." << std::endl;
        }
        return {};
    }
    int64_t n_partitions = partition_manager_->parent_->ntotal();
    int64_t num_queries = std::min(curr_query_id_, window_size_);
    shared_ptr<PartitionState> state = get_partition_state(false);
    vector<float> delete_delta = estimate_delete_delta(state);
    Tensor delete_delta_tensor =
            torch::from_blob(delete_delta.data(), {(int64_t) delete_delta.size()}, torch::kFloat32).clone();

    vector<int64_t> partitions_to_delete;
    vector<float> partition_to_delete_delta;
    vector<float> partition_to_delete_hit_rate;
    for (int i = 0; i < delete_delta.size(); i++) {
        if (delete_delta[i] < -delete_threshold_ns_) {
            int64_t curr_partition_id = state->partition_ids[i];
            partitions_to_delete.push_back(curr_partition_id);
            partition_to_delete_delta.push_back(delete_delta[i]);
            partition_to_delete_hit_rate.push_back(state->partition_hit_rate[i]);
            if (debug_) {
                std::cout << "[QueryCostMaintenance] check_and_delete_partitions: Marking partition " << curr_partition_id
                          << " for deletion with delta " << delete_delta[i]
                          << " and hit rate " << state->partition_hit_rate[i] << std::endl;
            }
            remove_partition(curr_partition_id);
        }
    }

    Tensor partition_ids_tensor = torch::from_blob(partitions_to_delete.data(),
                                                   {(int64_t) partitions_to_delete.size()}, torch::kInt64).clone();
    shared_ptr<Clustering> clustering = partition_manager_->select_partitions(partition_ids_tensor, true);
    partition_manager_->delete_partitions(partition_ids_tensor, true);
    return partition_ids_tensor;
}

std::tuple<Tensor, Tensor, Tensor> QueryCostMaintenance::check_and_split_partitions() {
    if (debug_) {
        std::cout << "[QueryCostMaintenance] check_and_split_partitions: Starting split check." << std::endl;
    }
    if (partition_manager_->parent_ == nullptr) {
        if (debug_) {
            std::cout << "[QueryCostMaintenance] check_and_split_partitions: No parent index; skipping splits." << std::endl;
        }
        return {};
    }
    int64_t n_partitions = partition_manager_->nlist();
    shared_ptr<PartitionState> state = get_partition_state(false);
    vector<float> split_deltas = estimate_split_delta(state);

    vector<int64_t> partitions_to_split;
    for (int i = 0; i < split_deltas.size(); i++) {
        int64_t partition_size = state->partition_sizes[i];
        if (split_deltas[i] < -split_threshold_ns_ && partition_size > 2 * min_partition_size_) {
            partitions_to_split.push_back(state->partition_ids[i]);
            if (debug_) {
                std::cout << "[QueryCostMaintenance] check_and_split_partitions: Partition "
                          << state->partition_ids[i] << " marked for splitting (size=" << partition_size
                          << ", delta=" << split_deltas[i] << ")." << std::endl;
            }
        }
    }
    Tensor partitions_to_split_tensor = torch::from_blob(partitions_to_split.data(),
                                                         {(int64_t) partitions_to_split.size()}, torch::kInt64);
    shared_ptr<Clustering> split_partitions = partition_manager_->split_partitions(partitions_to_split_tensor);

    Tensor old_centroids = partition_manager_->parent_->get(partitions_to_split_tensor);
    partition_manager_->delete_partitions(partitions_to_split_tensor, false);
    partition_manager_->add_partitions(split_partitions);
    Tensor new_ids = split_partitions->partition_ids;
    if (debug_) {
        std::cout << "[QueryCostMaintenance] check_and_split_partitions: Added " << new_ids.size(0)
                  << " new partitions." << std::endl;
    }
    auto new_ids_accessor = split_partitions->partition_ids.accessor<int64_t, 1>();
    auto split_ids_accessor = partitions_to_split_tensor.accessor<int64_t, 1>();
    for (int i = 0; i < partitions_to_split_tensor.size(0); i++) {
        int64_t old_partition_id = split_ids_accessor[i];
        int64_t left_partition_id = new_ids_accessor[i * 2];
        int64_t right_partition_id = new_ids_accessor[i * 2 + 1];
        add_split(old_partition_id, left_partition_id, right_partition_id);
    }

    return {split_partitions->partition_ids, old_centroids, partitions_to_split_tensor};
}

shared_ptr<MaintenanceTimingInfo> DeDriftMaintenance::maintenance() {
    if (debug_) {
        std::cout << "[DeDriftMaintenance] maintenance: Starting dedrift maintenance." << std::endl;
    }
    shared_ptr<MaintenanceTimingInfo> timing_info;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Perform dedrift maintenance tasks (e.g., reassign centroids)
    // For now, we just call local refinement on selected partitions.
    Tensor partition_ids = partition_manager_->get_partition_ids();
    Tensor partition_sizes = partition_manager_->get_partition_sizes(partition_ids);

    Tensor sort_args = partition_sizes.argsort(0, true);
    Tensor small_partition_ids = partition_ids.index_select(0, sort_args.narrow(0, 0, k_small_));
    Tensor large_partition_ids = partition_ids.index_select(0, sort_args.narrow(0, partition_ids.size(0) - k_large_, k_large_));
    Tensor all_partition_ids = torch::cat({small_partition_ids, large_partition_ids}, 0);
    refine_partitions(all_partition_ids, refinement_iterations_);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info->total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    if (debug_) {
        std::cout << "[DeDriftMaintenance] maintenance: Completed dedrift maintenance in "
                  << timing_info->total_time_us << " us." << std::endl;
    }
    return timing_info;
}