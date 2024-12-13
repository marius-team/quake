//
// Created by Jason on 10/7/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <maintenance_policies.h>
#include <dynamic_ivf.h>
#include <geometry.h>
#include <list_scanning.h>


void ListScanLatencyEstimator::profile_scan_latency() {
    // Generate random vectors
    int max_n = n_values_.back();
    torch::Tensor vectors = torch::rand({max_n, d_});
    torch::Tensor ids = torch::randperm(max_n);
    torch::Tensor query = torch::rand({d_});

    for (size_t i = 0; i < n_values_.size(); ++i) {
        for (size_t j = 0; j < k_values_.size(); ++j) {
            int n = n_values_[i];
            int k = k_values_[j];

            torch::Tensor curr_vectors = vectors.narrow(0, 0, n);
            torch::Tensor curr_ids = ids.narrow(0, 0, n);
            TopkBuffer topk_buffer(k, false);

            const float* query_ptr = query.data_ptr<float>();
            const float* curr_vectors_ptr = curr_vectors.data_ptr<float>();
            const int64_t* curr_ids_ptr = curr_ids.data_ptr<int64_t>();

            uint64_t total_latency_ns = 0;
            for (int m = 0; m < n_trials_; ++m) {
                auto start = std::chrono::high_resolution_clock::now();
                scan_list(query_ptr, curr_vectors_ptr, curr_ids_ptr, n, d_, topk_buffer);
                auto end = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                total_latency_ns += duration.count();
            }
            double mean_latency_ns = static_cast<double>(total_latency_ns) / n_trials_;
            scan_latency_model_[i][j] = static_cast<float>(mean_latency_ns);
        }
    }
}

bool ListScanLatencyEstimator::get_interpolation_info(const std::vector<int>& values, int target, int& lower, int& upper, float& frac) const {
    if (target < values.front() || target > values.back()) {
        return false; // Out of bounds
    }

    // Find the first element greater than target
    auto it = std::upper_bound(values.begin(), values.end(), target);
    if (it == values.end()) {
        lower = values.size() - 2;
        upper = values.size() - 1;
    } else {
        lower = std::distance(values.begin(), it) - 1;
        upper = lower + 1;
    }

    int lower_val = values[lower];
    int upper_val = values[upper];
    if (upper_val == lower_val) {
        frac = 0.0f;
    } else {
        frac = static_cast<float>(target - lower_val) / (upper_val - lower_val);
    }

    return true;
}

float ListScanLatencyEstimator::estimate_scan_latency(int n, int k) const {
    if (n == 0 || k == 0) {
        return 0.0f;
    }
    // Check if n or k is below the minimum values
    if (n < n_values_.front() || k < k_values_.front()) {
        std::cout << "n=" << n << ", k=" << k << std::endl;
        throw std::out_of_range("n or k is below the minimum supported values.");
    }

    // Determine if n and k are within the grid
    bool n_within = (n <= n_values_.back());
    bool k_within = (k <= k_values_.back());

    // Variables to hold interpolation indices and fractions for n
    size_t i_lower, i_upper;
    float t; // Fraction for n

    if (n_within) {
        // Find the lower index for n
        auto it = std::upper_bound(n_values_.begin(), n_values_.end(), n);
        if (it == n_values_.end()) {
            // n is exactly the last value
            i_lower = n_values_.size() - 2;
            i_upper = n_values_.size() - 1;
            t = 1.0f;
        } else {
            i_upper = std::distance(n_values_.begin(), it);
            i_lower = i_upper - 1;
            int n1 = n_values_[i_lower];
            int n2 = n_values_[i_upper];
            t = static_cast<float>(n - n1) / static_cast<float>(n2 - n1);
        }
    } else {
        // Extrapolate in n using the last two n_values_
        i_lower = n_values_.size() - 2;
        i_upper = n_values_.size() - 1;
        int n1 = n_values_[i_lower];
        int n2 = n_values_[i_upper];
        t = static_cast<float>(n - n2) / static_cast<float>(n2 - n1); // t > 1
    }

    // Variables to hold interpolation indices and fractions for k
    size_t j_lower, j_upper;
    float u; // Fraction for k

    if (k_within) {
        // Find the lower index for k
        auto it = std::upper_bound(k_values_.begin(), k_values_.end(), k);
        if (it == k_values_.end()) {
            // k is exactly the last value
            j_lower = k_values_.size() - 2;
            j_upper = k_values_.size() - 1;
            u = 1.0f;
        } else {
            j_upper = std::distance(k_values_.begin(), it);
            j_lower = j_upper - 1;
            int k1 = k_values_[j_lower];
            int k2 = k_values_[j_upper];
            u = static_cast<float>(k - k1) / static_cast<float>(k2 - k1);
        }
    } else {
        // Extrapolate in k using the last two k_values_
        j_lower = k_values_.size() - 2;
        j_upper = k_values_.size() - 1;
        int k1 = k_values_[j_lower];
        int k2 = k_values_[j_upper];
        u = static_cast<float>(k - k2) / static_cast<float>(k2 - k1); // u > 1
    }

    // Case 1: Both n and k are within the grid (Bilinear Interpolation)
    if (n_within && k_within) {
        float f11 = scan_latency_model_[i_lower][j_lower];
        float f12 = scan_latency_model_[i_lower][j_upper];
        float f21 = scan_latency_model_[i_upper][j_lower];
        float f22 = scan_latency_model_[i_upper][j_upper];

        // Bilinear interpolation formula
        float interpolated_latency = (1 - t) * (1 - u) * f11 +
                                      t * (1 - u) * f21 +
                                      (1 - t) * u * f12 +
                                      t * u * f22;
        return interpolated_latency;
    }

    // Helper lambda to perform linear extrapolation
    auto linear_extrapolate = [&](float f1, float f2, float fraction) -> float {
        float slope = f2 - f1;
        return f2 + slope * fraction;
    };

    // Case 2: Extrapolate in n while k is within the grid
    if (!n_within && k_within) {
        // Extrapolate latency at j_lower
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_upper][j_lower];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, t);

        // Extrapolate latency at j_upper
        float f3 = scan_latency_model_[i_lower][j_upper];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, t);

        // Now interpolate between extrapolated_f_lower and extrapolated_f_upper based on u
        float interpolated_latency = (1 - u) * extrapolated_f_lower + u * extrapolated_f_upper;
        return interpolated_latency;
    }

    // Case 3: Extrapolate in k while n is within the grid
    if (n_within && !k_within) {
        // Extrapolate latency at i_lower
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_lower][j_upper];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, u);

        // Extrapolate latency at i_upper
        float f3 = scan_latency_model_[i_upper][j_lower];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, u);

        // Now interpolate between extrapolated_f_lower and extrapolated_f_upper based on t
        float interpolated_latency = (1 - t) * extrapolated_f_lower + t * extrapolated_f_upper;
        return interpolated_latency;
    }

    // Case 4: Extrapolate in both n and k
    if (!n_within && !k_within) {
        // Extrapolate latency at j_lower
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_upper][j_lower];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, t);

        // Extrapolate latency at j_upper
        float f3 = scan_latency_model_[i_lower][j_upper];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, t);

        // Now extrapolate across k using u
        float extrapolated_latency = linear_extrapolate(extrapolated_f_lower, extrapolated_f_upper, u);
        return extrapolated_latency;
    }

    // If none of the above cases match, throw an error
    throw std::runtime_error("Unable to estimate scan latency.");
}


MaintenanceAuditInfo MaintenancePolicy::audit() {
    MaintenanceAuditInfo audit_info;
    audit_info.per_partition_hits = per_partition_hits_;
    audit_info.per_query_hits = per_query_hits_;
    audit_info.per_query_scanned_partitions_sizes = per_query_scanned_partitions_sizes_;
    audit_info.running_sum_scan_fraction = running_sum_scan_fraction_;
    audit_info.current_scan_fraction = current_scan_fraction_;
    audit_info.deleted_partition_hit_rate = deleted_partition_hit_rate_;
    audit_info.ancestor_partition_hits = ancestor_partition_hits_;
    audit_info.split_records = split_records_;

    // number of queries in the window
    int64_t n_queries = std::min(curr_query_id_, window_size_);

    // compute the average hit rate
    audit_info.average_hit_rate = 0.0;
    for (const auto& partition_hit : per_partition_hits_) {
        audit_info.average_hit_rate += (partition_hit.second / (float) n_queries);
    }
    audit_info.average_hit_rate /= (float) per_partition_hits_.size();

    return audit_info;
}


MaintenanceTimingInfo MaintenancePolicy::maintenance() {

    MaintenanceAuditInfo before = audit();

    MaintenanceTimingInfo timing_info;
    auto total_start = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::high_resolution_clock::now();
    Tensor delete_ids;
    Tensor delete_centroids;
    std::tie(delete_ids, delete_centroids) = check_and_delete_partitions();
    auto end = std::chrono::high_resolution_clock::now();
    timing_info.delete_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    timing_info.n_deletes = delete_ids.size(0);

    start = std::chrono::high_resolution_clock::now();
    if (delete_ids.size(0) > 0) {
        refine_delete(delete_centroids);
    }
    end = std::chrono::high_resolution_clock::now();
    timing_info.delete_refine_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    Tensor split_ids;
    Tensor old_centroids;
    Tensor old_ids;
    std::tie(split_ids, old_centroids, old_ids) = check_and_split_partitions();
    end = std::chrono::high_resolution_clock::now();
    timing_info.split_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    timing_info.n_splits = split_ids.size(0);

    start = std::chrono::high_resolution_clock::now();
    if (split_ids.size(0) > 0) {
        refine_split(split_ids, old_centroids);
    }
    end = std::chrono::high_resolution_clock::now();
    timing_info.split_refine_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    MaintenanceAuditInfo after = audit();

    after.split_partition_ids = old_ids;
    after.new_partition_ids = split_ids;
    after.delete_partition_ids = delete_ids;

    timing_info.audit_before = before;
    timing_info.audit_after = after;

    // print out how many splits and deletes
    std::cout << "Number of splits: " << timing_info.n_splits << std::endl;
    std::cout << "Number of deletes: " << timing_info.n_deletes << std::endl;
    std::cout << "Total time (us): " << timing_info.total_time_us << std::endl;

    return timing_info;
}

void MaintenancePolicy::set_params(MaintenancePolicyParams params) {
    window_size_ = params.window_size;
    refinement_radius_ = params.refinement_radius;
    refinement_iterations_ = params.refinement_iterations;
    min_partition_size_ = params.min_partition_size;
    alpha_ = params.alpha;
    enable_split_rejection_ = params.enable_split_rejection;
    enable_delete_rejection_ = params.enable_delete_rejection;

    split_threshold_ns_ = params.split_threshold_ns;
    delete_threshold_ns_ = params.delete_threshold_ns;

    // reset the query_hits_
    per_query_hits_ = vector<vector<int64_t> >(window_size_);
    per_query_scanned_partitions_sizes_ = vector<vector<int64_t> >(window_size_);
}

vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> MaintenancePolicy::get_split_history() {

    vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> split_history;

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

        split_history.push_back(std::make_tuple(parent_id, parent_hits, left_id, left_hits, right_id, right_hits));
    }

    return split_history;
}


void MaintenancePolicy::decrement_hit_count(int64_t partition_id) {
    // check if id is in the partition hits
    if (per_partition_hits_.find(partition_id) == per_partition_hits_.end()) {
        // find it in the removed partition hits
        if (ancestor_partition_hits_.find(partition_id) == ancestor_partition_hits_.end()) {
            // return;
            // std::cout << "Partition id: " << partition_id << std::endl;
            //
            // // print out partition ids
            // std::cout << index_->get_partition_ids() << std::endl;
            //
            // // print out all removed partition hits
            // for (const auto& removed_partition_hit : removed_partition_hits_) {
            //     std::cout << removed_partition_hit.first << std::endl;
            // }
            //
            throw std::runtime_error("Partition not found");
            return;
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


void MaintenancePolicy::update_hits(vector<int64_t> hits) {

    // // for debugging get all the hits before updating
    // vector<std::pair<int64_t, int64_t>> hits_before_update;
    // for (const auto& partition_hit : per_partition_hits_) {
    //     hits_before_update.push_back(std::make_pair(partition_hit.first, partition_hit.second));
    // }
    // // sort the hits by the partition id
    // std::sort(hits_before_update.begin(), hits_before_update.end(), [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
    //     return a.first < b.first;
    // });

    // Ensure ntotal is not zero to avoid division by zero
    const int64_t total_vectors = index_->ntotal();
    if (total_vectors == 0) {
        std::cerr << "Error: index_->ntotal() is zero." << std::endl;
        return;
    }

    // Calculate the scan fraction for the new query
    int vectors_scanned_new = 0;
    for (const auto &hit: hits) {
        per_partition_hits_[hit]++;
        vectors_scanned_new += index_->get_invlists()->list_size(hit);
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
    per_query_hits_[current_query_index] = hits;

    // Calculate and store the sizes for the new hits
    std::vector<int64_t> hits_sizes;
    hits_sizes.reserve(hits.size());
    for (const auto &hit: hits) {
        hits_sizes.push_back(index_->get_invlists()->list_size(hit));
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

    // Debug output
    // std::cout << "Query scan fraction: " << new_scan_fraction << std::endl;
    // std::cout << "Current running scan fraction: " << current_scan_fraction_ << std::endl;

    // Increment the query ID for the next update
    curr_query_id_++;
}

void MaintenancePolicy::add_split(int64_t old_partition_id, int64_t left_partition_id, int64_t right_partition_id) {

    int64_t num_queries = std::min(curr_query_id_, window_size_);
    split_records_[old_partition_id] = std::make_pair(left_partition_id, right_partition_id);
    per_partition_hits_[left_partition_id] = per_partition_hits_[old_partition_id];
    per_partition_hits_[right_partition_id] = per_partition_hits_[old_partition_id];
    ancestor_partition_hits_[old_partition_id] = per_partition_hits_[old_partition_id];
    deleted_partition_hit_rate_[old_partition_id] = (float) per_partition_hits_[old_partition_id] / num_queries;
    per_partition_hits_.erase(old_partition_id);
}

void MaintenancePolicy::add_partition(int64_t partition_id, int64_t hits) {
    per_partition_hits_[partition_id] = hits;
}

void MaintenancePolicy::remove_partition(int64_t partition_id) {
    int64_t num_queries = std::min(curr_query_id_, window_size_);
    ancestor_partition_hits_[partition_id] = per_partition_hits_[partition_id];
    deleted_partition_hit_rate_[partition_id] = per_partition_hits_[partition_id] / num_queries;
    per_partition_hits_.erase(partition_id);

}

QueryCostMaintenance::QueryCostMaintenance(std::shared_ptr<DynamicIVF_C> index, MaintenancePolicyParams params) {
    maintenance_policy_name_ = "query_cost";
    window_size_ = params.window_size;
    refinement_radius_ = params.refinement_radius;
    min_partition_size_ = params.min_partition_size;
    alpha_ = params.alpha;
    enable_split_rejection_ = params.enable_split_rejection;
    enable_delete_rejection_ = params.enable_delete_rejection;

    curr_query_id_ = 0;
    per_query_hits_ = vector<vector<int64_t> >(window_size_);
    per_query_scanned_partitions_sizes_ = vector<vector<int64_t> >(window_size_);
    index_ = index;
    current_scan_fraction_ = 1.0;

    // initial the latency estimator
    latency_estimator_ = std::make_shared<ListScanLatencyEstimator>(index_->d_, latency_grid_n_values_, latency_grid_k_values_, n_trials_);
    latency_estimator_->profile_scan_latency();
}

float QueryCostMaintenance::compute_alpha_for_window() {
    // go over the split history and compute the alpha
    // alpha is the fraction of decrease in hit rate resulting from a split
    // go over the split history and compute the alpha

    if (split_records_.empty()) {
        return 0;
    }

    float total_alpha = 0.0;
    for (const auto& split : split_records_) {
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


std::pair<Tensor, Tensor> QueryCostMaintenance::check_and_delete_partitions() {
    if (index_->parent_ == nullptr) {
        return {};
    }

    int64_t n_partitions = index_->parent_->ntotal();
    Tensor partition_ids = index_->parent_->get_ids().contiguous();
    float parent_scan_fraction = index_->parent_->maintenance_policy_->current_scan_fraction_;
    auto partition_ids_accessor = partition_ids.data_ptr<int64_t>();

    vector<float> per_partition_hit_rate(partition_ids.size(0));
    vector<uint32_t> per_partition_size(partition_ids.size(0));

    int64_t num_queries = std::min(curr_query_id_, window_size_);
    for (int i = 0; i < n_partitions; i++) {
        int64_t partition_id = partition_ids_accessor[i];
        per_partition_size[i] = index_->get_invlists()->list_size(partition_id);
        per_partition_hit_rate[i] = (float) per_partition_hits_[partition_id] / num_queries;
    }

    int64_t total_nprobe = 0;
    for (int i = 0; i < per_query_hits_.size(); i++) {
        total_nprobe += per_query_hits_[i].size();
    }
    float mean_nprobe = (float) total_nprobe / num_queries;

    // compute the delta in overhead
    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions - 1, 10) - latency_estimator_->estimate_scan_latency(n_partitions, 10));
    vector<float> delete_delta(partition_ids.size(0));
    double total_delta = 0.0;
    int mean_partition_size = index_->ntotal() / n_partitions;

    for (int i = 0; i < partition_ids.size(0); i++) {

        // assume each vector is assigned to a different partition
        float delta_reassign = current_scan_fraction_ * (latency_estimator_->estimate_scan_latency(2 * per_partition_size[i], 10) - latency_estimator_->estimate_scan_latency(per_partition_size[i], 10));
        float delta_delete = -per_partition_hit_rate[i] * (latency_estimator_->estimate_scan_latency(per_partition_size[i], 10));

        // assume that the queries that scan this partition will scan all neighboring partitions
        // since each vector is assigned to a different partition, the number of neighboring partitions is the number of vectors in the partition
        // float delta_delete_increase = per_partition_hit_rate[i] * (latency_estimator_->estimate_scan_latency(per_partition_size[i], 10));
        delete_delta[i] = delta_overhead + delta_reassign + delta_delete;
        total_delta += delete_delta[i];
    }
    Tensor delete_delta_tensor =
            torch::from_blob(delete_delta.data(), {(int64_t) delete_delta.size()}, torch::kFloat32).clone();

    vector<int64_t> partitions_to_delete;
    vector<float> partition_to_delete_delta;
    vector<float> partition_to_delete_hit_rate;
    for (int i = 0; i < partition_ids.size(0); i++) {
        if (delete_delta[i] < -delete_threshold_ns_) {
            int64_t curr_partition_id = partition_ids_accessor[i];
            partitions_to_delete.push_back(curr_partition_id);
            partition_to_delete_delta.push_back(delete_delta[i]);
            partition_to_delete_hit_rate.push_back(per_partition_hit_rate[i]);
            remove_partition(curr_partition_id);
        }
    }

    // delete the partitions
    vector<int64_t> deleted_partition_ids;
    Tensor centroids;
    vector<Tensor> centroid_list;
    vector<Tensor> cluster_vectors;
    vector<Tensor> cluster_ids;
    Tensor partition_ids_tensor = torch::from_blob(partitions_to_delete.data(),
                                                   {(int64_t) partitions_to_delete.size()}, torch::kInt64).clone();

    std::tie(centroids, cluster_vectors, cluster_ids) = index_->select_clusters(partition_ids_tensor, /*copy=*/true);

    for (int i = 0; i < partitions_to_delete.size(); i++) {
        int64_t partition_id = partitions_to_delete[i];

        // get reassignments
        Tensor p_id = torch::tensor({partition_id}, torch::kInt64);
        Tensor centroid = index_->parent_->select_vectors(p_id);
        Tensor reassignments;
        std::tie(reassignments, std::ignore, std::ignore) = index_->parent_->search(cluster_vectors[i], 1, 2);

        // take the first nearest neighbor as the reassignment, if the nearest neighbor is the same as the current partition, take the second nearest neighbor
        Tensor reassignments_mask = reassignments.select(1, 0) == p_id;
        Tensor curr_reassignments = reassignments.select(1, 0);
        Tensor second_reassignments = reassignments.select(1, 1);
        reassignments = torch::where(reassignments_mask, second_reassignments, curr_reassignments);

        Tensor unique_reassignments;
        Tensor unique_reassignments_counts;
        std::tie(unique_reassignments, std::ignore,unique_reassignments_counts) = torch::_unique2(reassignments, /* sorted */ true, false, true);
        auto unique_reassignments_accessor = unique_reassignments.accessor<int64_t, 1>();
        auto unique_reassignments_counts_accessor = unique_reassignments_counts.accessor<int64_t, 1>();

        // recompute cost
        float curr_delta = delta_overhead;
        float delta_reassign = 0;
        float delta_delete_increase = 0;

        float curr_partition_hit_rate = per_partition_hit_rate[i];

        for (int j = 0; j < unique_reassignments.size(0); j++) {
            int64_t reassign_partition_id = unique_reassignments_accessor[j];
            int64_t reassign_partition_size = index_->get_invlists()->list_size(reassign_partition_id);

            float reassign_partition_hit_rate = (float) per_partition_hits_[reassign_partition_id] / num_queries;

            // assume uniform distribution of queries

            if (num_queries == 0) {
                reassign_partition_hit_rate = 1.0 / n_partitions;
                curr_partition_hit_rate = 1.0 / n_partitions;
            }

            int64_t reassign_count_delta = unique_reassignments_counts_accessor[j];
            delta_reassign += reassign_partition_hit_rate * (latency_estimator_->estimate_scan_latency(reassign_partition_size + reassign_count_delta, 10) - latency_estimator_->estimate_scan_latency(reassign_partition_size, 10));
            delta_delete_increase += curr_partition_hit_rate * (latency_estimator_->estimate_scan_latency(reassign_partition_size + reassign_count_delta, 10));
        }
        // delta_delete_increase = 5 * per_partition_hit_rate[i] * (latency_estimator_->estimate_scan_latency(per_partition_size[i], 10));

        curr_delta += delta_reassign + delta_delete_increase;

        // check for nans

        if (curr_delta < -delete_threshold_ns_) {
            // remove it
            centroid_list.push_back(centroid);
            remove_partition(partition_id);
            Tensor p_ids = torch::tensor({partition_id}, torch::kInt64);
            index_->delete_partitions(p_ids, true, reassignments);
            deleted_partition_ids.push_back(partition_id);
        } else {

        }

    }
    Tensor partitions_to_delete_tensor = torch::from_blob(deleted_partition_ids.data(),
                                                          {(int64_t) deleted_partition_ids.size()}, torch::kInt64).clone();
    if (deleted_partition_ids.size() == 0) {
        centroids = torch::empty({0, index_->d_}, torch::kFloat32);
    } else {
        centroids = torch::cat(centroid_list, 0);
    }

    // index_->delete_partitions(partitions_to_delete_tensor, true);
    return {partitions_to_delete_tensor, centroids};
}

std::tuple<Tensor, Tensor, Tensor> QueryCostMaintenance::check_and_split_partitions() {
    if (index_->parent_ == nullptr) {
        return {};
    }
    int64_t n_partitions = index_->parent_->ntotal();
    Tensor partition_ids = index_->parent_->get_ids();
    auto partition_ids_accessor = partition_ids.accessor<int64_t, 1>();
    float parent_scan_fraction = index_->parent_->maintenance_policy_->current_scan_fraction_;

    vector<float> per_partition_hit_rate(partition_ids.size(0));
    vector<uint32_t> per_partition_size(partition_ids.size(0));

    int64_t num_queries = std::min(curr_query_id_, window_size_);
    for (int i = 0; i < n_partitions; i++) {
        int64_t partition_id = partition_ids_accessor[i];
        per_partition_size[i] = index_->get_invlists()->list_size(partition_id);
        per_partition_hit_rate[i] = (float) per_partition_hits_[partition_id] / num_queries;
        if (per_partition_hit_rate[i] > 1) {
            throw std::runtime_error("Hit rate is greater than 1");
        }
    }

    float delta_overhead = (latency_estimator_->estimate_scan_latency(n_partitions + 1, 10) - latency_estimator_->estimate_scan_latency(n_partitions, 10));

    float alpha = alpha_;
    // float alpha = compute_alpha_for_window();
    // if (alpha == 0) {
    //     alpha = alpha_;
    // }
    // alpha = std::min(alpha_, alpha);
    // alpha = std::max(alpha, (float) 0.5);


    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Compute alpha: " << compute_alpha_for_window() << std::endl;

    vector<float> split_delta(partition_ids.size(0));
    for (int i = 0; i < partition_ids.size(0); i++) {
        split_delta[i] = delta_overhead + (latency_estimator_->estimate_scan_latency(per_partition_size[i], 10) / 2 * per_partition_hit_rate[i] * (2 * alpha)) - (latency_estimator_->estimate_scan_latency(per_partition_size[i], 10) * per_partition_hit_rate[i]);
    }

    vector<int64_t> partitions_to_split;
    for (int i = 0; i < partition_ids.size(0); i++) {
        int64_t partition_size = per_partition_size[i];

        if (split_delta[i] < -split_threshold_ns_ && partition_size > 2 * min_partition_size_) {
            partitions_to_split.push_back(partition_ids_accessor[i]);
        }
    }

    // split the partitions
    Tensor partitions_to_split_tensor = torch::from_blob(partitions_to_split.data(),
                                                         {(int64_t) partitions_to_split.size()}, torch::kInt64);
    std::tuple<Tensor, vector<Tensor>, vector<Tensor> > split_partitions = index_->split_partitions(
        partitions_to_split_tensor);

    vector<int64_t> removed_partitions;
    vector<int64_t> kept_splits;
    vector<Tensor> split_vectors;
    vector<Tensor> split_ids;

    // rejection mechanism
    for (int i = 0; i < partitions_to_split.size(); i++) {
        int64_t left_split_size = std::get<1>(split_partitions)[i * 2].size(0);
        int64_t right_split_size = std::get<1>(split_partitions)[i * 2 + 1].size(0);

        if (left_split_size > min_partition_size_ && right_split_size > min_partition_size_) {
            removed_partitions.push_back(partitions_to_split[i]);
            kept_splits.push_back(i * 2);
            kept_splits.push_back(i * 2 + 1);
            split_vectors.push_back(std::get<1>(split_partitions)[i * 2]);
            split_vectors.push_back(std::get<1>(split_partitions)[i * 2 + 1]);
            split_ids.push_back(std::get<2>(split_partitions)[i * 2]);
            split_ids.push_back(std::get<2>(split_partitions)[i * 2 + 1]);
        }
    }

    // print out how many splits were kept
    std::cout << "Kept " << kept_splits.size() << " splits out of " << 2 * partitions_to_split.size() << std::endl;

    if (kept_splits.size() == 0) {
        return {};
    }

    Tensor kept_splits_tensor = torch::from_blob(kept_splits.data(), {(int64_t) kept_splits.size()}, torch::kInt64);
    Tensor split_centroids = std::get<0>(split_partitions).index_select(0, kept_splits_tensor);

    // Delete the old partitions and add the new ones
    Tensor removed_partitions_tensor = torch::from_blob(removed_partitions.data(),
                                                        {(int64_t) removed_partitions.size()}, torch::kInt64);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    Tensor old_centroids = index_->parent_->select_vectors(removed_partitions_tensor);    
    index_->delete_partitions(removed_partitions_tensor);
    Tensor new_ids = index_->add_partitions(split_centroids, split_vectors, split_ids);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Delete and Add of partitions took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms." << std::endl;    

    start_time = std::chrono::high_resolution_clock::now();
    auto new_ids_accessor = new_ids.accessor<int64_t, 1>();
    auto split_ids_accessor = removed_partitions_tensor.accessor<int64_t, 1>();
    for (int i = 0; i < removed_partitions_tensor.size(0); i++) {
        int64_t old_partition_id = split_ids_accessor[i];
        int64_t left_partition_id = new_ids_accessor[i * 2];
        int64_t right_partition_id = new_ids_accessor[i * 2 + 1];
        add_split(old_partition_id, left_partition_id, right_partition_id);
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Add splits took " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << " ms." << std::endl;

    return {new_ids, old_centroids, removed_partitions_tensor};
}

void QueryCostMaintenance::refine_delete(Tensor old_centroids) {

    vector<Tensor> refine_ids_list;
    for (int i = 0; i < old_centroids.size(0); i++) {
        Tensor old_centroid = old_centroids[i];
        auto result = index_->parent_->search(old_centroid, 1000, refinement_radius_);
        Tensor curr_neighbors = std::get<0>(result);
        curr_neighbors = curr_neighbors.masked_select(curr_neighbors != -1);
        refine_ids_list.emplace_back(curr_neighbors);
    }

    Tensor refine_ids = std::get<0>(torch::_unique(torch::cat(refine_ids_list, 0)));
    refine_ids = refine_ids.masked_select(refine_ids != -1);
    index_->refine_clusters(refine_ids);
}

void QueryCostMaintenance::refine_split(Tensor partition_ids, Tensor old_centroids) {
    // refinement
    Tensor split_centroids = index_->parent_->select_vectors(partition_ids);
    int d = split_centroids.size(1);


    // for each split get the neighboring partitions
    // first get top n neighbors by distance
    // then filter using a geometric test for overlap

    vector<Tensor> refine_ids_list;
    for (int i = 0; i < split_centroids.size(0); i++) {

        Tensor curr_centroid = split_centroids[i];
        Tensor old_centroid = old_centroids[i / 2];

        auto result = index_->parent_->search(old_centroid, 1000, refinement_radius_);
        Tensor curr_neighbors = std::get<0>(result);
        curr_neighbors = curr_neighbors.masked_select(curr_neighbors != -1);
        Tensor neighbor_centroids = index_->parent_->select_vectors(curr_neighbors);

        // Tensor overlap_ratio = estimate_overlap(curr_centroid, old_centroid, neighbor_centroids);
        // Tensor filter_mask = overlap_ratio > .15;
        // print count nonzero for debugging

        refine_ids_list.emplace_back(curr_neighbors);
    }
    refine_ids_list.emplace_back(partition_ids);


    // kNN search to get neighboring partitions
    Tensor refine_ids = std::get<0>(torch::_unique(torch::cat(refine_ids_list, 0)));

    // remove any -1
    refine_ids = refine_ids.masked_select(refine_ids != -1);

    index_->refine_clusters(refine_ids);
}

void LireMaintenance::refine_split(Tensor partition_ids, Tensor old_centroids) {
    // refinement
    Tensor split_centroids = index_->parent_->select_vectors(partition_ids);
    int d = split_centroids.size(1);


    // for each split get the neighboring partitions
    // first get top n neighbors by distance
    // then filter using a geometric test for overlap

    vector<Tensor> refine_ids_list;
    for (int i = 0; i < split_centroids.size(0); i++) {

        Tensor curr_centroid = split_centroids[i];
        Tensor old_centroid = old_centroids[i / 2];

        auto result = index_->parent_->search(old_centroid, 1000, refinement_radius_);
        Tensor curr_neighbors = std::get<0>(result);
        curr_neighbors = curr_neighbors.masked_select(curr_neighbors != -1);
        Tensor neighbor_centroids = index_->parent_->select_vectors(curr_neighbors);

        // Tensor overlap_ratio = estimate_overlap(curr_centroid, old_centroid, neighbor_centroids);
        // Tensor filter_mask = overlap_ratio > .15;
        // print count nonzero for debugging

        refine_ids_list.emplace_back(curr_neighbors);
    }
    refine_ids_list.emplace_back(partition_ids);


    // kNN search to get neighboring partitions
    Tensor refine_ids = std::get<0>(torch::_unique(torch::cat(refine_ids_list, 0)));

    // remove any -1
    refine_ids = refine_ids.masked_select(refine_ids != -1);

    index_->refine_clusters(refine_ids, 0);
}

std::pair<Tensor, Tensor> LireMaintenance::check_and_delete_partitions() {
    // check for partitions less than min_partition_size_
    if (index_->parent_ == nullptr) {
        return {};
    }

    Tensor partition_sizes = index_->get_cluster_sizes();
    Tensor partition_ids = index_->get_partition_ids();
    Tensor delete_ids = partition_ids.masked_select(partition_sizes < min_partition_size_);

    Tensor delete_centroids = index_->parent_->select_vectors(delete_ids);
    // remove the partitions
    index_->delete_partitions(delete_ids);

    return {delete_ids, delete_centroids};
}

std::tuple<Tensor, Tensor, Tensor> LireMaintenance::check_and_split_partitions() {
    // check for partitions greater than 2 * min_partition_size_
    if (index_->parent_ == nullptr) {
        return {};
    }

    Tensor partition_sizes = index_->get_cluster_sizes();
    Tensor partition_ids = index_->get_partition_ids();

    int max_partition_size = (int) (max_partition_ratio_ * target_partition_size_);
    Tensor old_ids = partition_ids.masked_select(partition_sizes > max_partition_size);


    Tensor old_centroids = index_->parent_->select_vectors(old_ids);

    // perform the split
    std::tuple<Tensor, vector<Tensor>, vector<Tensor> > split_partitions = index_->split_partitions(old_ids);
    Tensor split_centroids = std::get<0>(split_partitions);
    vector<Tensor> split_vectors = std::get<1>(split_partitions);
    vector<Tensor> split_ids = std::get<2>(split_partitions);

    index_->delete_partitions(old_ids);

    Tensor new_ids = index_->add_partitions(split_centroids, split_vectors, split_ids);

    return {new_ids, old_centroids, old_ids};
}

MaintenanceTimingInfo DeDriftMaintenance::maintenance() {

    MaintenanceTimingInfo timing_info;

    auto total_start = std::chrono::high_resolution_clock::now();

    // recompute the centroids
    index_->recompute_centroids();

    // select the top small and top large partitions
    Tensor partition_sizes = index_->get_cluster_sizes();
    Tensor partition_ids = index_->get_partition_ids();

    // sort the partition size
    Tensor sort_args = partition_sizes.argsort(0, true);

    // select the top small and top large partitions
    Tensor small_partition_ids =  partition_ids.index_select(0, sort_args.narrow(0, 0, k_small_));
    Tensor large_partition_ids = partition_ids.index_select(0, sort_args.narrow(0, partition_ids.size(0) - k_large_, k_large_));

    // run refinement on the small and large partitions
    Tensor all_partition_ids = torch::cat({small_partition_ids, large_partition_ids}, 0);
    index_->refine_clusters(all_partition_ids, refinement_iterations_);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing_info.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    return timing_info;
}
