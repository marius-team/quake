#include "maintenance_policies.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

#include "quake_index.h"

using std::chrono::steady_clock;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::vector;
using std::unordered_map;
using std::shared_ptr;

//
// Constructor: Initialize the PartitionManager, parameters, cost estimator, and hit tracker.
//
MaintenancePolicy::MaintenancePolicy(
    shared_ptr<PartitionManager> partition_manager,
    shared_ptr<MaintenancePolicyParams> params)
    : partition_manager_(partition_manager),
      params_(params) {
    // Initialize the cost estimator.
    cost_estimator_ = std::make_shared<MaintenanceCostEstimator>(
        partition_manager_->d(), // Assumes PartitionManager::get_dimension() exists.
        params_->alpha,
        10);
    // Initialize the hit count tracker using the window size and total vector count.
    hit_count_tracker_ = std::make_shared<HitCountTracker>(
        params_->window_size, partition_manager_->ntotal());
}

//
// perform_maintenance() orchestrates deletion, splitting, and local refinement.
// It aggregates hit counts from the HitCountTracker, uses the cost estimator
// to decide on actions, and calls the PartitionManager methods accordingly.
//
shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::perform_maintenance() {
    // only consider split/deletion once the window is full
    if (hit_count_tracker_->get_num_queries_recorded() < params_->window_size) {
        return std::make_shared<MaintenanceTimingInfo>();
    }

    auto start_total = steady_clock::now();
    // STEP 1: Aggregate hit counts from the HitCountTracker.
    vector<vector<int64_t> > per_query_hits = hit_count_tracker_->get_per_query_hits();
    unordered_map<int64_t, int> aggregated_hits;
    for (const auto &query_hits: per_query_hits) {
        for (int64_t pid: query_hits) {
            aggregated_hits[pid]++;
        }
    }

    Tensor all_partition_ids_tens = partition_manager_->get_partition_ids();
    vector<int64_t> all_partition_ids = vector<int64_t>(all_partition_ids_tens.data_ptr<int64_t>(),
                                                        all_partition_ids_tens.data_ptr<int64_t>() +
                                                        all_partition_ids_tens.size(0));

    // STEP 2: Use cost estimation to decide which partitions to delete or split.
    int total_partitions = partition_manager_->nlist();
    float current_scan_fraction = hit_count_tracker_->get_current_scan_fraction();
    vector<int64_t> partitions_to_delete;
    vector<int64_t> partitions_to_split;

    int avg_partition_size = partition_manager_->ntotal() / total_partitions;
    for (const auto &partition_id: all_partition_ids) {
        // Get hit count and hit rate for the partition.
        int hit_count = aggregated_hits[partition_id];
        float hit_rate = static_cast<float>(hit_count) / static_cast<float>(params_->window_size);
        int partition_size = partition_manager_->get_partition_size(partition_id);

        // Deletion decision.
        float delete_delta = cost_estimator_->compute_delete_delta(
            partition_size, hit_rate, total_partitions, current_scan_fraction, avg_partition_size);

        if (delete_delta < -params_->delete_threshold_ns) {
            partitions_to_delete.push_back(partition_id);
        } else {
            if (partition_size > params_->min_partition_size) {
                float split_delta = cost_estimator_->compute_split_delta(
                    partition_size, hit_rate, total_partitions);
                if (split_delta < -params_->split_threshold_ns) {
                    partitions_to_split.push_back(partition_id);
                }
            }
        }
    }

    // Convert partition ID vectors to Torch tensors.
    torch::Tensor partitions_to_delete_tens = torch::from_blob(
        partitions_to_delete.data(), {static_cast<int64_t>(partitions_to_delete.size())},
        torch::kInt64).clone();
    torch::Tensor partitions_to_split_tens = torch::from_blob(
        partitions_to_split.data(), {static_cast<int64_t>(partitions_to_split.size())},
        torch::kInt64).clone();

    // STEP 3: Process deletions.
    auto start_delete = steady_clock::now();
    if (partitions_to_delete_tens.numel() > 0) {
        std::cout << "Deleting partitions " << partitions_to_delete_tens << std::endl;
        partition_manager_->delete_partitions(partitions_to_delete_tens);
    }
    for (int64_t partition_id: partitions_to_delete) {
        hit_count_tracker_->record_delete(partition_id, aggregated_hits[partition_id]);
    }
    auto end_delete = steady_clock::now();

    // STEP 4: Process splits.
    auto start_split = steady_clock::now();
    shared_ptr<Clustering> split_partitions;
    if (partitions_to_split_tens.numel() > 0) {
        split_partitions = partition_manager_->split_partitions(partitions_to_split_tens);
        // Assume each split yields two new partitions.
        std::cout << "Splitting partitions '" << partitions_to_split_tens.numel() << "' into " << 2 *
                partitions_to_split_tens.numel() << " partitions." << std::endl;
        auto new_partition_ids_accessor = split_partitions->partition_ids.accessor<int64_t, 1>();
        int64_t split_partition_offset = 0;
        int n_splits = 2;
        for (int64_t partition_id: partitions_to_split) {
            for (int i = 0; i < n_splits; i++) {
                split_partition_offset++;
                hit_count_tracker_->record_split(partition_id,
                                                 aggregated_hits[partition_id],
                                                 new_partition_ids_accessor[2 * split_partition_offset],
                                                 aggregated_hits[new_partition_ids_accessor[
                                                     2 * split_partition_offset]],
                                                 new_partition_ids_accessor[2 * split_partition_offset + 1],
                                                 aggregated_hits[new_partition_ids_accessor[
                                                     2 * split_partition_offset + 1]]);
            }
        }

        // remove old partitions
        partition_manager_->delete_partitions(partitions_to_split_tens, false);

        // add new partitions
        partition_manager_->add_partitions(split_partitions);
    }
    auto end_split = steady_clock::now();
    // STEP 5: Perform local refinement on newly split partitions.
    if (split_partitions && split_partitions->partition_ids.numel() > 0) {
        local_refinement(split_partitions->partition_ids);
    }
    auto end_total = steady_clock::now();

    // STEP 6: Fill in timing details.
    shared_ptr<MaintenanceTimingInfo> timing_info = std::make_shared<MaintenanceTimingInfo>();
    timing_info->delete_time_us = duration_cast<microseconds>(end_delete - start_delete).count();
    timing_info->split_time_us = duration_cast<microseconds>(end_split - start_split).count();
    timing_info->total_time_us = duration_cast<microseconds>(end_total - start_total).count();

    return timing_info;
}

//
// record_partition_hit() records a hit event for a partition by adding it as a query.
//
void MaintenancePolicy::record_query_hits(vector<int64_t> partition_ids) {
    vector<int64_t> scanned_sizes = partition_manager_->get_partition_sizes(partition_ids);
    hit_count_tracker_->add_query_data(partition_ids, scanned_sizes);
}

//
// reset() clears the hit count tracker’s stored query data.
//
void MaintenancePolicy::reset() {
    hit_count_tracker_->reset();
}

//
// check_partitions_for_deletion() aggregates hit counts and uses the cost estimator
// to determine which partitions are underutilized and candidates for deletion.
//
vector<int64_t> MaintenancePolicy::check_partitions_for_deletion() {
    vector<vector<int64_t> > per_query_hits = hit_count_tracker_->get_per_query_hits();
    unordered_map<int64_t, int> aggregated_hits;
    for (const auto &query_hits: per_query_hits) {
        for (int64_t pid: query_hits) {
            aggregated_hits[pid]++;
        }
    }

    vector<int64_t> partitions_to_delete;
    int total_partitions = partition_manager_->nlist();
    float current_scan_fraction = hit_count_tracker_->get_current_scan_fraction();
    int avg_partition_size = partition_manager_->ntotal() / total_partitions;
    for (const auto &entry: aggregated_hits) {
        int64_t partition_id = entry.first;
        int hit_count = entry.second;
        float hit_rate = static_cast<float>(hit_count) / static_cast<float>(params_->window_size);
        int partition_size = partition_manager_->get_partition_size(partition_id);
        float delta = cost_estimator_->compute_delete_delta(partition_size,
                                                            hit_rate,
                                                            total_partitions,
                                                            current_scan_fraction,
                                                            avg_partition_size);
        if (delta < -params_->delete_threshold_ns) {
            partitions_to_delete.push_back(partition_id);
        }
    }
    return partitions_to_delete;
}

//
// check_partitions_for_splitting() aggregates hit counts and uses the cost estimator
// to decide which partitions are overutilized and should be split.
//
vector<int64_t> MaintenancePolicy::check_partitions_for_splitting() {
    vector<vector<int64_t> > per_query_hits = hit_count_tracker_->get_per_query_hits();
    unordered_map<int64_t, int> aggregated_hits;
    for (const auto &query_hits: per_query_hits) {
        for (int64_t pid: query_hits) {
            aggregated_hits[pid]++;
        }
    }

    vector<int64_t> partitions_to_split;
    int total_partitions = partition_manager_->nlist();
    for (const auto &entry: aggregated_hits) {
        int64_t partition_id = entry.first;
        int hit_count = entry.second;
        float hit_rate = static_cast<float>(hit_count) / static_cast<float>(params_->window_size);
        int partition_size = partition_manager_->get_partition_size(partition_id);
        if (partition_size <= params_->min_partition_size) continue;
        float delta = cost_estimator_->compute_split_delta(partition_size, hit_rate, total_partitions);
        if (delta < -params_->split_threshold_ns) {
            partitions_to_split.push_back(partition_id);
        }
    }
    return partitions_to_split;
}

//
// local_refinement() delegates to the PartitionManager's refine_partitions() method.
//
void MaintenancePolicy::local_refinement(const torch::Tensor &partition_ids) {
    return;
    // Tensor centroids = partition_manager_->get(partition_ids);
    // auto search_params = make_shared<SearchParams>();
    // search_params->k = params_->refinement_radius;
    // auto search_result = partition_manager_->parent_->search(centroids, search_params);
    //
    // Tensor curr_partition_ids = torch::cat({search_result->ids.flatten(0, 1), partition_ids});
    // Tensor unique_partitions = std::get<0>(torch::_unique(curr_partition_ids));
    //
    // // ignore -1 partition id
    // unique_partitions = unique_partitions.masked_select(unique_partitions >= 0);
    //
    // partition_manager_->refine_partitions(unique_partitions, params_->refinement_iterations);
}
