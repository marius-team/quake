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

shared_ptr<MaintenanceTimingInfo> MaintenancePolicy::perform_maintenance() {
    // only consider split/deletion once the window is full

    int64_t num_queries = hit_count_tracker_->get_num_queries_recorded();
    if (hit_count_tracker_->get_num_queries_recorded() < params_->window_size) {
        std::cout << "Window not full yet. " << num_queries << " queries recorded and " << params_->window_size
                  << " queries required." << std::endl;
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

            if (params_->enable_delete_rejection && partition_size > params_->min_partition_size) {
                // check the assignments of the partitions to be deleted.
                auto search_params = make_shared<SearchParams>();
                search_params->k = 2; // get the top 2 partitions, ignore the first one as it is the partition itself
                search_params->batched_scan = true;
                float *partition_vectors = (float *) partition_manager_->partition_store_->partitions_[partition_id]->codes_;
                Tensor part_vecs = torch::from_blob(partition_vectors, {(int64_t) partition_manager_->partition_store_->list_size(partition_id),
                                                                       partition_manager_->d()}, torch::kFloat32);
                auto res = partition_manager_->parent_->search(part_vecs, search_params);

                Tensor reassign_ids = res->ids.flatten();

                // remove the partition itself
                reassign_ids = reassign_ids.masked_select(reassign_ids != partition_id);

                // Get A) the unique partitions, B) the number reassigned, C) the size of the partitions, D) hit rates of the partitions
                Tensor uniques;
                Tensor counts;
                std::tie(uniques, std::ignore, counts) = torch::_unique2(reassign_ids, true, false, true);
                Tensor part_sizes = partition_manager_->get_partition_sizes(uniques);

                // convert to vectors
                vector<int64_t> reassign_id_vec = vector<int64_t>(uniques.data_ptr<int64_t>(), uniques.data_ptr<int64_t>() + uniques.size(0));

                vector<int64_t> reassign_sizes = vector<int64_t>(part_sizes.data_ptr<int64_t>(),
                                                                 part_sizes.data_ptr<int64_t>() + part_sizes.size(0));
                vector<int64_t> reassign_counts = vector<int64_t>(counts.data_ptr<int64_t>(),
                                                                  counts.data_ptr<int64_t>() + counts.size(0));
                vector<float> hit_rates;
                for (int64_t reassign_id: reassign_id_vec) {
                    hit_rates.push_back(static_cast<float>(aggregated_hits[reassign_id]) / static_cast<float>(params_->window_size));
                }

                float delta = cost_estimator_->compute_delete_delta_w_reassign(partition_manager_->get_partition_size(partition_id),
                                                                              static_cast<float>(aggregated_hits[partition_id]) / static_cast<float>(params_->window_size),
                                                                              total_partitions,
                                                                              reassign_counts,
                                                                              reassign_sizes,
                                                                              hit_rates);

                if (delta < -params_->delete_threshold_ns) {
                    partitions_to_delete.push_back(partition_id);
                }
            } else {
                partitions_to_delete.push_back(partition_id);
            }
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
    Tensor partitions_to_delete_tens = torch::from_blob(
        partitions_to_delete.data(), {static_cast<int64_t>(partitions_to_delete.size())},
        torch::kInt64).clone();
    Tensor partitions_to_split_tens = torch::from_blob(
        partitions_to_split.data(), {static_cast<int64_t>(partitions_to_split.size())},
        torch::kInt64).clone();

    // STEP 3: Process deletions.
    auto start_delete = steady_clock::now();
    if (partitions_to_delete_tens.numel() > 0) {
        partition_manager_->delete_partitions(partitions_to_delete_tens);
    }
    auto end_delete = steady_clock::now();

    // STEP 4: Process splits.
    auto start_split = steady_clock::now();
    shared_ptr<Clustering> split_partitions;
    if (partitions_to_split_tens.numel() > 0) {

        // split the partitions into two
        split_partitions = partition_manager_->split_partitions(partitions_to_split_tens);

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

void MaintenancePolicy::record_query_hits(vector<int64_t> partition_ids) {
    vector<int64_t> scanned_sizes = partition_manager_->get_partition_sizes(partition_ids);
    hit_count_tracker_->add_query_data(partition_ids, scanned_sizes);
}

void MaintenancePolicy::reset() {
    hit_count_tracker_->reset();
}

void MaintenancePolicy::local_refinement(const torch::Tensor &partition_ids) {
    Tensor split_centroids = partition_manager_->parent_->get(partition_ids);
    auto search_params = std::make_shared<SearchParams>();
    search_params->nprobe = 1000;
    search_params->k = params_->refinement_radius;

    if (params_->refinement_radius == 0) {
        return;
    }

    auto result = partition_manager_->parent_->search(split_centroids, search_params);
    Tensor refine_ids = std::get<0>(torch::_unique(result->ids));
    refine_ids = refine_ids.masked_select(refine_ids != -1);
    partition_manager_->refine_partitions(refine_ids, params_->refinement_iterations);
}
