//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef MAINTENANCE_POLICIES_H
#define MAINTENANCE_POLICIES_H

#include <common.h>
#include <latency_estimation.h>
#include <partition_manager.h>

class PartitionManager;

class MaintenancePolicy {
public:
    int curr_query_id_;
    vector<vector<int64_t> > per_query_hits_;
    vector<vector<int64_t> > per_query_scanned_partitions_sizes_;
    float running_sum_scan_fraction_;
    float current_scan_fraction_;
    std::string maintenance_policy_name_;

    std::shared_ptr<PartitionManager> partition_manager_;
    std::unordered_map<int64_t, int64_t> per_partition_hits_;
    std::unordered_map<int64_t, int64_t> ancestor_partition_hits_;
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> split_records_;

    std::unordered_map<int64_t, float> deleted_partition_hit_rate_;

    // parameters
    int window_size_ = 2500;
    int refinement_radius_ = 25;
    int refinement_iterations_ = 3;
    int min_partition_size_ = 32;
    float alpha_ = .9;
    bool enable_split_rejection_ = true;
    bool enable_delete_rejection_ = true;
    float delete_threshold_ns_ = 20.0;
    float split_threshold_ns_ = 20.0;

    // latency estimator
    std::vector<int> latency_grid_n_values_ = {1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536};
    std::vector<int> latency_grid_k_values_ = {1, 4, 16, 64, 256};
    int n_trials_ = 5;

    std::shared_ptr<ListScanLatencyEstimator> latency_estimator_ = nullptr;
    std::shared_ptr<ListScanLatencyEstimator> latency_estimator_adaptive_nprobe = nullptr;

    vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> get_split_history();

    void set_params(MaintenancePolicyParams params);

    void decrement_hit_count(int64_t partition_id);

    void update_hits(vector<int64_t> hits);

    void add_split(int64_t old_partition_id, int64_t left_partition_id, int64_t right_partition_id);

    void add_partition(int64_t partition_id, int64_t hits=0);

    void remove_partition(int64_t partition_id);

    void refine_partitions(Tensor partition_ids, int refinement_iterations);

    virtual void refine_delete(Tensor old_centroids) {}

    virtual void refine_split(Tensor partition_ids, Tensor old_centroids) {}

    virtual Tensor check_and_delete_partitions() { return {}; }

    virtual std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() { return {}; }

    virtual MaintenanceTimingInfo maintenance();
};

class QueryCostMaintenance : public MaintenancePolicy {
public:
    QueryCostMaintenance(std::shared_ptr<PartitionManager> partition_manager, MaintenancePolicyParams params = MaintenancePolicyParams());

    float compute_alpha_for_window();

    void refine_delete(Tensor old_centroids) override;

    void refine_split(Tensor partition_ids, Tensor old_centroids) override;

    Tensor check_and_delete_partitions() override;

    std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() override;
};

class LireMaintenance : public MaintenancePolicy {
public:
    int target_partition_size_;
    float max_partition_ratio_;
    int min_partition_size_;

    LireMaintenance(std::shared_ptr<PartitionManager> partition_manager, int target_partition_size, float max_partition_ratio, int min_partition_size) : target_partition_size_(target_partition_size), max_partition_ratio_(max_partition_ratio), min_partition_size_(min_partition_size) {
        maintenance_policy_name_ = "lire";
        partition_manager_ = partition_manager;
        refinement_iterations_ = 0;
    }

    void refine_split(Tensor partition_ids, Tensor old_centroids) override;

    Tensor check_and_delete_partitions() override;

    std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() override;
};


class DeDriftMaintenance : public MaintenancePolicy {
public:
    int k_large_;
    int k_small_;
    bool modify_centroids_;

    DeDriftMaintenance(std::shared_ptr<PartitionManager> partition_manager, int k_large, int k_small, bool modify_centroids) : k_large_(k_large), k_small_(k_small), modify_centroids_(modify_centroids) {
        maintenance_policy_name_ = "dedrift";
        partition_manager_ = partition_manager;
    }

    MaintenanceTimingInfo maintenance() override;
};



#endif //MAINTENANCE_POLICIES_H
