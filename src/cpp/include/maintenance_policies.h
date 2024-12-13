//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef MAINTENANCE_POLICIES_H
#define MAINTENANCE_POLICIES_H

#include <dynamic_ivf.h>
#include <torch/torch.h>
// #include <dynamic_ivf.h>

class DynamicIVF_C;
using std::vector;
using torch::Tensor;

// 2D function that estimates the scan latency of a list given it's size and the number of elements to retrieve
// l(n, k) = latency
// function is a linear interpolation of measured scan latency for different list sizes and k
// for points beyond the measured range, the function is extrapolated using the slope of the last two points
// default grid: n = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576], k = [1, 4, 16, 64, 256, 1024]
class ListScanLatencyEstimator {
public:
    ListScanLatencyEstimator(int d, const std::vector<int>& n_values, const std::vector<int>& k_values, int n_trials = 100, bool adaptive_nprobe = false)
        : d_(d), n_values_(n_values), k_values_(k_values), n_trials_(n_trials) {
        // Initialize the latency model grid
        scan_latency_model_ = std::vector<std::vector<float>>(n_values_.size(), std::vector<float>(k_values_.size(), 0.0f));

        // Ensure n_values and k_values are sorted in ascending order
        if (!std::is_sorted(n_values_.begin(), n_values_.end())) {
            throw std::runtime_error("n_values must be sorted in ascending order.");
        }

        if (!std::is_sorted(k_values_.begin(), k_values_.end())) {
            throw std::runtime_error("k_values must be sorted in ascending order.");
        }
    }

    // Profiles the scan latency and populates the scan_latency_model_
    void profile_scan_latency();

    // Estimates the scan latency for given n and k
    float estimate_scan_latency(int n, int k) const;

    // Setter for n_trials_
    void set_n_trials(int n_trials) {
        n_trials_ = n_trials;
    }

    int d_;
    std::vector<int> n_values_;
    std::vector<int> k_values_;
    std::vector<std::vector<float>> scan_latency_model_;
    int n_trials_;

    // Helper function for interpolation
    // Returns the lower and upper indices and the fractional part for a given target
    bool get_interpolation_info(const std::vector<int>& values, int target, int& lower, int& upper, float& frac) const;
};


struct MaintenanceAuditInfo {
    float average_hit_rate;
    float total_cost;
    float average_split_delta_cost;
    float average_delete_delta_cost;
    float average_alpha;

    int64_t n_splits;
    int64_t n_deletes;
    int64_t vectors_reassigned_;

    Tensor split_partition_ids;
    Tensor new_partition_ids;
    Tensor delete_partition_ids;

    Tensor split_delta_costs;
    Tensor delete_delta_costs;

    // hit tracking
    std::unordered_map<int64_t, int64_t> per_partition_hits;
    std::unordered_map<int64_t, int64_t> ancestor_partition_hits;
    std::unordered_map<int64_t, std::pair<int64_t, int64_t>> split_records;
    std::unordered_map<int64_t, float> deleted_partition_hit_rate;

    // scan fraction tracking
    vector<vector<int64_t> > per_query_hits;
    vector<vector<int64_t> > per_query_scanned_partitions_sizes;
    float running_sum_scan_fraction;
    float current_scan_fraction;
};


/**
 * @brief Structure to hold timing information for maintenance operations.
 */
struct MaintenanceTimingInfo {
    int64_t n_splits; ///< Number of splits.
    int64_t n_deletes; ///< Number of merges.
    int64_t delete_time_us; ///< Time spent on deletions in microseconds.
    int64_t delete_refine_time_us; ///< Time spent on deletions with refinement in microseconds.
    int64_t split_time_us; ///< Time spent on splits in microseconds.
    int64_t split_refine_time_us; ///< Time spent on splits with refinement in microseconds.
    int64_t total_time_us; ///< Total time spent in microseconds.
    MaintenanceAuditInfo audit_before;
    MaintenanceAuditInfo audit_after;
    /**
     * @brief Prints the timing information.
     */
    void print() const {
        std::cout << "#### Maintenance Timing Information ####" << std::endl;
        std::cout << "Splits: " << n_splits << ", Deletes: " << n_deletes << std::endl;
        std::cout << "Delete time (us): " << delete_time_us << std::endl;
        std::cout << "Delete refine time (us): " << delete_refine_time_us << std::endl;
        std::cout << "Split time (us): " << split_time_us << std::endl;
        std::cout << "Split refine time (us): " << split_refine_time_us << std::endl;
        std::cout << "Total time (us): " << total_time_us << std::endl;
    }
};

struct MaintenancePolicyParams {

    std::string maintenance_policy = "query_cost";
    int window_size = 1000;
    int refinement_radius = 100;
    int refinement_iterations = 3;
    int min_partition_size = 32;
    float alpha = .9;
    bool enable_split_rejection = true;
    bool enable_delete_rejection = true;

    float delete_threshold_ns = 20.0;
    float split_threshold_ns = 20.0;

    // de-drift parameters
    int k_large = 50;
    int k_small = 50;
    bool modify_centroids = true;

    // lire parameters
    int target_partition_size = 1000;
    float max_partition_ratio = 2.0;

    MaintenancePolicyParams() = default;
};

class MaintenancePolicy {
public:
    int curr_query_id_;
    vector<vector<int64_t> > per_query_hits_;
    vector<vector<int64_t> > per_query_scanned_partitions_sizes_;
    float running_sum_scan_fraction_;
    float current_scan_fraction_;
    std::string maintenance_policy_name_;

    std::shared_ptr<DynamicIVF_C> index_;
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

    virtual void refine_delete(Tensor old_centroids) {}

    virtual void refine_split(Tensor partition_ids, Tensor old_centroids) {}

    virtual std::pair<Tensor, Tensor> check_and_delete_partitions() {}

    virtual std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() {}

    MaintenanceAuditInfo audit();

    virtual MaintenanceTimingInfo maintenance();
};

class QueryCostMaintenance : public MaintenancePolicy {
public:
    QueryCostMaintenance(std::shared_ptr<DynamicIVF_C> index, MaintenancePolicyParams params = MaintenancePolicyParams());

    float compute_alpha_for_window();

    void refine_delete(Tensor old_centroids) override;

    void refine_split(Tensor partition_ids, Tensor old_centroids) override;

    std::pair<Tensor, Tensor> check_and_delete_partitions() override;

    std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() override;
};

class LireMaintenance : public MaintenancePolicy {
public:
    int target_partition_size_;
    float max_partition_ratio_;
    int min_partition_size_;

    LireMaintenance(std::shared_ptr<DynamicIVF_C> index, int target_partition_size, float max_partition_ratio, int min_partition_size) : target_partition_size_(target_partition_size), max_partition_ratio_(max_partition_ratio), min_partition_size_(min_partition_size) {
        maintenance_policy_name_ = "lire";
        index_ = index;
        refinement_iterations_ = 0;
    }

    void refine_split(Tensor partition_ids, Tensor old_centroids) override;

    std::pair<Tensor, Tensor> check_and_delete_partitions() override;

    std::tuple<Tensor, Tensor, Tensor> check_and_split_partitions() override;
};


class DeDriftMaintenance : public MaintenancePolicy {
public:
    int k_large_;
    int k_small_;
    bool modify_centroids_;

    DeDriftMaintenance(std::shared_ptr<DynamicIVF_C> index, int k_large, int k_small, bool modify_centroids) : k_large_(k_large), k_small_(k_small), modify_centroids_(modify_centroids) {
        maintenance_policy_name_ = "dedrift";
        index_ = index;
    }

    MaintenanceTimingInfo maintenance() override;
};



#endif //MAINTENANCE_POLICIES_H
