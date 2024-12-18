//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LATENCY_ESTIMATION_H
#define LATENCY_ESTIMATION_H

#include <common.h>

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

#endif //LATENCY_ESTIMATION_H
