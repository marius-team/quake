//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LATENCY_ESTIMATION_H
#define LATENCY_ESTIMATION_H

#include <common.h>
#include <fstream>

// 2D function that estimates the scan latency of a list given it's size and the number of elements to retrieve
// l(n, k) = latency
// function is a linear interpolation of measured scan latency for different list sizes and k
// for points beyond the measured range, the function is extrapolated using the slope of the last two points
// default grid: n = [16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576], k = [1, 4, 16, 64, 256, 1024]
class ListScanLatencyEstimator {
public:
    // Constructor attempts to load latency profile from disk if the file path is
    // provided. If loading fails (file not found or grid mismatch), it performs
    // the profile_scan_latency() and saves to file.
    ListScanLatencyEstimator(int d,
                             const std::vector<int> &n_values,
                             const std::vector<int> &k_values,
                             int n_trials = 100,
                             bool adaptive_nprobe = false,
                             const std::string &profile_filename = "");

    // Profiles the scan latency and populates scan_latency_model_.
    // This is expensive and should typically be called only once.
    void profile_scan_latency();

    // Estimates the scan latency for given n and k.
    float estimate_scan_latency(int n, int k) const;

    // Setter for n_trials_.
    void set_n_trials(int n_trials) {
        n_trials_ = n_trials;
    }

    // Saves the internally profiled latency model to a CSV file.
    // Returns true on success, false otherwise.
    bool save_latency_profile(const std::string &filename) const;

    // Loads an existing latency profile from a CSV file.
    // Returns true on success, false otherwise.
    bool load_latency_profile(const std::string &filename);

    // Public members for convenience/access.
    int d_;
    std::vector<int> n_values_;
    std::vector<int> k_values_;
    std::vector<std::vector<float> > scan_latency_model_;
    int n_trials_;

private:
    // Helper function for interpolation (not used directly in this code, but
    // shown as an example of how you might handle it).
    bool get_interpolation_info(const std::vector<int> &values,
                                int target,
                                int &lower,
                                int &upper,
                                float &frac) const;

    // Helper function to do linear extrapolation in 1D.
    inline float linear_extrapolate(float f1, float f2, float fraction) const {
        float slope = f2 - f1;
        return f2 + slope * fraction;
    }

    // The name of the CSV file to load/save from. Empty means "don't load/save."
    std::string profile_filename_;
};


#endif //LATENCY_ESTIMATION_H
