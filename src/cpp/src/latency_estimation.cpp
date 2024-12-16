//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <latency_estimation.h>
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
