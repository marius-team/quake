//
// Created by Jason on 10/21/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

// list_scan_latency_estimator_test.cpp

#include "maintenance_policies.h"
#include "list_scanning.h"
#include <gtest/gtest.h>

// Helper function to measure actual latency for given n and k
float measure_actual_latency(const ListScanLatencyEstimator& estimator, int n, int k) {
    // Generate random vectors
    torch::Tensor vectors = torch::rand({n, estimator.d_});
    torch::Tensor ids = torch::randperm(n);
    torch::Tensor query = torch::rand({estimator.d_});

    TopkBuffer topk_buffer(k, false);

    const float* query_ptr = query.data_ptr<float>();
    const float* vectors_ptr = vectors.data_ptr<float>();
    const int64_t* ids_ptr = ids.data_ptr<int64_t>();

    uint64_t total_latency_ns = 0;
    for (int m = 0; m < estimator.n_trials_; ++m) {
        auto start = std::chrono::high_resolution_clock::now();
        scan_list(query_ptr, vectors_ptr, ids_ptr, n, estimator.d_, topk_buffer);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total_latency_ns += duration.count();
    }
    double mean_latency_ns = static_cast<double>(total_latency_ns) / estimator.n_trials_;
    return static_cast<float>(mean_latency_ns);
}

TEST(ListScanLatencyEstimatorTest, EstimateVsActualLatency) {
    // Define dimensions and parameters
    int d = 128;
    std::vector<int> n_values = {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};
    std::vector<int> k_values = {1, 4, 16, 64, 256, 1024};
    int n_trials = 5;

    // Initialize the estimator
    ListScanLatencyEstimator estimator(d, n_values, k_values, n_trials);

    // Profile the scan latency
    auto start = std::chrono::high_resolution_clock::now();
    estimator.profile_scan_latency();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Profiled scan latency in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;

    // Define test cases: pairs of (n, k)
    std::vector<std::pair<int, int>> test_cases = {
        // Interpolation Cases
        {1500, 15},    // n between 1024 and 4096, k between 16 and 64
        {3000, 30},    // n between 1024 and 4096, k between 16 and 64
        {7500, 75},    // n between 4096 and 16384, k between 64 and 256
        {12000, 120},  // n between 4096 and 16384, k between 64 and 256
        {500, 5},      // n between 256 and 1024, k between 4 and 16
        {10000, 150},  // n between 4096 and 16384, k between 64 and 256
        {11000, 200},  // n between 4096 and 16384, k between 64 and 256

        // Extrapolation Cases
        {2000000, 2048}, // n and k exceed maximum grid values
        {8, 0},           // n and k below minimum grid values
        {32, 2},          // n and k below minimum grid values but within positive range
        {1048577, 1025},  // n just above maximum, k just above maximum
        {5000, 2048},     // k exceeds maximum
        {2097152, 64},    // n exceeds maximum
    };
    for (const auto& [n, k] : test_cases) {
        if (n < n_values.front() || k < k_values.front()) {
            // Expect an exception for below minimum values
            EXPECT_THROW({
                estimator.estimate_scan_latency(n, k);
            }, std::out_of_range);
            continue;
        }

        try {
            // Estimate the latency
            float estimated_latency_ns = estimator.estimate_scan_latency(n, k);

            // Measure the actual latency
            float actual_latency_ns = measure_actual_latency(estimator, n, k);

            float estimated_latency = estimated_latency_ns / 1e6;  // Convert to milliseconds
            float actual_latency = actual_latency_ns / 1e6;        // Convert to milliseconds

            std::cout << "Estimated latency: " << estimated_latency << " ms, Actual latency: " << actual_latency << " ms" << std::endl;

            // Allow a reasonable tolerance, e.g., 20% difference
            float tolerance = 0.2f * actual_latency;
            EXPECT_NEAR(estimated_latency, actual_latency, tolerance)
                << "Failed for n=" << n << ", k=" << k;
        } catch (const std::exception& e) {
            FAIL() << "Exception for n=" << n << ", k=" << k << ": " << e.what();
        }
    }
}
