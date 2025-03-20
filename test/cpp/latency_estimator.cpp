//
// Created by Jason on 10/21/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
//

#include <gtest/gtest.h>
#include "maintenance_cost_estimator.h"
#include "list_scanning.h"  // Must include your scan_list(...) definition
#include <cstdio>           // For remove()
#include <fstream>          // For file I/O

// Helper function to measure actual latency for given n and k
static float measure_actual_latency(const ListScanLatencyEstimator& estimator,
                                    int n, int k) {
  // Generate random vectors
  torch::Tensor vectors = torch::rand({n, estimator.d_});
  torch::Tensor ids = torch::randperm(n);
  torch::Tensor query = torch::rand({estimator.d_});

  auto topk_buffer = make_shared<TopkBuffer>(k, false);

  const float* query_ptr = query.data_ptr<float>();
  const float* vectors_ptr = vectors.data_ptr<float>();
  const int64_t* ids_ptr = ids.data_ptr<int64_t>();

  uint64_t total_latency_ns = 0;
  for (int m = 0; m < estimator.n_trials_; ++m) {
    auto start = std::chrono::high_resolution_clock::now();
    scan_list(query_ptr, vectors_ptr, ids_ptr, n, estimator.d_, *topk_buffer);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    total_latency_ns += duration.count();
  }
  double mean_latency_ns = static_cast<double>(total_latency_ns) / estimator.n_trials_;
  return static_cast<float>(mean_latency_ns);
}

TEST(ListScanLatencyEstimatorTest, BasicInterpolationExtrapolation) {
  // Use smaller grids for fast tests
  int d = 8;
  std::vector<int> n_values = {16, 32, 64};
  std::vector<int> k_values = {1, 2, 4};
  int n_trials = 2;  // small for testing

  // Use a test file name (we'll remove it later)
  std::string test_filename = "test_latency_profile.csv";
  // Ensure it's not present at start
  std::remove(test_filename.c_str());

  ListScanLatencyEstimator estimator(d, n_values, k_values, n_trials,
                                     /*adaptive_nprobe=*/false, test_filename);

  // Because the file didn't exist, it should have profiled and created the file.
  // Test that file was created
  {
    std::ifstream ifs(test_filename);
    EXPECT_TRUE(ifs.good()) << "Profile file should be created.";
  }

  // Check some interpolation
  // n=48 => between 32 and 64, k=3 => between 2 and 4
  float latency = estimator.estimate_scan_latency(48, 3);
  EXPECT_GT(latency, 0.0f) << "Should have a positive interpolated latency.";

  // Check boundary conditions
  float boundary1 = estimator.estimate_scan_latency(16, 1);
  float boundary2 = estimator.estimate_scan_latency(64, 4);
  EXPECT_GT(boundary1, 0.0f);
  EXPECT_GT(boundary2, 0.0f);

  // Below min n or k => should throw
  EXPECT_THROW(estimator.estimate_scan_latency(1, 1), std::out_of_range);

  // Clean up file
  std::remove(test_filename.c_str());
}

TEST(ListScanLatencyEstimatorTest, ReloadFromFile) {
  int d = 8;
  std::vector<int> n_values = {16, 32};
  std::vector<int> k_values = {1, 2};
  int n_trials = 1;
  std::string test_filename = "reload_test_profile.csv";
  std::remove(test_filename.c_str());

  // First instance => profiles, saves to file
  {
    ListScanLatencyEstimator estimator(d, n_values, k_values, n_trials,
                                       false, test_filename);
    // Just call something
    float latency = estimator.estimate_scan_latency(16, 2);
    EXPECT_GT(latency, 0.0f);
  }

  // Second instance => loads from the existing file, no fresh profiling
  // We can detect this by artificially removing read permissions or
  // by verifying it doesn't throw. We'll do a simpler check:
  {
    // We can store some timestamp or do an ephemeral test. For now, we trust
    // that if the file exists, it tries to load. If it doesn't match grids,
    // it won't load. So let's ensure it DOES match and it DOES load.

    ListScanLatencyEstimator estimator2(d, n_values, k_values, n_trials,
                                        false, test_filename);
    float latency = estimator2.estimate_scan_latency(16, 1);
    EXPECT_GT(latency, 0.0f) << "Should have loaded from file successfully.";
  }

  // Clean up
  std::remove(test_filename.c_str());
}

TEST(ListScanLatencyEstimatorTest, MismatchedGridsForFile) {
  int d = 8;
  std::vector<int> n_values_1 = {16, 32};
  std::vector<int> k_values_1 = {1, 2};
  int n_trials = 1;
  std::string test_filename = "mismatch_test_profile.csv";
  std::remove(test_filename.c_str());

  // First: create a file with certain n_values, k_values
  {
    ListScanLatencyEstimator estimator(d, n_values_1, k_values_1, n_trials,
                                       false, test_filename);
    float latency = estimator.estimate_scan_latency(32, 2);
    EXPECT_GT(latency, 0.0f);
  }

  // Now create a new estimator with a different grid
  std::vector<int> n_values_2 = {16, 32, 64};
  ListScanLatencyEstimator estimator2(d, n_values_2, k_values_1, n_trials,
                                      false, test_filename);
  // Because the file grid won't match (mismatch in n_values), it should have
  // re-profiled. We just do a basic call to confirm it works
  float lat = estimator2.estimate_scan_latency(64, 2);
  EXPECT_GT(lat, 0.0f);

  // Clean up
  std::remove(test_filename.c_str());
}

// TEST(ListScanLatencyEstimatorTest, EstimateVsActualLatency) {
//   int d = 32;
//   std::vector<int> n_values = {64, 256, 1024};
//   std::vector<int> k_values = {1, 4, 16};
//   int n_trials = 25;
//
//   // clear old profile file if it exists
//   std::string test_filename = "latency_profile.csv";
//   std::remove(test_filename.c_str());
//
//   // In practice, you might have a bigger grid, but let's keep it short for test
//   ListScanLatencyEstimator estimator(d, n_values, k_values, n_trials);
//
//   // Profile is performed in constructor if not already loaded from file
//   std::vector<std::pair<int, int>> test_cases = {
//       {16, 4},
//       {64, 1},
//       {64, 16},
//       {256, 16},
//       {100, 5},   // interpolation
//       {300, 16},  // interpolation
//       {512, 16},  // extrapolation
//   };
//
//   for (auto& tc : test_cases) {
//     int n = tc.first;
//     int k = tc.second;
//     if (n < n_values.front() || k < k_values.front()) {
//       EXPECT_THROW(estimator.estimate_scan_latency(n, k), std::out_of_range);
//       continue;
//     }
//
//     float estimated_latency_ns = estimator.estimate_scan_latency(n, k);
//     float actual_latency_ns = measure_actual_latency(estimator, n, k);
//
//     float estimated_ms = estimated_latency_ns / 1e6f;
//     float actual_ms = actual_latency_ns / 1e6f;
//     std::cout << "n=" << n << ", k=" << k
//               << " => estimated=" << estimated_ms << "ms, actual=" << actual_ms
//               << "ms\n";
//
//     // Tolerance of 40% because these are quite approximate with small n_trials
//     float tolerance = 0.4f * actual_ms;
//     EXPECT_NEAR(estimated_ms, actual_ms, tolerance)
//         << "Difference is too large for n=" << n << ", k=" << k;
//   }
// }