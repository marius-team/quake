// test_cuvs_clustering.cpp
//
// This file tests the cuVS-based k-means clustering implementation.
// It verifies that the centroids are computed correctly and that the
// entire dataset is partitioned among the clusters.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include "cuv_cluster.h"  // Contains ClusterWithCuVS, ClusteringResult, and MetricType

// Test that clustering with L2 metric produces the expected output shapes,
// and that every input vector is assigned to one and only one cluster.
TEST(CUVSClusteringTest, ClusteringL2) {
  // Skip the test if CUDA is not available.
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available, skipping cuVS clustering test.";
  }

  const int64_t num_vectors = 1000;
  const int64_t dim = 32;
  const int64_t num_clusters = 10;

  // Generate random data and corresponding sequential IDs on the GPU.
  auto vectors = torch::randn({num_vectors, dim}, torch::kCUDA).to(torch::kFloat32).contiguous();
  auto ids = torch::arange(0, num_vectors, torch::kCUDA).contiguous();

  // Call the cuVS-based clustering function using L2 metric.
  ClusteringResult result = ClusterWithCuVS(vectors, ids, num_clusters, faiss::METRIC_L2);

  // Validate that the centroids tensor has shape [num_clusters, dim].
  ASSERT_EQ(result.centroids.dim(), 2);
  ASSERT_EQ(result.centroids.size(0), num_clusters);
  ASSERT_EQ(result.centroids.size(1), dim);

  // Verify that all input vectors are partitioned by summing counts from each cluster.
  int64_t total_count = 0;
  for (const auto& cluster : result.clusters) {
    // Each cluster is returned as a pair: <cluster_vectors, cluster_ids>
    torch::Tensor cluster_vectors = cluster.first;
    torch::Tensor cluster_ids = cluster.second;
    // The number of vectors in each cluster must match the number of IDs.
    ASSERT_EQ(cluster_vectors.size(0), cluster_ids.size(0));
    total_count += cluster_vectors.size(0);
  }
  ASSERT_EQ(total_count, num_vectors);
}

// Test that clustering with the inner product metric (which performs normalization)
// yields centroids that are unit-norm and partitions the dataset correctly.
TEST(CUVSClusteringTest, ClusteringInnerProduct) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available, skipping cuVS clustering test.";
  }

  const int64_t num_vectors = 1000;
  const int64_t dim = 64;
  const int64_t num_clusters = 8;

  // Generate random input data and corresponding IDs.
  auto vectors = torch::randn({num_vectors, dim}, torch::kCUDA).to(torch::kFloat32).contiguous();
  auto ids = torch::arange(0, num_vectors, torch::kCUDA).contiguous();

  // Run clustering with inner product metric. The implementation should normalize the vectors.
  ClusteringResult result = ClusterWithCuVS(vectors, ids, num_clusters, faiss::METRIC_INNER_PRODUCT);

  // Check that the centroids tensor has the expected shape.
  ASSERT_EQ(result.centroids.dim(), 2);
  ASSERT_EQ(result.centroids.size(0), num_clusters);
  ASSERT_EQ(result.centroids.size(1), dim);

  // Optionally, validate that each centroid is normalized (i.e. its L2 norm is ~1)
  auto cent_norms = result.centroids.norm(2, /*dim=*/1);
  for (int64_t i = 0; i < cent_norms.size(0); ++i) {
    float norm_val = cent_norms[i].item<float>();
    // Allow a small numerical tolerance.
    ASSERT_NEAR(norm_val, 1.0, 1e-3);
  }

  // Confirm that the total number of vectors across all clusters is equal to the input size.
  int64_t total_count = 0;
  for (const auto& cluster : result.clusters) {
    torch::Tensor cluster_vectors = cluster.first;
    torch::Tensor cluster_ids = cluster.second;
    ASSERT_EQ(cluster_vectors.size(0), cluster_ids.size(0));
    total_count += cluster_vectors.size(0);
  }
  ASSERT_EQ(total_count, num_vectors);
}
