#include <gtest/gtest.h>
#include <torch/torch.h>
#include "clustering.h"
#include <vector>

// Helper functions to generate random test data.
static Tensor generate_random_data(int64_t num_vectors, int64_t dim) {
  return torch::randn({num_vectors, dim}, torch::kFloat32).contiguous();
}

static Tensor generate_sequential_ids(int64_t count, int64_t start = 0) {
  return torch::arange(start, start + count, torch::kInt64).contiguous();
}

// Helpers to compute mean squared error (MSE) for clustering.
// For FAISS-based clustering: 'vectors' is a vector of tensors (one per cluster).
static double compute_mse(const Tensor& centroids,
                              const std::vector<Tensor>& clusters) {
  double total_error = 0.0;
  int64_t total_count = 0;
  auto centroids_cpu = centroids.to(torch::kCPU);
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (clusters[i].size(0) > 0) {
      auto cluster = clusters[i].to(torch::kCPU);
      auto diff = cluster - centroids_cpu[i].unsqueeze(0);
      total_error += diff.pow(2).sum().item<double>();
      total_count += cluster.size(0);
    }
  }
  return total_count > 0 ? total_error / total_count : 0.0;
}

// Test fixture for clustering tests.
class ClusteringTest : public ::testing::Test {
 protected:
  const int64_t num_vectors = 5000;
  const int64_t dim = 64;
  const int num_clusters = 20;
  Tensor vectors_cpu, ids_cpu;
  Tensor vectors_cuda, ids_cuda;

  void SetUp() override {
    // Skip these tests if CUDA is not available.
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available; skipping clustering tests.";
    }
    vectors_cpu = generate_random_data(num_vectors, dim);
    ids_cpu = generate_sequential_ids(num_vectors);
#ifdef QUAKE_ENABLE_GPU
    vectors_cuda = vectors_cpu.to(torch::kCUDA).contiguous();
    ids_cuda = ids_cpu.to(torch::kCUDA).contiguous();
#endif
  }
};

// Compare clustering methods using the L2 (Euclidean) metric.
#ifdef QUAKE_ENABLE_GPU
TEST_F(ClusteringTest, CompareClustering_L2) {
  const int niter = 20;
  // FAISS-based clustering on CPU.
  auto clustering_cpu = kmeans_cpu(vectors_cpu, ids_cpu, num_clusters, faiss::METRIC_L2, niter, Tensor());
  // cuVS-based clustering on GPU.
  auto clustering_cuvs = kmeans_cuvs(vectors_cuda, ids_cuda, num_clusters, faiss::METRIC_L2);

  // Verify centroid shapes.
  ASSERT_EQ(clustering_cpu->centroids.dim(), 2);
  ASSERT_EQ(clustering_cpu->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering_cpu->centroids.size(1), dim);
  ASSERT_EQ(clustering_cuvs->centroids.dim(), 2);
  ASSERT_EQ(clustering_cuvs->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering_cuvs->centroids.size(1), dim);

  // Compare the mean squared errors (MSEs) between the methods.
  double mse_cpu = compute_mse(clustering_cpu->centroids, clustering_cpu->vectors);
  double mse_cuvs = compute_mse(clustering_cuvs->centroids, clustering_cuvs->vectors);
  // They should agree within roughly 20% relative difference.
  ASSERT_NEAR(mse_cpu, mse_cuvs, mse_cpu * 0.20);
}

// Compare clustering methods using the inner product metric.
TEST_F(ClusteringTest, CompareClustering_InnerProduct) {
  const int niter = 20;
  auto clustering_cpu = kmeans(vectors_cpu, ids_cpu, num_clusters,
                               faiss::METRIC_INNER_PRODUCT, niter, false, Tensor());
  auto clustering_cuvs = kmeans_cuvs(vectors_cuda, ids_cuda, num_clusters, faiss::METRIC_INNER_PRODUCT);

  // Verify centroid shapes.
  ASSERT_EQ(clustering_cpu->centroids.dim(), 2);
  ASSERT_EQ(clustering_cpu->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering_cpu->centroids.size(1), dim);
  ASSERT_EQ(clustering_cuvs->centroids.dim(), 2);
  ASSERT_EQ(clustering_cuvs->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering_cuvs->centroids.size(1), dim);

  // For inner product, check that cuVS centroids are normalized.
  auto norms = clustering_cuvs->centroids.norm(2, 1);
  for (int i = 0; i < norms.size(0); ++i) {
    ASSERT_NEAR(norms[i].item<float>(), 1.0f, 1e-3);
  }

  double mse_cpu = compute_mse(clustering_cpu->centroids, clustering_cpu->vectors);
  double mse_cuvs = compute_mse(clustering_cuvs->centroids, clustering_cuvs->vectors);
  // Allow a bit larger relative difference for inner product.
  ASSERT_NEAR(mse_cpu, mse_cuvs, mse_cpu * 0.25);
}

// Test that cuVS clustering partitions all vectors correctly (L2 metric).
TEST_F(ClusteringTest, CUVSClustering_Partitioning_L2) {
  auto result = kmeans_cuvs(vectors_cuda, ids_cuda, num_clusters, faiss::METRIC_L2);
  ASSERT_EQ(result->centroids.dim(), 2);
  ASSERT_EQ(result->centroids.size(0), num_clusters);
  ASSERT_EQ(result->centroids.size(1), dim);

  int64_t total_vectors = 0;
  for (int i = 0; i < num_clusters; ++i) {
    auto cluster = result->vectors[i];
    ASSERT_EQ(cluster.size(0), result->vector_ids[i].size(0));
    total_vectors += cluster.size(0);
  }
  ASSERT_EQ(total_vectors, num_vectors);
}


// Test that cuVS clustering (inner product) produces unit-norm centroids and correctly partitions vectors.
TEST_F(ClusteringTest, CUVSClustering_Partitioning_InnerProduct) {
  auto result = kmeans_cuvs(vectors_cuda, ids_cuda, num_clusters, faiss::METRIC_INNER_PRODUCT);
  ASSERT_EQ(result->centroids.dim(), 2);
  ASSERT_EQ(result->centroids.size(0), num_clusters);
  ASSERT_EQ(result->centroids.size(1), dim);

  // Verify centroids are normalized.
  auto norms = result->centroids.norm(2, 1);
  for (int i = 0; i < norms.size(0); ++i) {
    ASSERT_NEAR(norms[i].item<float>(), 1.0f, 1e-3);
  }

  int64_t total_vectors = 0;
  for (int i = 0; i < num_clusters; ++i) {
    auto cluster = result->vectors[i];
    ASSERT_EQ(cluster.size(0), result->vector_ids[i].size(0));
    total_vectors += cluster.size(0);
  }
  ASSERT_EQ(total_vectors, num_vectors);
}
#endif

TEST_F(ClusteringTest, KMeansCPU_L2) {
  // Test CPU-based k-means clustering.
  int niter = 10;
  auto clustering = kmeans_cpu(vectors_cpu, ids_cpu, num_clusters, faiss::METRIC_L2, niter);
  ASSERT_EQ(clustering->centroids.dim(), 2);
  ASSERT_EQ(clustering->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering->centroids.size(1), dim);

  int64_t total_vectors = 0;
  for (int i = 0; i < num_clusters; ++i) {
    auto cluster = clustering->vectors[i];
    ASSERT_EQ(cluster.size(0), clustering->vector_ids[i].size(0));
    total_vectors += cluster.size(0);
  }
  ASSERT_EQ(total_vectors, num_vectors);
}

TEST_F(ClusteringTest, KMeansCPU_InnerProduct) {
  // Test CPU-based k-means clustering with inner product metric.
  int niter = 10;
  auto clustering = kmeans_cpu(vectors_cpu, ids_cpu, num_clusters, faiss::METRIC_INNER_PRODUCT, niter);
  ASSERT_EQ(clustering->centroids.dim(), 2);
  ASSERT_EQ(clustering->centroids.size(0), num_clusters);
  ASSERT_EQ(clustering->centroids.size(1), dim);

  int64_t total_vectors = 0;
  for (int i = 0; i < num_clusters; ++i) {
    auto cluster = clustering->vectors[i];
    ASSERT_EQ(cluster.size(0), clustering->vector_ids[i].size(0));
    total_vectors += cluster.size(0);
  }
  ASSERT_EQ(total_vectors, num_vectors);
}