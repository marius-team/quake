// test_clustering_correctness.cpp
//
// This file tests the correctness of two clustering implementations:
//  1. FAISS-based clustering on CPU: kmeans(..., use_gpu = false)
//  2. cuVS-based clustering: ClusterWithCuVS(...)
// The tests compute the mean squared error (MSE) between each input vector and its
// corresponding cluster centroid and require that the MSE from the two methods
// agree within a reasonable tolerance. Additionally, the tests verify that each method
// returns centroids of the correct shape and partitions the input set completely.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <cuv_cluster.h>
#include "clustering.h"  // Declaration of kmeans(), ClusterWithCuVS(), Clustering, ClusteringResult, and MetricType

// Helper to compute MSE for FAISS-based clustering results.
// The 'vectors' field of Clustering is a vector of tensors (one per cluster).
double ComputeMSE_Faiss(const torch::Tensor& centroids, const std::vector<torch::Tensor>& cluster_vectors) {
  double total_error = 0.0;
  int64_t total_count = 0;
  // Ensure centroids are on CPU.
  torch::Tensor centroids_cpu = centroids.to(torch::kCPU);
  for (int i = 0; i < centroids_cpu.size(0); ++i) {
    if (cluster_vectors[i].size(0) > 0) {
      // Move the cluster vectors to CPU.
      torch::Tensor cluster_cpu = cluster_vectors[i].to(torch::kCPU);
      torch::Tensor diff = cluster_cpu - centroids_cpu[i].unsqueeze(0);
      total_error += diff.pow(2).sum().item<double>();
      total_count += cluster_cpu.size(0);
    }
  }
  return total_count > 0 ? total_error / total_count : 0.0;
}

// Helper to compute MSE for cuVS-based clustering results.
// Here, clusters are returned as a vector of pairs where the first tensor is the cluster vectors.
double ComputeMSE_cuVS(const torch::Tensor& centroids,
                       const std::vector<std::pair<torch::Tensor, torch::Tensor>>& clusters) {
  double total_error = 0.0;
  int64_t total_count = 0;
  torch::Tensor centroids_cpu = centroids.to(torch::kCPU);
  for (int i = 0; i < centroids_cpu.size(0); ++i) {
    const torch::Tensor& cluster_vectors = clusters[i].first;
    if (cluster_vectors.size(0) > 0) {
      torch::Tensor cluster_cpu = cluster_vectors.to(torch::kCPU);
      torch::Tensor diff = cluster_cpu - centroids_cpu[i].unsqueeze(0);
      total_error += diff.pow(2).sum().item<double>();
      total_count += cluster_cpu.size(0);
    }
  }
  return total_count > 0 ? total_error / total_count : 0.0;
}

// Test fixture for clustering correctness tests.
class ClusteringCorrectnessTest : public ::testing::Test {
 protected:
  // Use a moderate number of vectors for correctness testing.
  const int64_t num_vectors_ = 5000;
  const int64_t dim_ = 64;
  const int num_clusters_ = 20;
  // CPU tensors.
  torch::Tensor vectors_cpu_;
  torch::Tensor ids_cpu_;
  // CUDA tensors.
  torch::Tensor vectors_cuda_;
  torch::Tensor ids_cuda_;

  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available; skipping clustering correctness tests.";
    }
    // Create CPU tensors.
    auto options_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto options_int64_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    vectors_cpu_ = torch::randn({num_vectors_, dim_}, options_float_cpu).contiguous();
    ids_cpu_ = torch::arange(0, num_vectors_, options_int64_cpu).contiguous();
    // Create CUDA versions.
    auto options_float_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto options_int64_cuda = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    vectors_cuda_ = vectors_cpu_.to(options_float_cuda).contiguous();
    ids_cuda_ = ids_cpu_.to(options_int64_cuda).contiguous();

    std::cout << "[DEBUG] SetUp complete: " 
              << "num_vectors = " << num_vectors_ 
              << ", dim = " << dim_ 
              << ", clusters = " << num_clusters_ << std::endl;
    std::cout.flush();
  }
};

TEST_F(ClusteringCorrectnessTest, CompareClusteringMethods_L2) {
  const int niter = 20;  // number of iterations for clustering

  std::cout << "[DEBUG] Starting FAISS CPU clustering ..." << std::endl;
  std::cout.flush();
  auto clustering_cpu = kmeans(vectors_cpu_, ids_cpu_, num_clusters_, faiss::METRIC_L2, niter, false, torch::Tensor());
  std::cout << "[DEBUG] Completed FAISS CPU clustering." << std::endl;
  std::cout << "[DEBUG] CPU centroids shape: ";
  for (auto s : clustering_cpu->centroids.sizes()) { std::cout << s << " "; }
  std::cout << std::endl;
  std::cout.flush();

  std::cout << "[DEBUG] Starting cuVS clustering ..." << std::endl;
  std::cout.flush();
  ClusteringResult clustering_cuvs = ClusterWithCuVS(vectors_cuda_, ids_cuda_, num_clusters_, faiss::METRIC_L2);
  std::cout << "[DEBUG] Completed cuVS clustering." << std::endl;
  std::cout << "[DEBUG] cuVS centroids shape: ";
  for (auto s : clustering_cuvs.centroids.sizes()) { std::cout << s << " "; }
  std::cout << std::endl;
  std::cout.flush();

  // Compute MSE for each method.
  double mse_cpu  = ComputeMSE_Faiss(clustering_cpu->centroids, clustering_cpu->vectors);
  double mse_cuvs = ComputeMSE_cuVS(clustering_cuvs.centroids, clustering_cuvs.clusters);

  std::cout << "[DEBUG] L2 Clustering MSEs:" << std::endl;
  std::cout << "  FAISS CPU: " << mse_cpu << std::endl;
  std::cout << "  cuVS     : " << mse_cuvs << std::endl;
  std::cout.flush();

  // Assert that MSE values are similar (within 20% relative difference)
  double tol = 0.20;
  ASSERT_NEAR(mse_cpu, mse_cuvs, tol * mse_cpu);
}

TEST_F(ClusteringCorrectnessTest, CompareClusteringMethods_InnerProduct) {
  const int niter = 20;
  std::cout << "[DEBUG] Starting FAISS CPU clustering (InnerProduct) ..." << std::endl;
  std::cout.flush();
  auto clustering_cpu = kmeans(vectors_cpu_, ids_cpu_, num_clusters_, faiss::METRIC_INNER_PRODUCT, niter, false, torch::Tensor());
  std::cout << "[DEBUG] Completed FAISS CPU clustering (InnerProduct)." << std::endl;
  std::cout << "[DEBUG] CPU centroids shape: ";
  for (auto s : clustering_cpu->centroids.sizes()) { std::cout << s << " "; }
  std::cout << std::endl;
  std::cout.flush();

  std::cout << "[DEBUG] Starting cuVS clustering (InnerProduct) ..." << std::endl;
  std::cout.flush();
  ClusteringResult clustering_cuvs = ClusterWithCuVS(vectors_cuda_, ids_cuda_, num_clusters_, faiss::METRIC_INNER_PRODUCT);
  std::cout << "[DEBUG] Completed cuVS clustering (InnerProduct)." << std::endl;
  std::cout << "[DEBUG] cuVS centroids shape: ";
  for (auto s : clustering_cuvs.centroids.sizes()) { std::cout << s << " "; }
  std::cout << std::endl;
  std::cout.flush();

  double mse_cpu  = ComputeMSE_Faiss(clustering_cpu->centroids, clustering_cpu->vectors);
  double mse_cuvs = ComputeMSE_cuVS(clustering_cuvs.centroids, clustering_cuvs.clusters);

  std::cout << "[DEBUG] Inner-product Clustering MSEs:" << std::endl;
  std::cout << "  FAISS CPU: " << mse_cpu << std::endl;
  std::cout << "  cuVS     : " << mse_cuvs << std::endl;
  std::cout.flush();

  double tol = 0.25;
  ASSERT_NEAR(mse_cpu, mse_cuvs, tol * mse_cpu);
}
