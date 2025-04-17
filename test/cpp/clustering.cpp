#include <gtest/gtest.h>
#include <torch/torch.h>
#include "clustering.h"

// Helpers to generate random data and sequential ids
static torch::Tensor generate_random_data(int64_t N, int64_t D) {
  return torch::randn({N, D}, torch::kFloat32).contiguous();
}
static torch::Tensor generate_sequential_ids(int64_t N, int64_t start = 0) {
  return torch::arange(start, start + N, torch::kInt64).contiguous();
}

// Compute mean squared error for clustering (for CPU sanity)
static double compute_mse(const torch::Tensor& centroids,
                          const std::vector<torch::Tensor>& clusters) {
  double total_err = 0.0;
  int64_t count = 0;
  auto C = centroids.to(torch::kCPU);
  for (size_t i = 0; i < clusters.size(); ++i) {
    auto cl = clusters[i].to(torch::kCPU);
    if (cl.size(0) == 0) continue;
    auto diff = cl - C[i].unsqueeze(0);
    total_err += diff.pow(2).sum().item<double>();
    count += cl.size(0);
  }
  return count>0 ? total_err / count : 0.0;
}

// Fixture
class ClusteringTest : public ::testing::Test {
 protected:
  const int64_t num_vectors = 5000;
  const int64_t dim         = 64;
  const int     num_clusters= 20;

  torch::Tensor vectors_cpu, ids_cpu;
#ifdef QUAKE_ENABLE_GPU
  torch::Tensor vectors_cuda, ids_cuda;
#endif

  void SetUp() override {
    vectors_cpu = generate_random_data(num_vectors, dim);
    ids_cpu     = generate_sequential_ids(num_vectors);

#ifdef QUAKE_ENABLE_GPU
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available";
    }
    vectors_cuda = vectors_cpu.to(torch::kCUDA).contiguous();
    ids_cuda     = ids_cpu.to(torch::kCUDA).contiguous();
#endif
  }
};

// Test existing CPU kmeans
TEST_F(ClusteringTest, KMeansCPU_L2) {
  auto cl = kmeans_cpu(vectors_cpu, ids_cpu, num_clusters,
                       faiss::METRIC_L2, /*niter=*/10, torch::Tensor());
  ASSERT_EQ(cl->centroids.sizes(), (std::vector<int64_t>{num_clusters, dim}));
  int64_t tot=0;
  for (int i=0;i<num_clusters;++i) {
    ASSERT_EQ(cl->vectors[i].size(0), cl->vector_ids[i].size(0));
    tot += cl->vectors[i].size(0);
  }
  ASSERT_EQ(tot, num_vectors);
}

// Compare CPU vs CPU wrapper
TEST_F(ClusteringTest, KMeansWrapper_CPU) {
  auto cl = kmeans(vectors_cpu, ids_cpu, num_clusters,
                   faiss::METRIC_L2, /*niter=*/10, /*use_gpu=*/false, torch::Tensor());
  ASSERT_EQ(cl->centroids.sizes(), (std::vector<int64_t>{num_clusters, dim}));
  int64_t tot=0;
  for (int i=0;i<num_clusters;++i) {
    tot += cl->vectors[i].size(0);
  }
  ASSERT_EQ(tot, num_vectors);
}

#ifdef QUAKE_ENABLE_GPU
TEST_F(ClusteringTest, SampleAndPredict_GPU_L2) {
  const int sample_size    = 1000;
  const int niter          = 5;
  const int gpu_batch_size = 512;

  auto cl = kmeans_cuvs_sample_and_predict(
      vectors_cpu, ids_cpu,
      num_clusters,
      faiss::METRIC_L2,
      sample_size,
      niter,
      gpu_batch_size);

  // centroids must live on CPU and have correct shape
  ASSERT_EQ(cl->centroids.device().type(), torch::kCPU);
  ASSERT_EQ(cl->centroids.sizes(), (std::vector<int64_t>{num_clusters, dim}));

  // all vectors accounted for
  int64_t tot=0;
  for (int i=0;i<num_clusters;++i) {
    auto &part = cl->vectors[i];
    ASSERT_EQ(part.device().type(), torch::kCPU);
    ASSERT_EQ(part.size(0), cl->vector_ids[i].size(0));
    tot += part.size(0);
  }
  ASSERT_EQ(tot, num_vectors);

  // Optional quality check: rough MSE vs CPU run
  auto cl_cpu = kmeans_cpu(vectors_cpu, ids_cpu, num_clusters,
                           faiss::METRIC_L2, niter, torch::Tensor());
  double mse_cpu = compute_mse(cl_cpu->centroids, cl_cpu->vectors);
  double mse_gpu = compute_mse(cl->centroids, cl->vectors);
  ASSERT_NEAR(mse_cpu, mse_gpu, mse_cpu * 0.30);
}

// Full wrapper test for GPU
TEST_F(ClusteringTest, KMeansWrapper_GPU) {
  const int gpu_sample    = 2000;
  const int gpu_iter      = 3;
  const int gpu_batch     = 1024;

  auto cl = kmeans(vectors_cpu, ids_cpu, num_clusters,
                   faiss::METRIC_L2, gpu_iter,
                   /*use_gpu=*/true,
                   torch::Tensor());
  ASSERT_EQ(cl->centroids.device().type(), torch::kCPU);
  ASSERT_EQ(cl->vectors.size(), size_t(num_clusters));
  int64_t tot=0;
  for (auto &p : cl->vectors) tot += p.size(0);
  ASSERT_EQ(tot, num_vectors);
}
#endif  // QUAKE_ENABLE_GPU
