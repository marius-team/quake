// benchmark_clustering_gtest.cpp
//
// This Google Test benchmarks two clustering implementations:
//  1. FAISS-based CPU clustering (kmeans with use_gpu == false)
//  2. cuVS-based GPU clustering (ClusterWithCuVS)
// on a dataset of 4,000,000 vectors (of dimension 128) and 10,000 clusters.
// The test prints the elapsed time (milliseconds) for each method.

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include "clustering.h"  // Declaration of kmeans() and ClusterWithCuVS()
#include "cuv_cluster.h"

using namespace std::chrono;

TEST(ClusteringBenchmark, CPU_vs_cuVS) {
  // Verify CUDA is available.
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available; skipping benchmark.";
  }

  // Define benchmark parameters.
  const int64_t num_vectors = 10000000;  // 10 million vectors
  const int64_t dim = 128;
  const int num_clusters = 10000;
  const int niter = 5;  // number of clustering iterations

  std::cout << "Benchmarking clustering with " << num_vectors << " vectors, dimension "
            << dim << ", and " << num_clusters << " clusters." << std::endl;

  // Create CPU tensors.
  auto options_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto options_int64_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor vectors_cpu = torch::randn({num_vectors, dim}, options_float_cpu).contiguous();
  torch::Tensor ids_cpu = torch::arange(0, num_vectors, options_int64_cpu).contiguous();
  
  // Create CUDA tensors.
  auto options_float_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto options_int64_cuda = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor vectors_cuda = vectors_cpu.to(options_float_cuda).contiguous();
  torch::Tensor ids_cuda = ids_cpu.to(options_int64_cuda).contiguous();

  // Warm up the GPU using the cuVS clustering (to remove initialization overhead).
  std::cout << "[DEBUG] Warming up GPU with cuVS clustering..." << std::endl;
  {
    auto dummy = ClusterWithCuVS(vectors_cuda, ids_cuda, 10, faiss::METRIC_L2);
    torch::cuda::synchronize();
  }
  std::cout << "[DEBUG] Warm-up complete." << std::endl;

  // Benchmark FAISS CPU clustering.
  std::cout << "[DEBUG] Starting FAISS CPU clustering benchmark..." << std::endl;
  auto start_cpu = high_resolution_clock::now();
  auto clustering_cpu = kmeans(vectors_cpu, ids_cpu, num_clusters, faiss::METRIC_L2, niter, false, torch::Tensor());
  auto end_cpu = high_resolution_clock::now();
  auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu).count();
  std::cout << "FAISS CPU clustering time: " << duration_cpu << " ms" << std::endl;

  // Benchmark cuVS GPU clustering.
  std::cout << "[DEBUG] Starting cuVS clustering benchmark..." << std::endl;
  auto start_cuvs = high_resolution_clock::now();
  auto clustering_cuvs = ClusterWithCuVS(vectors_cuda, ids_cuda, num_clusters, faiss::METRIC_L2);
  torch::cuda::synchronize();  // ensure GPU work is complete
  auto end_cuvs = high_resolution_clock::now();
  auto duration_cuvs = duration_cast<milliseconds>(end_cuvs - start_cuvs).count();
  std::cout << "cuVS clustering time: " << duration_cuvs << " ms" << std::endl;

  // Print summary.
  std::cout << "---------------------------" << std::endl;
  std::cout << "FAISS CPU clustering took " << duration_cpu << " ms." << std::endl;
  std::cout << "cuVS clustering took " << duration_cuvs << " ms." << std::endl;
  std::cout << "---------------------------" << std::endl;

  // Basic checks.
  EXPECT_GT(duration_cpu, 0);
  EXPECT_GT(duration_cuvs, 0);
}
