#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>   // For at::cuda::getCurrentCUDAStream()
#include <raft/core/resources.hpp>   // RAFT resources (handle)
#include <raft/core/device_mdspan.hpp> // RAFT device view (make_device_matrix_view, etc.)
#include <cuvs/cluster/kmeans.hpp>   // cuVS k-means API

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <common.h>

// Optionally, if you have your own enum, you can use that instead of faiss::METRIC_L2/INNER_PRODUCT.
struct ClusteringResult {
  torch::Tensor centroids;
  // Each pair is (cluster_vectors, cluster_ids) for one cluster.
  std::vector<std::pair<torch::Tensor, torch::Tensor>> clusters;
};


inline ClusteringResult ClusterWithCuVS(const torch::Tensor& vectors,
                                          const torch::Tensor& ids,
                                          int64_t num_clusters,
                                          int metric = faiss::METRIC_L2) {
  using clock = std::chrono::high_resolution_clock;
  auto t0 = clock::now();
  
  // Validate input shapes and sizes.
  TORCH_CHECK(vectors.dim() == 2, "Input 'vectors' must be a 2D tensor");
  TORCH_CHECK(ids.dim() == 1 || (ids.dim() == 2 && ids.size(1) == 1),
              "Input 'ids' must be a 1D tensor or 2D with shape (N,1)");
  TORCH_CHECK(vectors.size(0) == ids.size(0), "Number of ids must match number of vectors");
  TORCH_CHECK(vectors.size(0) >= num_clusters, "Number of clusters cannot exceed number of points");
  
  // Move data and ids to GPU (if needed) and ensure contiguous memory.
  torch::Tensor data = vectors.to(torch::kCUDA, torch::kFloat32).contiguous();
  torch::Tensor id_dev = ids.to(torch::kCUDA).contiguous();
  int64_t n_samples  = data.size(0);
  int64_t n_features = data.size(1);
  
  auto t1 = clock::now();
  std::cout << "[DEBUG] Data transfer & contiguity: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;
  
  // If using inner-product (cosine), normalize input vectors.
  if (metric == faiss::METRIC_INNER_PRODUCT) {
    torch::Tensor norms = torch::sqrt((data * data).sum(1, /*keepdim=*/true));
    data = data / norms;
  }
  
  auto t2 = clock::now();
  std::cout << "[DEBUG] Data normalization: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
  
  // RAFT handle and stream setup.
  raft::resources handle;
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  raft::resource::set_cuda_stream(handle, cuda_stream);
  
  auto t3 = clock::now();
  std::cout << "[DEBUG] RAFT handle & stream setup: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms" << std::endl;
  
  // Wrap the input data in a RAFT device_matrix_view.
  float* data_ptr = data.data_ptr<float>();
  auto X_view = raft::make_device_matrix_view<const float, int>(data_ptr, (int)n_samples, (int)n_features);
  
  // Allocate output centroids on GPU.
  torch::Tensor centroids_tensor = torch::empty({num_clusters, n_features},
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  float* centroids_ptr = centroids_tensor.data_ptr<float>();
  auto centroids_view = raft::make_device_matrix_view<float, int>(centroids_ptr, (int)num_clusters, (int)n_features);
  
  auto t4 = clock::now();
  std::cout << "[DEBUG] Memory allocation for centroids & data wrapping: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms" << std::endl;
  
  // Set up k-means parameters.
  cuvs::cluster::kmeans::params params;
  params.n_clusters = (int)num_clusters;
  params.max_iter = 5;
  params.init = cuvs::cluster::kmeans::params::InitMethod::Random;

  // Prepare host-side scalars to capture inertia and iterations.
  float inertia = 0.0f;
  int iterations = 0;
  auto inertia_view = raft::make_host_scalar_view(&inertia);
  auto iter_view = raft::make_host_scalar_view(&iterations);
  
  // Run k-means clustering (fit).
  cuvs::cluster::kmeans::fit(handle, params, X_view, std::nullopt,
                             centroids_view, inertia_view, iter_view);
  
  auto t5 = clock::now();
  std::cout << "[DEBUG] cuVS fit (k-means clustering): " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count() << " ms" << std::endl;
  
  // If inner-product, renormalize centroids.
  if (metric == faiss::METRIC_INNER_PRODUCT) {
    torch::Tensor cent_norms = torch::sqrt((centroids_tensor * centroids_tensor).sum(1, /*keepdim=*/true));
    centroids_tensor.div_(cent_norms);
  }
  
  auto t6 = clock::now();
  std::cout << "[DEBUG] Centroid renormalization (if applicable): " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << " ms" << std::endl;
  
  // Allocate memory for labels and run prediction.
  torch::Tensor labels = torch::empty({n_samples}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  int* labels_ptr = labels.data_ptr<int>();
  auto labels_view = raft::make_device_vector_view<int, int>(labels_ptr, (int)n_samples);
  
  cuvs::cluster::kmeans::predict(handle, params, X_view, std::nullopt,
                                 centroids_view, labels_view, false,
                                 raft::make_host_scalar_view(&inertia));
  
  auto t7 = clock::now();
  std::cout << "[DEBUG] cuVS predict: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count() << " ms" << std::endl;
  
  // Synchronize the stream.
  raft::resource::sync_stream(handle);
  auto t8 = clock::now();
  std::cout << "[DEBUG] CUDA stream synchronization: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << " ms" << std::endl;
  
  // ----- Grouping (GPU vectorized) -----
  // Sort the labels and get the sorted indices.
  torch::Tensor sorted_tuple = std::get<1>(torch::sort(labels));  // sorted indices only, since sorted labels are not needed
  torch::Tensor sorted_labels = labels.index_select(0, sorted_tuple);
  
  // Reorder the data and ids using the sorted indices.
  torch::Tensor sorted_data = data.index_select(0, sorted_tuple);
  torch::Tensor sorted_ids = id_dev.index_select(0, sorted_tuple);
  
  // Compute per-cluster counts using torch::bincount.
  torch::Tensor counts = torch::bincount(sorted_labels.to(torch::kInt64), /*weights=*/{}, num_clusters);
  
  // Transfer counts to CPU and build a vector for split sizes.
  auto counts_cpu = counts.to(torch::kCPU);
  std::vector<int64_t> split_sizes(counts_cpu.data_ptr<int64_t>(), counts_cpu.data_ptr<int64_t>() + counts_cpu.numel());
  
  auto t9 = clock::now();
  std::cout << "[DEBUG] Sorting, counting, and preparing split sizes: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t9 - t8).count() << " ms" << std::endl;
  
  // Split the sorted data and ids into clusters.
  std::vector<torch::Tensor> cluster_vectors = torch::split(sorted_data, split_sizes, 0);
  std::vector<torch::Tensor> cluster_ids     = torch::split(sorted_ids, split_sizes, 0);
  
  auto t10 = clock::now();
  std::cout << "[DEBUG] Splitting into clusters: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t9).count() << " ms" << std::endl;
  
  // Build the final clustering result.
  std::vector<std::pair<torch::Tensor, torch::Tensor>> clusters;
  clusters.reserve(num_clusters);
  for (int i = 0; i < num_clusters; ++i) {
    clusters.emplace_back(cluster_vectors[i], cluster_ids[i]);
  }
  
  auto t_end = clock::now();
  std::cout << "[DEBUG] Total grouping time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t8).count() << " ms" << std::endl;
  
  ClusteringResult result;
  result.centroids = centroids_tensor;  // Shape: (num_clusters, n_features)
  result.clusters = std::move(clusters);
  
  std::cout << "[DEBUG] Total cuVS clustering time: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t0).count() << " ms" << std::endl;
  
  return result;
}
