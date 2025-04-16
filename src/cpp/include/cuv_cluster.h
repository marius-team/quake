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

}
