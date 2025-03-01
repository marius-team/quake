//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "clustering.h"
#include <faiss/IndexFlat.h>
#include "faiss/Clustering.h"

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter,
                              Tensor /* initial_centroids */) {
    // Ensure enough vectors are available and sizes match.
    assert(vectors.size(0) >= n_clusters * 2);
    assert(vectors.size(0) == ids.size(0));

    // Normalize vectors for inner product
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        vectors = vectors / vectors.norm(2, 1).unsqueeze(1);

    int n = vectors.size(0);
    int d = vectors.size(1);

    // Create a flat index appropriate to the metric.
    faiss::IndexFlat* index_ptr = nullptr;
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        index_ptr = new faiss::IndexFlatIP(d);
    else
        index_ptr = new faiss::IndexFlatL2(d);

    faiss::ClusteringParameters cp;
    cp.niter = niter;

    faiss::Clustering clus(d, n_clusters, cp);
    clus.train(n, vectors.data_ptr<float>(), *index_ptr);

    // Retrieve centroids as a torch Tensor.
    Tensor centroids = torch::from_blob(clus.centroids.data(), {n_clusters, d}, torch::kFloat32).clone();
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        centroids = centroids / centroids.norm(2, 1).unsqueeze(1);

    // Use the index to assign each vector to its nearest centroid.
    std::vector<idx_t> assign_vec(n);
    std::vector<float> distance_vec(n);
    index_ptr->search(n, vectors.data_ptr<float>(), 1, distance_vec.data(), assign_vec.data());
    Tensor assignments = torch::from_blob(assign_vec.data(), {n}, torch::kInt64).clone();

    // Partition vectors and ids by cluster.
    vector<Tensor> cluster_vectors(n_clusters);
    vector<Tensor> cluster_ids(n_clusters);
    for (int i = 0; i < n_clusters; i++) {
        cluster_vectors[i] = vectors.index({assignments == i});
        cluster_ids[i] = ids.index({assignments == i});
    }
    Tensor partition_ids = torch::arange(n_clusters, torch::kInt64);

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = partition_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    delete index_ptr;
    return clustering;
}