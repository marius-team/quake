//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "clustering.h"

std::tuple<Tensor, vector<Tensor>, vector<Tensor> > kmeans(Tensor vectors,
                                                           Tensor ids,
                                                           int n_clusters,
                                                           faiss::MetricType metric_type,
                                                           int niter,
                                                           Tensor initial_centroids) {

    assert(vectors.size(0) >= n_clusters * 2);
    assert(vectors.size(0) == ids.size(0));

    // if the metric is IP, we need to normalize the vectors
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        vectors = vectors / vectors.norm(2, 1).unsqueeze(1);
    }

    if (!initial_centroids.defined()) {
        initial_centroids = vectors.index({torch::randperm(vectors.size(0)).narrow(0, 0, n_clusters)});
    }
    assert(vectors.size(1) == initial_centroids.size(1));
    assert(initial_centroids.size(0) == n_clusters);

    // normalize the initial centroids
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        initial_centroids = initial_centroids / initial_centroids.norm(2, 1).unsqueeze(1);
    }

    Tensor centroids = initial_centroids.clone();

    for (int i = 0; i < niter; i++) {
        // find the nearest centroid for each vector
        Tensor assignments;
        if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            auto dists = torch::mm(vectors, initial_centroids.t());
            assignments = std::get<1>(dists.max(1));
        } else {
            auto dists = torch::cdist(vectors, initial_centroids);
            assignments = std::get<1>(dists.min(1));
        }

        // update the centroids
        Tensor new_centroids = torch::zeros_like(initial_centroids);
        Tensor counts = torch::zeros({n_clusters}, torch::kInt64);
        new_centroids.scatter_add_(0, assignments.view({-1, 1}).expand({-1, vectors.size(1)}), vectors);
        counts.scatter_add_(0, assignments, torch::ones_like(assignments));

        new_centroids /= counts.unsqueeze(1);

        if (metric_type == faiss::METRIC_INNER_PRODUCT) {
            new_centroids = new_centroids / new_centroids.norm(2, 1).unsqueeze(1);
        }

        centroids = new_centroids;
    }

    // compute the final assignments
    Tensor assignments;
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        auto dists = torch::mm(vectors, centroids.t());
        assignments = std::get<1>(dists.max(1));
    } else {
        auto dists = torch::cdist(vectors, centroids);
        assignments = std::get<1>(dists.min(1));
    }

    // get the vectors and ids for each cluster
    vector<Tensor> cluster_vectors(n_clusters);
    vector<Tensor> cluster_ids(n_clusters);

    for (int i = 0; i < n_clusters; i++) {
        cluster_vectors[i] = vectors.index({assignments == i});
        cluster_ids[i] = ids.index({assignments == i});
    }

    return std::make_tuple(centroids, cluster_vectors, cluster_ids);
}
