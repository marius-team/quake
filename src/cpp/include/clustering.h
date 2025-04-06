//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <common.h>

class IndexPartition;

/**
 * @brief Clusters vectors into partitions using k-means.
 *
 * Uses the faiss::Clustering class to cluster vectors into n_clusters partitions.
 *
 * @param vectors The vectors to cluster.
 * @param ids The IDs of the vectors.
 * @param n_clusters The number of clusters to create.
 * @param metric_type The metric type to use for clustering.
 * @param niter The number of iterations to run k-means.
 * @param initial_centroids The initial centroids to use for k-means.
 */
shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter = 5,
                              std::shared_ptr<arrow::Table> attributes_table = nullptr,
                              Tensor initial_centroids = Tensor()
                              );


/**
 * @brief Refines partitions using k-means.
 *
 * Uses batched_scan_list to reassign each vector (from every partition) to its nearest
 * centroid and then rebuilds the partitions and centroids.
 *
 * @param centroids  The current centroids as an IndexPartition.
 * @param index_partitions The current partitions.
 * @param metric The metric type to use for clustering.
 * @param refinement_iterations If 0, only reassign; otherwise, update centroids iteratively.
 *
 * @return A tuple with (updated centroids, new refined partitions)
 */
tuple<Tensor, vector<shared_ptr<IndexPartition>>> kmeans_refine_partitions(
    Tensor centroids,
    vector<shared_ptr<IndexPartition>> index_partitions,
    MetricType metric,
    int refinement_iterations = 0);

#endif //CLUSTERING_H
