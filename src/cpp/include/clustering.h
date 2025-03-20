//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <common.h>

class IndexPartition;

typedef tuple<Tensor, Tensor, shared_ptr<IndexPartition>> IndexPartitionClustering;

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter = 5,
                              Tensor initial_centroids = Tensor());


tuple<Tensor, vector<shared_ptr<IndexPartition>>> kmeans_refine_partitions(
    Tensor centroids,
    vector<shared_ptr<IndexPartition>> index_partitions,
    int refinement_iterations = 0);

#endif //CLUSTERING_H
