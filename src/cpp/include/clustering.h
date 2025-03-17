//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <common.h>

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter = 5,
                              std::vector<std::shared_ptr<arrow::Table>> data_frames = {},
                              Tensor initial_centroids = Tensor()
                              );

#endif //CLUSTERING_H
