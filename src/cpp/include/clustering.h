//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <torch/torch.h>
#include "faiss/index_io.h"
#include "faiss/Clustering.h"
#include <faiss/impl/platform_macros.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexRefine.h>
#include <chrono>
#include <tuple>
#include "simsimd/simsimd.h"

using torch::Tensor;
using std::vector;
using std::shared_ptr;
using std::chrono::high_resolution_clock;

std::tuple<Tensor, vector<Tensor>, vector<Tensor>> kmeans(Tensor vectors,
                                                           Tensor ids,
                                                           int n_clusters,
                                                           faiss::MetricType metric_type,
                                                           int niter=5,
                                                           Tensor initial_centroids = Tensor());

#endif //CLUSTERING_H
