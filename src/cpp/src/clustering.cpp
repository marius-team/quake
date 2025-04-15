//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "clustering.h"
#include <faiss/IndexFlat.h>
#include "faiss/Clustering.h"
#include "index_partition.h"
#include <list_scanning.h>

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter,
                              bool use_gpu /*=false*/,
                              Tensor /* initial_centroids */) {
    // Ensure enough vectors are available and sizes match.
    assert(vectors.size(0) >= n_clusters * 2);
    assert(vectors.size(0) == ids.size(0));

    // Normalize vectors for inner product
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
        vectors = vectors / vectors.norm(2, 1).unsqueeze(1);

    int n = vectors.size(0);
    int d = vectors.size(1);

    faiss::Index* index_ptr = nullptr;

    if (use_gpu) {
        // Check if GPU resources are available.
        #ifdef FAISS_ENABLE_GPU
        faiss::gpu::StandardGpuResources gpu_res;
        if (metric_type == faiss::METRIC_INNER_PRODUCT)
            index_ptr = new faiss::gpu::GpuIndexFlatIP(&gpu_res, d);
        else
            index_ptr = new faiss::gpu::GpuIndexFlatL2(&gpu_res, d);
        #else
        throw std::runtime_error("GPU resources are not available. Please compile with FAISS_ENABLE_GPU.");
        #endif
    } else {
        if (metric_type == faiss::METRIC_INNER_PRODUCT)
            index_ptr = new faiss::IndexFlatIP(d);
        else
            index_ptr = new faiss::IndexFlatL2(d);
    }

    faiss::ClusteringParameters cp;
    cp.niter = niter;

    faiss::Clustering clus(d, n_clusters, cp);
    clus.t
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

tuple<Tensor, vector<shared_ptr<IndexPartition> >> kmeans_refine_partitions(
    Tensor centroids,
    vector<shared_ptr<IndexPartition>> partitions,
    MetricType metric,
    int refinement_iterations) {

    // Determine number of clusters and dimension.
    int n_clusters = centroids.size(0);
    int d = centroids.size(1);

    // Run for the desired number of iterations (if refinement_iterations==0, do one pass).
    int iterations = (refinement_iterations > 0) ? refinement_iterations : 1;

    Tensor centroid_sums = torch::zeros_like(centroids);
    Tensor centroid_counts = torch::zeros({n_clusters}, torch::kInt64);
    auto centroid_sums_accessor = centroid_sums.accessor<float, 2>();
    auto centroid_counts_accessor = centroid_counts.accessor<int64_t, 1>();

    vector<shared_ptr<IndexPartition>> prev_partitions = partitions;
    vector<shared_ptr<IndexPartition>> new_partitions;

    for (int iter = 0; iter < iterations; iter++) {

        if (iter > 0) {
            centroids = centroid_sums / centroid_counts.unsqueeze(1).to(torch::kFloat32);
        }

        // Reset accumulators.
        centroid_sums.zero_();
        centroid_counts.zero_();
        new_partitions.clear();
        new_partitions.resize(n_clusters);

        for (int i = 0; i < n_clusters; i++) {
            new_partitions[i] = make_shared<IndexPartition>();
            new_partitions[i]->set_code_size(partitions[0]->code_size_);
            new_partitions[i]->resize(10);
        }

        float *centroids_ptr = centroids.data_ptr<float>();

        // Process each existing partition.
        for (auto &part: partitions) {
            int64_t nvec = part->num_vectors_;
            if (nvec <= 0) continue;

            float *part_vecs = (float *) part->codes_;
            int64_t *part_vec_ids = part->ids_;

            // Create batched TopK buffers (k=1 for nearest centroid).
            vector<shared_ptr<TopkBuffer> > buffers = create_buffers(nvec, 1, false);

            // Use batched_scan_list to get nearest centroid for each vector.
            batched_scan_list(part_vecs,
                              centroids_ptr,
                              nullptr,
                              nvec,
                              n_clusters,
                              d,
                              buffers,
                              metric);

            // For each vector in this partition, determine its assignment.
            for (int i = 0; i < nvec; i++) {
                vector<int64_t> assign = buffers[i]->get_topk_indices(); // top1 assignment
                int assigned_cluster = assign[0];

                // Update accumulators.
                float *vec_ptr = part_vecs + i * d;
                int64_t *vec_id = part_vec_ids + i;

                for (int j = 0; j < d; j++) {
                    centroid_sums_accessor[assigned_cluster][j] += vec_ptr[j];
                }
                centroid_counts_accessor[assigned_cluster]++;

                new_partitions[assigned_cluster]->append(1, vec_id, (uint8_t *) vec_ptr);
            }
        } // end for each partition
        std::move(new_partitions.begin(), new_partitions.end(), partitions.begin());
    } // end iterations

    return std::make_tuple(centroids, partitions);
}
