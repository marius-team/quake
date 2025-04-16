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

#ifdef QUAKE_ENABLE_GPU
shared_ptr<Clustering> kmeans_cuvs(Tensor vectors,
    Tensor ids,
    int num_clusters,
    MetricType metric,
    int niter,
    Tensor initial_centroids) {
    // Validate input shapes and sizes.
    TORCH_CHECK(vectors.dim() == 2, "Input 'vectors' must be a 2D tensor");
    TORCH_CHECK(ids.dim() == 1 || (ids.dim() == 2 && ids.size(1) == 1),
              "Input 'ids' must be a 1D tensor or 2D with shape (N,1)");
    TORCH_CHECK(vectors.size(0) == ids.size(0), "Number of ids must match number of vectors");
    TORCH_CHECK(vectors.size(0) >= num_clusters, "Number of clusters cannot exceed number of points");

    int64_t n_samples  = data.size(0);
    int64_t n_features = data.size(1);

    // If using inner-product (cosine), normalize input vectors.
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        Tensor norms = torch::sqrt((data * data).sum(1, /*keepdim=*/true));
        data = data / norms;
    }

    // RAFT handle and stream setup.
    raft::resources handle;
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    raft::resource::set_cuda_stream(handle, cuda_stream);

    // Wrap the input data in a RAFT device_matrix_view.
    float* data_ptr = data.data_ptr<float>();
    auto X_view = raft::make_host_matrix_view<const float, int>(data_ptr, (int)n_samples, (int)n_features);

    // Allocate output centroids on GPU.
    Tensor centroids_tensor = torch::empty({num_clusters, n_features},
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float* centroids_ptr = centroids_tensor.data_ptr<float>();
    auto centroids_view = raft::make_device_matrix_view<float, int>(centroids_ptr, (int)num_clusters, (int)n_features);

    // Set up k-means parameters.
    cuvs::cluster::kmeans::params params;
    params.n_clusters = (int)num_clusters;
    params.max_iter = niter;
    params.init = cuvs::cluster::kmeans::params::InitMethod::Random;

    // Prepare host-side scalars to capture inertia and iterations.
    float inertia = 0.0f;
    int iterations = 0;
    auto inertia_view = raft::make_host_scalar_view(&inertia);
    auto iter_view = raft::make_host_scalar_view(&iterations);

    // Run k-means clustering (fit).
    cuvs::cluster::kmeans::fit(handle, params, X_view, std::nullopt,
                             centroids_view, inertia_view, iter_view);

    // If inner-product, renormalize centroids.
    if (metric == faiss::METRIC_INNER_PRODUCT) {
    Tensor cent_norms = torch::sqrt((centroids_tensor * centroids_tensor).sum(1, /*keepdim=*/true));
    centroids_tensor.div_(cent_norms);
    }

    // Allocate memory for labels and run prediction.
    Tensor labels = torch::empty({n_samples}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    int* labels_ptr = labels.data_ptr<int>();
    auto labels_view = raft::make_device_vector_view<int, int>(labels_ptr, (int)n_samples);

    cuvs::cluster::kmeans::predict(handle, params, X_view, std::nullopt,
                                 centroids_view, labels_view, false,
                                 raft::make_host_scalar_view(&inertia));

    // Synchronize the stream.
    raft::resource::sync_stream(handle);

    // ----- Grouping (GPU vectorized) -----
    // Sort the labels and get the sorted indices.
    Tensor sorted_tuple = std::get<1>(torch::sort(labels));  // sorted indices only, since sorted labels are not needed
    Tensor sorted_labels = labels.index_select(0, sorted_tuple);

    // Reorder the data and ids using the sorted indices.
    Tensor sorted_data = data.index_select(0, sorted_tuple);
    Tensor sorted_ids = id_dev.index_select(0, sorted_tuple);

    // Compute per-cluster counts using torch::bincount.
    Tensor counts = torch::bincount(sorted_labels.to(torch::kInt64), /*weights=*/{}, num_clusters);

    // Transfer counts to CPU and build a vector for split sizes.
    auto counts_cpu = counts.to(torch::kCPU);
    std::vector<int64_t> split_sizes(counts_cpu.data_ptr<int64_t>(), counts_cpu.data_ptr<int64_t>() + counts_cpu.numel());

    // Split the sorted data and ids into clusters.
    vector<Tensor> cluster_vectors = torch::split(sorted_data, split_sizes, 0);
    vector<Tensor> cluster_ids     = torch::split(sorted_ids, split_sizes, 0);

    Tensor partition_ids = torch::arange(num_clusters, torch::kInt64);

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids_tensor;
    clustering->partition_ids = partition_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    return clustering;
}
#endif

shared_ptr<Clustering> kmeans_cpu(Tensor vectors,
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

    faiss::Index* index_ptr = nullptr;
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

    // Sort assignments and select corresponding vectors and ids.
    Tensor sorted_assignments, sorted_indices;
    std::tie(sorted_assignments, sorted_indices) = torch::sort(assignments);
    Tensor sorted_vectors = vectors.index_select(0, sorted_indices);
    Tensor sorted_ids = ids.index_select(0, sorted_indices);

    // Compute counts per cluster using bincount.
    Tensor counts_tensor = torch::bincount(sorted_assignments, /*weights=*/{}, n_clusters);
    // Ensure counts are on CPU to extract split sizes.
    counts_tensor = counts_tensor.to(torch::kCPU);
    // Convert counts tensor to std::vector<int64_t>
    std::vector<int64_t> counts_vector(counts_tensor.data_ptr<int64_t>(),
                                        counts_tensor.data_ptr<int64_t>() + counts_tensor.numel());

    // Split the sorted vectors and sorted ids into clusters in one call.
    vector<Tensor> cluster_vectors = torch::split(sorted_vectors, counts_vector, 0);
    vector<Tensor> cluster_ids = torch::split(sorted_ids, counts_vector, 0);

    Tensor partition_ids = torch::arange(n_clusters, torch::kInt64);

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = partition_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;

    delete index_ptr;

    return clustering;
}

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter,
                              bool use_gpu /*=false*/,
                              Tensor /* initial_centroids */) {
    if (use_gpu) {
    #ifdef QUAKE_ENABLE_GPU
            return kmeans_cuvs(vectors, ids, n_clusters, metric_type, niter);
    #elif
            throw std::runtime_error("GPU support is not enabled. Please compile with QUAKE_ENABLE_GPU.");
    #endif
    } else {
        return kmeans_cpu(vectors, ids, n_clusters, metric_type, niter);
    }
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
