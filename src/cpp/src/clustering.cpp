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
#include <c10/cuda/CUDAStream.h>
#include <raft/core/resources.hpp>   // RAFT resources (handle)
#include <raft/core/device_mdspan.hpp> // RAFT device view (make_device_matrix_view, etc.)
#include <cuvs/cluster/kmeans.hpp>   // cuVS k-means API

shared_ptr<Clustering> kmeans_cuvs_sample_and_predict(
    Tensor vectors, Tensor ids,
    shared_ptr<IndexBuildParams> build_params) {

    int num_clusters       = build_params->nlist;
    int niter              = build_params->niter;
    int gpu_batch_size     = build_params->gpu_batch_size;
    int gpu_sample_size    = build_params->gpu_sample_size;
    MetricType metric      = str_to_metric_type(build_params->metric);


    TORCH_CHECK(vectors.dim() == 2, "vectors must be [N,D]");
    TORCH_CHECK(ids.dim() == 1,      "ids must be [N]");
    int64_t N = vectors.size(0), D = vectors.size(1);

    gpu_sample_size = std::min(gpu_sample_size, (int) N);
    gpu_batch_size = std::min(gpu_batch_size, (int) N);

    TORCH_CHECK(gpu_sample_size > 0 && gpu_sample_size <= N,
              "invalid sample_size");

    // 1) pin + normalize if needed
    Tensor cpu_pts = vectors.contiguous().pin_memory();
    if (metric == faiss::METRIC_INNER_PRODUCT) {
    auto norms = cpu_pts.norm(2, 1, true);
    cpu_pts = cpu_pts.div(norms);
    }

    // 2) choose a random sample of indices
    auto perm     = torch::randperm(N, torch::kLong);
    auto samp_idx = perm.slice(0, 0, gpu_sample_size);
    Tensor samp_pts = cpu_pts.index_select(0, samp_idx);

    // 3) move sample to GPU
    Tensor samp_gpu = samp_pts.to(torch::kCUDA, /*non_blocking=*/true).contiguous();

    // 4) prepare RAFT handle & cuVS params
    raft::resources handle;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    raft::resource::set_cuda_stream(handle, stream);

    cuvs::cluster::kmeans::params params;
    params.n_clusters = num_clusters;
    params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;
    params.max_iter   = niter;

    // 5) allocate centroids on GPU
    Tensor cent_gpu = torch::empty({num_clusters, D},
                                 torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCUDA))
                    .contiguous();

    // 6) run fit on just the sample
    {
    // host scalars
    float inertia    = 0.0f;
    int   actual_iter= 0;
    auto host_inertia = raft::make_host_scalar_view(&inertia);
    auto host_iter    = raft::make_host_scalar_view(&actual_iter);

    auto X_view = raft::make_device_matrix_view<const float,int>(
                     samp_gpu.data_ptr<float>(),
                     (int) gpu_sample_size, (int)D);
    auto C_view = raft::make_device_matrix_view<float,int>(
                     cent_gpu.data_ptr<float>(),
                     num_clusters, (int)D);

    cuvs::cluster::kmeans::fit(
      handle, params,
      X_view,
      std::nullopt,
      C_view,
      host_inertia,
      host_iter
    );
    }

    Tensor all_labels = torch::empty({N}, torch::kLong);

    auto predict_fn = [&](Tensor batch_cpu, int64_t off) {
    int64_t bs = batch_cpu.size(0);

    // allocate exactly bs labels on the GPU
    Tensor labels32 = torch::empty(
        {bs},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA));

    Tensor batch_gpu = batch_cpu.to(torch::kCUDA, /*NB=*/true)
                                 .contiguous();

    auto Xv = raft::make_device_matrix_view<const float,int>(
                batch_gpu.data_ptr<float>(),
                (int)bs, (int)D);
    auto Lv = raft::make_device_vector_view<int,int>(
                labels32.data_ptr<int>(), (int)bs);

    float pred_inertia = 0.0f;
    auto host_pred     = raft::make_host_scalar_view(&pred_inertia);

    cuvs::cluster::kmeans::predict(
      handle, params,
      Xv,
      std::nullopt,
      raft::make_device_matrix_view<float,int>(
        cent_gpu.data_ptr<float>(),
        num_clusters, (int)D),
      Lv,
      false,
      host_pred
    );

    // now safe to copy back exactly bs elements
    all_labels.narrow(0, off, bs)
              .copy_(
                labels32.to(torch::kLong)
                        .to(torch::kCPU)
              );
    };

    // predict the sample slice
    predict_fn(samp_pts, /*off=*/0);

    // predict the rest
    for (int64_t off = 0; off < N; off += gpu_batch_size) {
    int64_t bs = std::min<int64_t>(gpu_batch_size, N - off);
    if (off < gpu_sample_size) {
      if (off + bs <= gpu_sample_size) {
        continue;
      } else {
        int64_t overlap = gpu_sample_size - off;
        Tensor rest_chunk = cpu_pts.slice(0, off + overlap, off + bs);
        predict_fn(rest_chunk, /*off=*/off + overlap);
        continue;
      }
    }
    Tensor rest_chunk = cpu_pts.slice(0, off, off + bs);
    predict_fn(rest_chunk, /*off=*/off);
    }
    // 8) group on CPU
    Tensor sorted_lbl, sorted_idx;
    std::tie(sorted_lbl, sorted_idx) = torch::sort(all_labels);
    Tensor sorted_vecs = vectors.index_select(0, sorted_idx);
    Tensor sorted_ids  = ids.index_select(0, sorted_idx);

    Tensor counts = torch::bincount(sorted_lbl, /*weights=*/{}, num_clusters);
    auto cnt_cpu = counts.to(torch::kCPU);
    std::vector<int64_t> split_sizes(
    cnt_cpu.data_ptr<int64_t>(),
    cnt_cpu.data_ptr<int64_t>() + num_clusters
    );

    auto cluster_vecs = torch::split(sorted_vecs, split_sizes, 0);
    auto cluster_ids  = torch::split(sorted_ids,   split_sizes, 0);

    auto out = std::make_shared<Clustering>();
    out->centroids     = cent_gpu.cpu().contiguous();
    out->partition_ids = torch::arange(num_clusters, torch::kLong);
    out->vectors       = std::move(cluster_vecs);
    out->vector_ids    = std::move(cluster_ids);

    return out;
}
#endif

shared_ptr<Clustering> kmeans_cpu(Tensor vectors,
                              Tensor ids,
                              shared_ptr<IndexBuildParams> build_params,
                              Tensor /* initial_centroids */) {
    // Ensure enough vectors are available and sizes match.
    assert(vectors.size(0) >= build_params->nlist * 2);
    assert(vectors.size(0) == ids.size(0));

    MetricType metric_type = str_to_metric_type(build_params->metric);

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
    cp.niter = build_params->niter;

    faiss::Clustering clus(d, build_params->nlist, cp);
    clus.train(n, vectors.data_ptr<float>(), *index_ptr);

    // Retrieve centroids as a torch Tensor.
    Tensor centroids = torch::from_blob(clus.centroids.data(), {build_params->nlist, d}, torch::kFloat32).clone();
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
    Tensor counts_tensor = torch::bincount(sorted_assignments, /*weights=*/{}, build_params->nlist);
    // Ensure counts are on CPU to extract split sizes.
    counts_tensor = counts_tensor.to(torch::kCPU);
    // Convert counts tensor to std::vector<int64_t>
    std::vector<int64_t> counts_vector(counts_tensor.data_ptr<int64_t>(),
                                        counts_tensor.data_ptr<int64_t>() + counts_tensor.numel());

    // Split the sorted vectors and sorted ids into clusters in one call.
    vector<Tensor> cluster_vectors = torch::split(sorted_vectors, counts_vector, 0);
    vector<Tensor> cluster_ids = torch::split(sorted_ids, counts_vector, 0);

    Tensor partition_ids = torch::arange(build_params->nlist, torch::kInt64);

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
                              shared_ptr<IndexBuildParams> build_params,
                              Tensor /* initial_centroids */) {
    if (build_params->use_gpu) {
    #ifdef QUAKE_ENABLE_GPU
        return kmeans_cuvs_sample_and_predict(
            vectors,
            ids,
            build_params);
    #else
            throw std::runtime_error("GPU support is not enabled. Please compile with QUAKE_ENABLE_GPU.");
    #endif
    } else {
        return kmeans_cpu(vectors, ids, build_params);
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
