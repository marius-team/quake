//
// Created by Jason on 9/20/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include "clustering.h"
#include <faiss/IndexFlat.h>
#include "faiss/Clustering.h"
#include <arrow/compute/api_vector.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>

shared_ptr<Clustering> kmeans(Tensor vectors,
                              Tensor ids,
                              int n_clusters,
                              MetricType metric_type,
                              int niter,
                              std::shared_ptr<arrow::Table> attributes_table,
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
    vector<shared_ptr<arrow::Table>> cluster_attributes_tables(n_clusters);

    for (int i = 0; i < n_clusters; i++) {
        auto mask = (assignments == i);
        cluster_vectors[i] = vectors.index({mask});
        cluster_ids[i] = ids.index({mask});

        if(attributes_table == nullptr) {
            cluster_attributes_tables[i] = nullptr;
            continue;
        }

        auto cluster_ids_tensor = cluster_ids[i];  // Assuming this is a tensor with IDs
        std::vector<int64_t> cluster_ids_vec(cluster_ids_tensor.data<int64_t>(), 
                                             cluster_ids_tensor.data<int64_t>() + cluster_ids_tensor.numel());

        // Convert to Arrow Array
        arrow::Int64Builder id_builder;
        id_builder.AppendValues(cluster_ids_vec);
        std::shared_ptr<arrow::Array> cluster_ids_array;
        id_builder.Finish(&cluster_ids_array);

        // Get the "id" column from the attributes table
        std::shared_ptr<arrow::ChunkedArray> id_column = attributes_table->GetColumnByName("id");
        
        auto lookup_options = std::make_shared<arrow::compute::SetLookupOptions>(cluster_ids_array);
        // Apply set lookup to filter rows
        auto result = arrow::compute::CallFunction(
            "index_in", 
            {id_column->chunk(0)}, 
            lookup_options.get()
        );
        
        auto index_array = std::static_pointer_cast<arrow::Int32Array>(result->make_array());

        auto mask_result = arrow::compute::CallFunction(
            "not_equal",
            {index_array, arrow::MakeScalar(-1)}
        );

        // Convert result to a Boolean mask
        auto mask_table = std::static_pointer_cast<arrow::BooleanArray>(mask_result->make_array());

        // Filter the table using the mask
        auto filtered_table_result = arrow::compute::Filter(attributes_table, mask_table);

        cluster_attributes_tables[i] = filtered_table_result->table();
        std::cout<<cluster_attributes_tables[i]->ToString()<<std::endl;
    }


    Tensor partition_ids = torch::arange(n_clusters, torch::kInt64);

    shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->centroids = centroids;
    clustering->partition_ids = partition_ids;
    clustering->vectors = cluster_vectors;
    clustering->vector_ids = cluster_ids;
    clustering->attributes_tables = cluster_attributes_tables;

    delete index_ptr;
    return clustering;
}