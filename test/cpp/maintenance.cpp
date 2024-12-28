// //
// // Created by Jason on 10/7/24.
// // Prompt for GitHub Copilot:
// // - Conform to the google style guide
// // - Use descriptive variable names
//
//
// #include <gtest/gtest.h>
// #include "dynamic_ivf.h"
// #include "list_scanning.h"
//
// #include <torch/torch.h>
// #include <faiss/IndexFlat.h>
// #include <faiss/MetricType.h>
//
// #include <vector>
// #include <memory>
// #include <iostream>
//
// #ifdef __linux__
// #include <bits/stdc++.h>
// #endif
//
// using torch::Tensor;
// using std::vector;
// using std::shared_ptr;
//
// // Test fixture for MaintenanceTest
// class MaintenanceTest : public ::testing::Test {
// protected:
//     int dimension = 20;
//     int nlist = 100;
//     int nprobe = 20;
//     int num_codebooks = 8;
//     int code_size = 8;
//     int num_vectors = 1000000;
//     int num_queries = 100;
//     int k = 10;
//     Tensor data_vectors;
//     Tensor data_ids;
//     Tensor query_vectors;
//     Tensor ground_truth;
//     Tensor ground_truth_dists;
//     faiss::MetricType metric = faiss::METRIC_L2;
//
//     void SetUp() override {
//         // Generate random data and queries
//         data_vectors = torch::randn({num_vectors, dimension}, torch::kFloat32);
//         data_ids = torch::arange(num_vectors, torch::kInt64);
//         query_vectors = torch::randn({num_queries, dimension}, torch::kFloat32) * .1;
//
//         // normalize the vectors and queries
//         data_vectors = data_vectors / data_vectors.norm(2, 1, true);
//         query_vectors = query_vectors / query_vectors.norm(2, 1, true);
//
//         // generate ground truth
//         if (metric == faiss::METRIC_INNER_PRODUCT) {
//             Tensor dists = torch::mm(query_vectors, data_vectors.t());
//             auto topk = torch::topk(dists, k, 1, true, true);
//             ground_truth = std::get<1>(topk);
//             ground_truth_dists = std::get<0>(topk);
//         } else {
//             Tensor dists = torch::cdist(query_vectors, data_vectors);
//             auto topk = torch::topk(dists, k, 1, false, true);
//             ground_truth = std::get<1>(topk);
//             ground_truth_dists = std::get<0>(topk);
//         }
//     }
// };
//
// TEST_F(MaintenanceTest, InsertDeleteAndQuery) {
//     int64_t initial_size = 10000;
//     int64_t batch_size = 10000;
//     int64_t nlist = 50;
//
//     // Build initial index and do a search without anything
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 1, -1, -1, false);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//     auto result = index->search(query_vectors[0].unsqueeze(0), nlist, k, .75);
//     index->maintenance();
//
//     int64_t offset = initial_size;
//     int64_t delete_offset = 0;
//     while (offset < data_vectors.size(0)) {
//         int64_t end = std::min(offset + batch_size, data_vectors.size(0));
//         index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end), false);
//         offset = end;
//
//         Tensor delete_ids = data_ids.slice(0, delete_offset, delete_offset + batch_size);
//         index->remove(delete_ids, false);
//         delete_offset += batch_size;
//
//         // check that the size is correct
//         ASSERT_EQ(index->ntotal(), offset - delete_offset);
//
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .75);
//         }
//
//         index->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, InsertAndQuery) {
//     int64_t initial_size = 10000;
//     int64_t batch_size = 1000;
//     int64_t nlist = 10;
//
//     // Build initial index and do a search without anything
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 1, -1, -1, false);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//     auto result = index->search(query_vectors[0].unsqueeze(0), nlist, k, .75);
//     index->maintenance();
//
//     int64_t offset = initial_size;
//     while (offset < data_vectors.size(0)) {
//         int64_t end = std::min(offset + batch_size, data_vectors.size(0));
//         index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end), false);
//         offset = end;
//
//         // check that the size is correct
//         ASSERT_EQ(index->ntotal(), offset);
//
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .75);
//         }
//
//         index->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, InsertAndQueryLire) {
//     int64_t initial_size = 10000;
//     int64_t batch_size = 1000;
//     int64_t nlist = 10;
//
//     // Build initial index and do a search without anything
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 1, -1, -1, false);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//     auto result = index->search(query_vectors[0].unsqueeze(0), nlist, k, .75);
//
//     MaintenancePolicyParams params;
//     params.maintenance_policy = "lire";
//     index->set_maintenance_policy_params(params);
//
//     index->maintenance();
//
//     int64_t offset = initial_size;
//     while (offset < data_vectors.size(0)) {
//         int64_t end = std::min(offset + batch_size, data_vectors.size(0));
//         index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end), false);
//         offset = end;
//
//         // check that the size is correct
//         ASSERT_EQ(index->ntotal(), offset);
//
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .75);
//         }
//
//         index->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, InsertAndQueryCentroidSearch) {
//     int64_t initial_size = 1000;
//     int64_t batch_size = 100;
//     int64_t nlist = 10;
//
//     // Build initial index and do a search without anything
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 1, -1, -1, false, false, false, false, true);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//     auto parent_index = index->parent_;
//     auto result = index->search(query_vectors[0].unsqueeze(0), nlist, k, .75);
//     index->maintenance();
//
//     int64_t offset = initial_size;
//     while (offset < data_vectors.size(0)) {
//         int64_t end = std::min(offset + batch_size, data_vectors.size(0));
//         index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end), false);
//         offset = end;
//
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//
//             // Perform the query twice once using faiss and once using search all centroids
//             bool prev_use_centroid_val = parent_index->use_centroid_workers_;
//             auto first_result = index->search_one(query, k, .75);
//             parent_index->use_centroid_workers_ = !parent_index->use_centroid_workers_;
//             auto second_result = index->search_one(query, k, .75);
//             parent_index->use_centroid_workers_ = prev_use_centroid_val;
//
//             // Check that the returned IDs are identical
//             auto first_ids = std::get<0>(first_result); auto second_ids = std::get<0>(second_result);
//             ASSERT_TRUE(torch::equal(first_ids, second_ids))
//                 << "Mismatch in returned IDS for query " << i << " and offset " << offset;
//
//             // Check that the returned distances are identical within a small tolerance
//             auto first_dists = std::get<0>(first_result); auto second_dists = std::get<0>(second_result);
//             ASSERT_TRUE(torch::allclose(first_dists, second_dists, /*atol=*/1e-6, /*rtol=*/1e-4))
//                 << "Mismatch in returned IDS for query " << i << " and offset " << offset;
//         }
//
//         index->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, QueryOnly) {
//     int64_t initial_size = 1000000;
//     int64_t nlist = 100;
//
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//
//     // set maintenance policy
//     MaintenancePolicyParams params;
//     params.window_size = 1000;
//     params.refinement_radius = 25;
//     params.min_partition_size = 32;
//     params.alpha = .75;
//     params.enable_split_rejection = true;
//     params.enable_delete_rejection = true;
//     params.delete_threshold_ns = 1.0;
//     params.split_threshold_ns = 20.0;
//     index->maintenance_policy_->set_params(params);
//
//     int tmp_nprobe = 10;
//     for (int64_t j = 0; j < 250; j++) {
//         vector<Tensor> ids_list;
//         vector<Tensor> dists_list;
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search(query, tmp_nprobe, k);
//             auto ids = std::get<0>(result);
//             auto dists = torch::sqrt(std::get<1>(result));
//             ids_list.push_back(ids.unsqueeze(0));
//             dists_list.push_back(dists.unsqueeze(0));
//         }
//         auto ids = torch::cat(ids_list, 0);
//         auto dists = torch::cat(dists_list, 0);
//         Tensor recalls = calculate_recall(ids, ground_truth);
//         std::cout << "Recall: " << torch::stack(recalls).mean().item<float>() << std::endl;
//         MaintenanceTimingInfo timing_info = index->maintenance_policy_->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, QueryOnlyMultiLevel) {
//     int64_t initial_size = 10000;
//     int64_t nlist = 1000;
//
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//     index->add_level(nlist);
//
//     for (int64_t j = 0; j < 100; j++) {
//         vector<Tensor> ids_list;
//         vector<Tensor> dists_list;
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .9);
//             auto ids = std::get<0>(result);
//             auto dists = torch::sqrt(std::get<1>(result));
//             ids_list.push_back(ids.unsqueeze(0));
//             dists_list.push_back(dists.unsqueeze(0));
//         }
//         auto ids = torch::cat(ids_list, 0);
//         auto dists = torch::cat(dists_list, 0);
//         Tensor recalls = calculate_recall(ids, ground_truth);
//         std::cout << "Iteration " << j << " got recall of " << torch::stack(recalls).mean().item<float>() << std::endl;
//         index->maintenance_policy_->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, SaveAndLoadQueryOnly) {
//     int64_t initial_size = 10000;
//     int64_t nlist = 100;
//
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//
//     index->save("test_index.faiss");
//     auto loaded_index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     loaded_index->load("test_index.faiss");
//     index = loaded_index;
//
//     for (int64_t j = 0; j < 100; j++) {
//         vector<Tensor> ids_list;
//         vector<Tensor> dists_list;
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .9);
//             auto ids = std::get<0>(result);
//             auto dists = torch::sqrt(std::get<1>(result));
//             ids_list.push_back(ids.unsqueeze(0));
//             dists_list.push_back(dists.unsqueeze(0));
//         }
//         auto ids = torch::cat(ids_list, 0);
//         auto dists = torch::cat(dists_list, 0);
//         Tensor recalls = calculate_recall(ids, ground_truth);
//         std::cout << "Recall: " << torch::stack(recalls).mean().item<float>() << std::endl;
//         index->maintenance_policy_->maintenance();
//     }
// }
//
// TEST_F(MaintenanceTest, MainteanceSaveQueryOnly) {
//     int64_t initial_size = 10000;
//     int64_t nlist = 100;
//
//     // Do a run with mainteance
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
//
//     // set maintenance policy
//     MaintenancePolicyParams params;
//     params.window_size = 100;
//     params.refinement_radius = 25;
//     params.min_partition_size = 10;
//     params.alpha = .25;
//     params.enable_split_rejection = true;
//     params.enable_delete_rejection = true;
//     params.delete_threshold_ns = 25.0;
//     params.split_threshold_ns = 1.0;
//     index->maintenance_policy_->set_params(params);
//
//     int start_nlist = index->nlist();
//     for (int64_t j = 0; j < 25; j++) {
//         vector<Tensor> ids_list;
//         vector<Tensor> dists_list;
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = index->search_one(query, k, .9);
//             auto ids = std::get<0>(result);
//             auto dists = torch::sqrt(std::get<1>(result));
//             ids_list.push_back(ids.unsqueeze(0));
//             dists_list.push_back(dists.unsqueeze(0));
//         }
//         auto ids = torch::cat(ids_list, 0);
//         auto dists = torch::cat(dists_list, 0);
//         Tensor recalls = calculate_recall(ids, ground_truth);
//         std::cout << "With Mainteance Recall: " << torch::stack(recalls).mean().item<float>() << std::endl;
//         index->maintenance_policy_->maintenance();
//     }
//     int finish_nlist = index->nlist();
//     std::cout << "After mainteance updated nlist from " << start_nlist << " to " << finish_nlist << std::endl;
//
//     // Save the index and reload it
//     std::string save_path = "post_mainteance_index.faiss";
//     index->save(save_path);
//     std::cout << "Saved index to " << save_path << std::endl;
//
//     auto reloaded_index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     reloaded_index->load(save_path);
//     std::cout << "Reloaded index from " << save_path << " with " << reloaded_index->nlist() << " partitions" << std::endl;
//
//     // Now run queries without mainteance and ensure it doesn't impact performance
//     for (int64_t j = 0; j < 5; j++) {
//         vector<Tensor> ids_list;
//         vector<Tensor> dists_list;
//         for (int64_t i = 0; i < query_vectors.size(0); i++) {
//             auto query = query_vectors[i].unsqueeze(0);
//             auto result = reloaded_index->search_one(query, k, .9);
//             auto ids = std::get<0>(result);
//             auto dists = torch::sqrt(std::get<1>(result));
//             ids_list.push_back(ids.unsqueeze(0));
//             dists_list.push_back(dists.unsqueeze(0));
//         }
//         auto ids = torch::cat(ids_list, 0);
//         auto dists = torch::cat(dists_list, 0);
//         Tensor recalls = calculate_recall(ids, ground_truth);
//         std::cout << "Post Mainteance Recall: " << torch::stack(recalls).mean().item<float>() << std::endl;
//     }
// }