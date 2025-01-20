#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <memory>
#include <cmath>
#include "quake_index.h"

/**
 * Performs a naive, brute-force search for each query vector over all data vectors.
 * Returns a pair of (ids, distances) shaped [num_queries, k].
 *
 * @param data         Tensor of shape (N, D)
 * @param data_ids     Tensor of shape (N) with IDs
 * @param queries      Tensor of shape (Q, D)
 * @param k            Number of neighbors to retrieve
 *
 * NOTE: This implementation uses L2 distance. If your QuakeIndex is using
 * METRIC_INNER_PRODUCT or something else, adapt as needed, or store both versions.
 */
std::pair<torch::Tensor, torch::Tensor> brute_force_topk_l2(
    const torch::Tensor& data,
    const torch::Tensor& data_ids,
    const torch::Tensor& queries,
    int64_t k)
{
    // Basic checks
    TORCH_CHECK(data.dim() == 2, "data must be 2D");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D");
    TORCH_CHECK(data.size(1) == queries.size(1), "dim mismatch for data & queries");
    TORCH_CHECK(data_ids.dim() == 1, "data_ids must be 1D");
    TORCH_CHECK(data.size(0) == data_ids.size(0), "data and data_ids mismatch");

    auto device = data.device();
    int64_t N = data.size(0);
    int64_t Q = queries.size(0);
    int64_t D = data.size(1);

    auto data_cpu = data.to(torch::kCPU);
    auto ids_cpu = data_ids.to(torch::kCPU);
    auto queries_cpu = queries.to(torch::kCPU);

    // Prepare output
    auto out_ids = torch::empty({Q, k}, torch::kLong);
    auto out_distances = torch::empty({Q, k}, torch::kFloat);

    // For each query
    for (int64_t q = 0; q < Q; ++q) {
        auto query_vec = queries_cpu[q];
        // We'll store (distance, id) pairs
        std::vector<std::pair<float, int64_t>> dist_id_vec(N);

        // Compute distances
        for (int64_t i = 0; i < N; ++i) {
            auto data_vec = data_cpu[i];
            // L2 distance = || data_vec - query_vec ||^2
            float dist = (data_vec - query_vec).pow(2).sum().item<float>();
            dist_id_vec[i] = {dist, ids_cpu[i].item<int64_t>()};
        }

        // Partial sort to get top K
        std::nth_element(
            dist_id_vec.begin(),
            dist_id_vec.begin() + k,
            dist_id_vec.end(),
            [](const auto& a, const auto& b) {
                return a.first < b.first; // ascending order by distance
            }
        );
        dist_id_vec.resize(k);

        // Sort those top k to store them in ascending distance
        std::sort(dist_id_vec.begin(), dist_id_vec.end(),
                  [](const auto& a, const auto& b) {
                      return a.first < b.first;
                  });

        // Write back to output
        for (int64_t kk = 0; kk < k; ++kk) {
            out_distances[q][kk] = dist_id_vec[kk].first;
            out_ids[q][kk] = dist_id_vec[kk].second;
        }
    }

    return {out_ids, out_distances};
}

/**
 * Given the ground truth top-K IDs and the approximate top-K IDs for each query,
 * compute the recall, i.e., the fraction of true neighbors found in the approximate results.
 *
 * @param true_ids      [Q, K]
 * @param approx_ids    [Q, K]
 * @return recall for the entire batch, a float in [0,1].
 */
float compute_recall_at_k(
    const torch::Tensor& true_ids,
    const torch::Tensor& approx_ids)
{
    TORCH_CHECK(true_ids.dim() == 2, "true_ids must be 2D");
    TORCH_CHECK(approx_ids.dim() == 2, "approx_ids must be 2D");
    TORCH_CHECK(true_ids.size(0) == approx_ids.size(0), "query count mismatch");
    TORCH_CHECK(true_ids.size(1) == approx_ids.size(1), "k mismatch");

    int64_t Q = true_ids.size(0);
    int64_t K = true_ids.size(1);

    auto true_ids_cpu = true_ids.to(torch::kCPU);
    auto approx_ids_cpu = approx_ids.to(torch::kCPU);

    int64_t total_found = 0;
    int64_t total_possible = Q * K;

    for (int64_t q = 0; q < Q; ++q) {
        // We'll gather the true neighbors in a set to facilitate lookup
        std::unordered_set<int64_t> ground_truth_set;
        for (int64_t kk = 0; kk < K; ++kk) {
            ground_truth_set.insert(true_ids_cpu[q][kk].item<int64_t>());
        }
        // Now check how many of the approx results are in that set
        for (int64_t kk = 0; kk < K; ++kk) {
            int64_t candidate_id = approx_ids_cpu[q][kk].item<int64_t>();
            if (ground_truth_set.find(candidate_id) != ground_truth_set.end()) {
                total_found++;
            }
        }
    }

    float recall = float(total_found) / float(total_possible);
    return recall;
}

// ---------------------------------------------------------------
// Test fixture for recall tests
// ---------------------------------------------------------------
class QuakeIndexRecallTest : public ::testing::Test {
protected:
    // Data sizes
    int64_t dimension_ = 32;
    int64_t num_vectors_ = 100000;
    int64_t num_queries_ = 10;
    int64_t k_ = 10;

    // Pre-generated data
    Tensor data_vectors_;
    Tensor data_ids_;
    Tensor query_vectors_;

    void SetUp() override {
        // Generate random data
        data_vectors_ = torch::randn({num_vectors_, dimension_});
        data_ids_ = torch::arange(0, num_vectors_, torch::kInt64);

        // Queries
        query_vectors_ = torch::randn({num_queries_, dimension_});
    }
};

TEST_F(QuakeIndexRecallTest, FlatIndexRecallTest) {
    // Build flat index (nlist=1 => no coarse quantization)
    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1;  // Flat
    build_params->metric = "l2";
    index.build(data_vectors_, data_ids_, build_params);

    // Perform approximate (actually exact for a flat index) search
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = k_;
    auto approx_result = index.search(query_vectors_, search_params);

    std::cout << approx_result->ids << std::endl;
    std::cout << approx_result->distances << std::endl;

    // Ground truth
    auto ground_truth = brute_force_topk_l2(data_vectors_, data_ids_, query_vectors_, k_);

    // Compute recall
    float recall = compute_recall_at_k(
        /*true_ids=*/ground_truth.first,
        /*approx_ids=*/approx_result->ids
    );

    std::cout << "FlatIndexRecallTest => Recall: " << recall << std::endl;

    // We expect near-perfect (or perfect) recall for a flat L2 index.
    EXPECT_GE(recall, 0.99f);
}

TEST_F(QuakeIndexRecallTest, MultiPartitionRecallTest) {
    // Build partitioned index
    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 8;      // Multi-partition
    build_params->metric = "l2";
    build_params->niter = 5;      // small number of k-means iterations
    index.build(data_vectors_, data_ids_, build_params);

    std::vector<int64_t> nprobe_values = {1, 2, 4, 8};

    // Ground truth once (brute force)
    auto ground_truth = brute_force_topk_l2(data_vectors_, data_ids_, query_vectors_, k_);

    for (auto nprobe : nprobe_values) {
        // Search
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = k_;
        search_params->nprobe = nprobe;
        auto approx_result = index.search(query_vectors_, search_params);

        // Compute recall
        float recall = compute_recall_at_k(
            ground_truth.first,
            approx_result->ids
        );

        std::cout << "MultiPartitionRecallTest (nprobe=" << nprobe
                  << ") => Recall: " << recall << std::endl;
    }

    SUCCEED();
}

TEST_F(QuakeIndexRecallTest, InnerProductRecallTest) {
    auto data_vectors_ip = data_vectors_;
    auto data_ids_ip = data_ids_;
    auto query_vectors_ip = query_vectors_;

    // Build index with IP
    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1; // "flat"
    build_params->metric = "ip";
    index.build(data_vectors_ip, data_ids_ip, build_params);

    // Search
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = k_;
    auto approx_result = index.search(query_vectors_ip, search_params);

    // Ground truth for IP
    Tensor gt_dist = query_vectors_ip.matmul(data_vectors_ip.t());
    auto topk = gt_dist.topk(k_, /*dim=*/1, /*largest=*/true);
    auto gt_ids = std::get<1>(topk);

    gt_ids = data_ids_ip.index_select(0, gt_ids.view(-1)).view_as(gt_ids);

    // Evaluate recall
    float recall = compute_recall_at_k(gt_ids, approx_result->ids);

    std::cout << "InnerProductRecallTest => Recall: " << recall << std::endl;
    EXPECT_GE(recall, 0.99f);
}

TEST_F(QuakeIndexRecallTest, RecallVsNlistTest) {
    std::vector<int64_t> nlist_values = {1, 4, 16};
    auto ground_truth = brute_force_topk_l2(data_vectors_, data_ids_, query_vectors_, k_);

    for (auto nlist_val : nlist_values) {
        QuakeIndex index;
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = nlist_val;
        build_params->metric = "l2";
        build_params->niter = 5;
        index.build(data_vectors_, data_ids_, build_params);

        // Search with a moderate nprobe
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = k_;
        search_params->nprobe = std::max<int64_t>(1, nlist_val / 2);

        auto approx_result = index.search(query_vectors_, search_params);
        float recall = compute_recall_at_k(ground_truth.first, approx_result->ids);

        std::cout << "RecallVsNlistTest (nlist=" << nlist_val
                  << ", nprobe=" << search_params->nprobe
                  << ") => Recall: " << recall << std::endl;
    }

    SUCCEED();
}

TEST_F(QuakeIndexRecallTest, RecallVsRecallTargetL2) {
    std::vector<float> recall_target_values = {.5, .6, .7, .8, .9, .95, .99, 1.0};
    auto ground_truth = brute_force_topk_l2(data_vectors_, data_ids_, query_vectors_, k_);

    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1000;
    build_params->metric = "l2";
    build_params->niter = 5;
    index.build(data_vectors_, data_ids_, build_params);

    for (auto recall_target : recall_target_values) {
        // Search with a moderate nprobe
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = k_;
        search_params->recompute_threshold = 0.0;
        search_params->recall_target = recall_target;
        search_params->initial_search_fraction = .5;

        auto approx_result = index.search(query_vectors_, search_params);
        float recall = compute_recall_at_k(ground_truth.first, approx_result->ids);

        std::cout << "RecallVsRecallTarget (recall_target=" << recall_target
                  << ") => Recall: " << recall << std::endl;
    }
}

TEST_F(QuakeIndexRecallTest, RecallVsRecallTargetL2UsingWorkers) {
    std::vector<float> recall_target_values = {.5, .6, .7, .8, .9, .95, .99, 1.0};
    // std::vector<float> recall_target_values = {.5};
    auto ground_truth = brute_force_topk_l2(data_vectors_, data_ids_, query_vectors_, k_);

    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1000;
    build_params->metric = "l2";
    build_params->niter = 5;
    build_params->num_workers = 4;
    index.build(data_vectors_, data_ids_, build_params);

    for (auto recall_target : recall_target_values) {
        // Search with a moderate nprobe
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = k_;
        search_params->nprobe = 1;
        search_params->recompute_threshold = 0.0;
        search_params->recall_target = recall_target;
        search_params->initial_search_fraction = .1;
        search_params->aps_flush_period_us = 1;

        auto approx_result = index.search(query_vectors_, search_params);
        float recall = compute_recall_at_k(ground_truth.first, approx_result->ids);

        std::cout << "RecallVsRecallTarget (recall_target=" << recall_target
                  << ") => Recall: " << recall << std::endl;
    }
}