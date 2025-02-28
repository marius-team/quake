// list_scanning_test.cpp

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <limits>
#include <algorithm>

#include "list_scanning.h" // Ensure this path is correct

// Test fixture for list scanning functions
class ListScanningTest : public ::testing::Test {
protected:
    // Helper function to create a TopkBuffer
    std::shared_ptr<TopkBuffer> create_buffer(int k, bool is_descending) {
        return std::make_shared<TypedTopKBuffer<float, int64_t>>(k, is_descending);
    }
};

// Test scan_list with single query and L2 metric
TEST_F(ListScanningTest, ScanList_SingleQuery_L2) {
    int k = 2;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Define a query vector and a list of vectors
    float query_vec[3] = {1.0f, 0.0f, 0.0f};
    float list_vecs[4 * 3] = {
        1.0f, 0.0f, 0.0f, // Distance: 0.0
        0.0f, 1.0f, 0.0f, // Distance: 2.0
        1.0f, 1.0f, 0.0f, // Distance: 1.0
        2.0f, 0.0f, 0.0f  // Distance: 1.0
    };
    int64_t list_ids[4] = {10, 20, 30, 40};

    // Expected top-k: 0.0 (id=10), 1.0 (id=30 or 40)
    scan_list(query_vec, list_vecs, list_ids, 4, 3, *buffer, faiss::METRIC_L2);

    std::vector<float> expected_dists = {0.0f, 1.0f};
    std::vector<int64_t> expected_ids = {10, 30}; // Could also be {10, 40} based on order

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    EXPECT_FLOAT_EQ(topk_dists[0], expected_dists[0]);
    EXPECT_EQ(topk_ids[0], expected_ids[0]);

    // Second element could be id=30 or id=40
    EXPECT_FLOAT_EQ(topk_dists[1], expected_dists[1]);
    EXPECT_TRUE(topk_ids[1] == 30 || topk_ids[1] == 40);
}

// Test scan_list with single query and Inner Product metric
TEST_F(ListScanningTest, ScanList_SingleQuery_InnerProduct) {
    int k = 2;
    bool is_descending = true; // Inner Product
    auto buffer = create_buffer(k, is_descending);

    // Define a query vector and a list of vectors
    float query_vec[3] = {1.0f, 0.0f, 0.0f};
    float list_vecs[4 * 3] = {
        1.0f, 0.0f, 0.0f, // Inner product: 1.0
        0.0f, 1.0f, 0.0f, // Inner product: 0.0
        1.0f, 1.0f, 0.0f, // Inner product: 1.0
        2.0f, 0.0f, 0.0f  // Inner product: 2.0
    };
    int64_t list_ids[4] = {10, 20, 30, 40};

    // Expected top-k: 2.0 (id=40), 1.0 (id=10 or 30)
    scan_list(query_vec, list_vecs, list_ids, 4, 3, *buffer, faiss::METRIC_INNER_PRODUCT);

    std::vector<float> expected_dists = {2.0f, 1.0f};
    std::vector<int64_t> expected_ids = {40, 10}; // Could also be {40, 30}

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    EXPECT_FLOAT_EQ(topk_dists[0], expected_dists[0]);
    EXPECT_EQ(topk_ids[0], expected_ids[0]);

    // Second element could be id=10 or id=30
    EXPECT_FLOAT_EQ(topk_dists[1], expected_dists[1]);
    EXPECT_TRUE(topk_ids[1] == 10 || topk_ids[1] == 30);
}

// Test batched_scan_list with multiple queries and L2 metric
TEST_F(ListScanningTest, BatchedScanList_MultipleQueries_L2) {
    int k = 2;
    MetricType metric = faiss::METRIC_L2;

    // Define two query vectors (dim=3)
    float query_vecs[2 * 3] = {
        1.0f, 0.0f, 0.0f, // Query 0
        0.0f, 1.0f, 0.0f  // Query 1
    };

    // Define a list of vectors (dim=3)
    float list_vecs[4 * 3] = {
        1.0f, 0.0f, 0.0f, // Q0: 0.0, Q1: 2.0
        0.0f, 1.0f, 0.0f, // Q0: 2.0, Q1: 0.0
        1.0f, 1.0f, 0.0f, // Q0: 1.0, Q1: 1.0
        2.0f, 0.0f, 0.0f  // Q0: 4.0, Q1: 2.0
    };
    int64_t list_ids[4] = {10, 20, 30, 40};

    // Create TopkBuffers for each query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(2, k, false); // L2 metric (is_descending=false)

    // Perform batched_scan_list
    batched_scan_list(
        query_vecs,
        list_vecs,
        list_ids,
        2, // num_queries
        4, // list_size
        3, // dim
        topk_buffers,
        metric
    );

    // Expected results:
    // Query 0: top-k distances: 0.0 (id=10), 1.0 (id=30)
    // Query 1: top-k distances: 0.0 (id=20), 1.0 (id=30)

    // Verify Query 0
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();
    ASSERT_EQ(topk_q0.size(), 2);
    EXPECT_FLOAT_EQ(topk_q0[0], 0.0f);
    EXPECT_EQ(topk_q0_ids[0], 10);
    EXPECT_FLOAT_EQ(topk_q0[1], 1.0f);
    EXPECT_EQ(topk_q0_ids[1], 30);

    // Verify Query 1
    auto topk_q1 = topk_buffers[1]->get_topk();
    auto topk_q1_ids = topk_buffers[1]->get_topk_indices();
    ASSERT_EQ(topk_q1.size(), 2);
    EXPECT_FLOAT_EQ(topk_q1[0], 0.0f);
    EXPECT_EQ(topk_q1_ids[0], 20);
    EXPECT_FLOAT_EQ(topk_q1[1], 1.0f);
    EXPECT_EQ(topk_q1_ids[1], 30);
}

// Test batched_scan_list with multiple queries and Inner Product metric
TEST_F(ListScanningTest, BatchedScanList_MultipleQueries_InnerProduct) {
    int k = 2;
    MetricType metric = faiss::METRIC_INNER_PRODUCT;

    // Define two query vectors (dim=3)
    float query_vecs[2 * 3] = {
        1.0f, 0.0f, 0.0f, // Query 0
        0.0f, 1.0f, 0.0f  // Query 1
    };

    // Define a list of vectors (dim=3)
    float list_vecs[4 * 3] = {
        1.0f, 0.0f, 0.0f, // Inner products [1.0, 0.0] (id=10)
        0.0f, 1.0f, 0.0f, // Inner products [0.0, 1.0] (id=20)
        .5f, 1.5f, 0.0f, // Inner products [.5, 1.5] (id=30)
        2.0f, 0.0f, 0.0f  // Inner products [2.0, 0.0] (id=40)
    };
    int64_t list_ids[4] = {10, 20, 30, 40};

    // Create TopkBuffers for each query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(2, k, true); // Inner Product metric (is_descending=true)

    // Perform batched_scan_list
    batched_scan_list(
        query_vecs,
        list_vecs,
        list_ids,
        2, // num_queries
        4, // list_size
        3, // dim
        topk_buffers,
        metric
    );

    // Expected results:
    // Query 0: top-k scores: 2.0 (id=40), 1.0 (id=10)
    // Query 1: top-k scores: 1.5 (id=30), 1.0 (id=20)

    // Verify Query 0
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();

    std::cout << "topk_q0: " << topk_q0[0] << " " << topk_q0[1] << std::endl;
    std::cout << "topk_q0_ids: " << topk_q0_ids[0] << " " << topk_q0_ids[1] << std::endl;

    ASSERT_EQ(topk_q0.size(), 2);
    EXPECT_FLOAT_EQ(topk_q0[0], 2.0f);
    EXPECT_TRUE(topk_q0_ids[0] == 40);
    EXPECT_FLOAT_EQ(topk_q0[1], 1.0f);
    EXPECT_TRUE(topk_q0_ids[1] == 10 || topk_q0_ids[1] == 20);

    // Verify Query 1
    auto topk_q1 = topk_buffers[1]->get_topk();
    auto topk_q1_ids = topk_buffers[1]->get_topk_indices();
    ASSERT_EQ(topk_q1.size(), 2);
    EXPECT_FLOAT_EQ(topk_q1[0], 1.5f);
    EXPECT_TRUE(topk_q1_ids[0] == 30);
    EXPECT_FLOAT_EQ(topk_q1[1], 1.0f);
    EXPECT_TRUE(topk_q1_ids[1] == 20);
}

// Test batched_scan_list without list_ids (list_ids == nullptr)
TEST_F(ListScanningTest, BatchedScanList_NoListIds) {
    int k = 2;
    MetricType metric = faiss::METRIC_L2;

    // Define two query vectors
    float query_vecs[2 * 2] = {
        1.0f, 0.0f, // Query 0
        0.0f, 1.0f  // Query 1
    };

    // Define a list of vectors
    float list_vecs[3 * 2] = {
        1.0f, 0.0f, // Distance for Q0: 0.0, Q1: 2.0
        0.0f, 1.0f, // Distance for Q0: 2.0, Q1: 0.0
        1.0f, 1.0f  // Distance for Q0: 1.0, Q1: 1.0
    };
    // list_ids == nullptr

    // Create TopkBuffers for each query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(2, k, false); // Inner Product metric (is_descending=true)


    // Perform batched_scan_list
    batched_scan_list(
        query_vecs,
        list_vecs,
        nullptr, // list_ids
        2, // num_queries
        3, // list_size
        2, // dim
        topk_buffers,
        metric
    );

    // Expected results:
    // Query 0: top-k distances: 0.0 (id=0), 1.0 (id=2)
    // Query 1: top-k distances: 0.0 (id=1), 1.0 (id=2)

    std::vector<float> expected_dists_q0 = {0.0f, 1.0f};
    std::vector<int64_t> expected_ids_q0 = {0, 2};
    std::vector<float> expected_dists_q1 = {0.0f, 1.0f};
    std::vector<int64_t> expected_ids_q1 = {1, 2};

    // Verify Query 0
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();

    std::cout << "topk_q0: " << topk_q0[0] << " " << topk_q0[1] << std::endl;
    std::cout << "topk_q0_ids: " << topk_q0_ids[0] << " " << topk_q0_ids[1] << std::endl;

    ASSERT_EQ(topk_q0.size(), 2);
    EXPECT_FLOAT_EQ(topk_q0[0], 0.0f);
    EXPECT_EQ(topk_q0_ids[0], 0);
    EXPECT_FLOAT_EQ(topk_q0[1], 1.0f);
    EXPECT_EQ(topk_q0_ids[1], 2);

    // Verify Query 1
    auto topk_q1 = topk_buffers[1]->get_topk();
    auto topk_q1_ids = topk_buffers[1]->get_topk_indices();
    ASSERT_EQ(topk_q1.size(), 2);
    EXPECT_FLOAT_EQ(topk_q1[0], 0.0f);
    EXPECT_EQ(topk_q1_ids[0], 1);
    EXPECT_FLOAT_EQ(topk_q1[1], 1.0f);
    EXPECT_EQ(topk_q1_ids[1], 2);
}

// Test batched_scan_list with batch_size parameter
TEST_F(ListScanningTest, BatchedScanList_WithBatchSize) {
    int k = 1;
    MetricType metric = faiss::METRIC_L2;

    // Define three query vectors (dim=2)
    float query_vecs[3 * 2] = {
        1.0f, 0.0f, // Query 0
        0.0f, 1.0f, // Query 1
        1.0f, 1.0f  // Query 2
    };

    // Define four list vectors (dim=2)
    float list_vecs[4 * 2] = {
        1.0f, 0.0f, // Distance for Q0: 0.0, Q1: 2.0
        0.0f, 1.0f, // Distance for Q0: 2.0, Q1: 0.0
        1.0f, 1.0f, // Distance for Q0: 1.0, Q1: 1.0
        2.0f, 2.0f  // Distance for Q0: 4.0, Q1: 2.0
    };
    int64_t list_ids[4] = {100, 200, 300, 400};

    // Create TopkBuffers for each query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(3, k, false); // L2 metric (is_descending=false)

    // Perform batched_scan_list with batch_size=2
    batched_scan_list(
        query_vecs,
        list_vecs,
        list_ids,
        3, // num_queries
        4, // list_size
        2, // dim
        topk_buffers,
        metric
    );

    // Expected top-k per query:
    // Query 0: min distance: 0.0 (id=100)
    // Query 1: min distance: 0.0 (id=200)
    // Query 2: min distance: 1.0 (id=300)

    // Verify Query 0
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();
    ASSERT_EQ(topk_q0.size(), 1);
    EXPECT_FLOAT_EQ(topk_q0[0], 0.0f);
    EXPECT_EQ(topk_q0_ids[0], 100);

    // Verify Query 1
    auto topk_q1 = topk_buffers[1]->get_topk();
    auto topk_q1_ids = topk_buffers[1]->get_topk_indices();
    ASSERT_EQ(topk_q1.size(), 1);
    EXPECT_FLOAT_EQ(topk_q1[0], 0.0f);
    EXPECT_EQ(topk_q1_ids[0], 200);

    // Verify Query 2
    auto topk_q2 = topk_buffers[2]->get_topk();
    auto topk_q2_ids = topk_buffers[2]->get_topk_indices();
    ASSERT_EQ(topk_q2.size(), 1);
    EXPECT_FLOAT_EQ(topk_q2[0], 0.0f);
    EXPECT_EQ(topk_q2_ids[0], 300);
}

// Test batched_scan_list with empty list
TEST_F(ListScanningTest, BatchedScanList_EmptyList) {
    int k = 2;
    MetricType metric = faiss::METRIC_L2;

    // Define one query vector (dim=2)
    float query_vecs[1 * 2] = {
        1.0f, 1.0f // Query 0
    };
    int num_queries = 1;

    // Empty list
    float list_vecs[0 * 2] = {}; // No list vectors
    int64_t list_ids[0] = {};

    // Create TopkBuffers for the single query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(num_queries, k, false); // L2 metric

    // Perform batched_scan_list
    batched_scan_list(
        query_vecs,
        list_vecs,
        list_ids,
        num_queries,
        0, // list_size
        2, // dim
        topk_buffers,
        metric
    );

    // Expected top-k: all distances remain +infinity, ids remain -1
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();
    ASSERT_EQ(topk_q0.size(), 0); // No valid top-k elements

    // Since there are no list vectors, all top-k entries should be default values
    // However, based on TopkBuffer implementation, if no elements were inserted, get_topk() returns an empty vector
    // To verify default values, you might need to adjust TopkBuffer's get_topk method to include default entries
    // Alternatively, ensure that the test logic handles empty top-k appropriately
}

// Test batched_scan_list with less than k elements
// Test batched_scan_list with less than k elements
TEST_F(ListScanningTest, BatchedScanList_LessThanKElements) {
    int k = 5;
    MetricType metric = faiss::METRIC_L2;

    // Define one query vector (dim=3)
    float query_vecs[1 * 3] = {1.0f, 1.0f, 1.0f}; // Query 0
    int num_queries = 1;

    // Define a list with only 2 vectors (dim=3)
    float list_vecs[2 * 3] = {
        1.0f, 1.0f, 1.0f, // Distance: 0.0
        2.0f, 2.0f, 2.0f  // Distance: sqrt(3)
    };
    int64_t list_ids[2] = {100, 200};

    // Create TopkBuffers for the single query
    std::vector<std::shared_ptr<TopkBuffer>> topk_buffers = create_buffers(num_queries, k, false); // L2 metric

    // Perform batched_scan_list
    batched_scan_list(
        query_vecs,
        list_vecs,
        list_ids,
        num_queries,
        2, // list_size=2
        3, // dim=3
        topk_buffers,
        metric
    );

    // Expected top-k: 0.0 (id=100), sqrt(3) (~1.73205) (id=200), inf, inf, inf
    // Since TopkBuffer maintains only inserted elements, we need to handle default values separately
    // Modify TopkBuffer's get_topk and get_topk_indices to include default entries if necessary

    // Retrieve top-k elements
    auto topk_q0 = topk_buffers[0]->get_topk();
    auto topk_q0_ids = topk_buffers[0]->get_topk_indices();

    // Verify inserted elements
    ASSERT_EQ(topk_q0.size(), 2); // Only 2 elements were inserted
    EXPECT_FLOAT_EQ(topk_q0[0], 0.0f);
    EXPECT_EQ(topk_q0_ids[0], 100);
    EXPECT_FLOAT_EQ(topk_q0[1], sqrt(3.0));
    EXPECT_EQ(topk_q0_ids[1], 200);
}

TEST_F(ListScanningTest, LargeListCorrectnessInnerProduct) {
    int num_queries = 100;
    int64_t list_size = 10000;
    int d = 128;
    int k = 10;
    Tensor query_vectors = torch::randn({num_queries, d}, torch::kFloat32);
    Tensor list_vectors = torch::randn({list_size, d}, torch::kFloat32);
    Tensor list_ids = torch::arange(0, list_size, torch::kInt64);

    // compute ground truth using pytorch matmul + topk
    Tensor distances = torch::matmul(query_vectors, list_vectors.t());
    auto topk = torch::topk(distances, k, 1, true);
    auto gt_ids = std::get<1>(topk);
    auto gt_dists = std::get<0>(topk);

    auto gt_ids_accessor = gt_ids.accessor<int64_t, 2>();
    auto gt_dists_accessor = gt_dists.accessor<float, 2>();

    // perform single query scan first
    auto buffer = make_shared<TopkBuffer>(k, true);
    for (int i = 0; i < num_queries; i++) {
        scan_list(
            query_vectors[i].data_ptr<float>(),
            list_vectors.data_ptr<float>(),
            list_ids.data_ptr<int64_t>(),
            list_size,
            d,
            *buffer,
            faiss::METRIC_INNER_PRODUCT
            );

        // check result is correct
        auto topk_dist = buffer->get_topk();
        auto topk_ids = buffer->get_topk_indices();

        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(topk_dist[j], gt_dists_accessor[i][j], .01);
            EXPECT_EQ(topk_ids[j], gt_ids_accessor[i][j]);
        }
        buffer->reset();
    }

    // now perform batched scan
    auto buffers = create_buffers(num_queries, k, true);
    batched_scan_list(
        query_vectors.data_ptr<float>(),
        list_vectors.data_ptr<float>(),
        list_ids.data_ptr<int64_t>(),
        num_queries,
        list_size,
        d,
        buffers,
        faiss::METRIC_INNER_PRODUCT
    );

    for (int i = 0; i < num_queries; i++) {
        auto topk_dist = buffers[i]->get_topk();
        auto topk_ids = buffers[i]->get_topk_indices();

        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(topk_dist[j], gt_dists_accessor[i][j], .01);
            EXPECT_EQ(topk_ids[j], gt_ids_accessor[i][j]);
        }
    }
}

TEST_F(ListScanningTest, LargeListCorrectnessL2) {
    int num_queries = 100;
    int64_t list_size = 10000;
    int d = 128;
    int k = 10;
    Tensor query_vectors = torch::randn({num_queries, d}, torch::kFloat32);
    Tensor list_vectors = torch::randn({list_size, d}, torch::kFloat32);
    Tensor list_ids = torch::arange(0, list_size, torch::kInt64);

    // compute ground truth using pytorch matmul + topk
    Tensor distances = torch::cdist(query_vectors, list_vectors);
    auto topk = torch::topk(distances, k, 1, false);
    auto gt_ids = std::get<1>(topk);
    auto gt_dists = std::get<0>(topk);

    auto gt_ids_accessor = gt_ids.accessor<int64_t, 2>();
    auto gt_dists_accessor = gt_dists.accessor<float, 2>();

    // perform single query scan first
    auto buffer = make_shared<TopkBuffer>(k, false);
    for (int i = 0; i < num_queries; i++) {
        scan_list(
            query_vectors[i].data_ptr<float>(),
            list_vectors.data_ptr<float>(),
            list_ids.data_ptr<int64_t>(),
            list_size,
            d,
            *buffer,
            faiss::METRIC_L2
            );

        // check result is correct
        auto topk_dist = buffer->get_topk();
        auto topk_ids = buffer->get_topk_indices();

        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(topk_dist[j], gt_dists_accessor[i][j], .01);
            EXPECT_EQ(topk_ids[j], gt_ids_accessor[i][j]);
        }
        buffer->reset();
    }

    // now perform batched scan
    auto buffers = create_buffers(num_queries, k, false);
    batched_scan_list(
        query_vectors.data_ptr<float>(),
        list_vectors.data_ptr<float>(),
        list_ids.data_ptr<int64_t>(),
        num_queries,
        list_size,
        d,
        buffers,
        faiss::METRIC_L2
    );

    for (int i = 0; i < num_queries; i++) {
        auto topk_dist = buffers[i]->get_topk();
        auto topk_ids = buffers[i]->get_topk_indices();

        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(topk_dist[j], gt_dists_accessor[i][j], .01);
            EXPECT_EQ(topk_ids[j], gt_ids_accessor[i][j]);
        }
    }
}