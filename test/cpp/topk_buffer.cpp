// topk_buffer_test.cpp

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <limits>

#include "list_scanning.h"

// Test fixture for TypedTopKBuffer
class TypedTopKBufferTest : public ::testing::Test {
protected:
    // Helper function to create a TopkBuffer
    std::shared_ptr<TopkBuffer> create_buffer(int k, bool is_descending) {
        return std::make_shared<TypedTopKBuffer<float, int64_t>>(k, is_descending);
    }
};

// Test adding individual elements and verifying top-k for L2
TEST_F(TypedTopKBufferTest, AddElements_L2) {
    int k = 3;
    bool is_descending = false; // L2: smaller distances are better
    auto buffer = create_buffer(k, is_descending);

    // Add distances
    buffer->add(5.0, 1);
    buffer->add(3.0, 2);
    buffer->add(4.0, 3);
    buffer->add(2.0, 4); // This should replace 5.0

    // Expected top-k: 2.0, 3.0, 4.0
    std::vector<float> expected_dists = {2.0, 3.0, 4.0};
    std::vector<int64_t> expected_ids = {4, 2, 3};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test adding individual elements and verifying top-k for Inner Product
TEST_F(TypedTopKBufferTest, AddElements_InnerProduct) {
    int k = 3;
    bool is_descending = true; // Inner Product: larger scores are better
    auto buffer = create_buffer(k, is_descending);

    // Add distances (similar to inner product scores)
    buffer->add(0.5, 1);
    buffer->add(0.3, 2);
    buffer->add(0.4, 3);
    buffer->add(0.6, 4); // This should replace 0.3

    // Expected top-k: 0.6, 0.5, 0.4
    std::vector<float> expected_dists = {0.6, 0.5, 0.4};
    std::vector<int64_t> expected_ids = {4, 1, 3};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test batch adding elements for L2
TEST_F(TypedTopKBufferTest, BatchAddElements_L2) {
    int k = 2;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Batch add: distances and ids
    std::vector<float> distances = {1.0, 4.0, 2.0};
    std::vector<int64_t> ids = {10, 20, 30};
    buffer->batch_add(distances.data(), ids.data(), distances.size());

    // Expected top-k: 1.0, 2.0
    std::vector<float> expected_dists = {1.0, 2.0};
    std::vector<int64_t> expected_ids = {10, 30};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test batch adding elements for Inner Product
TEST_F(TypedTopKBufferTest, BatchAddElements_InnerProduct) {
    int k = 2;
    bool is_descending = true; // Inner Product
    auto buffer = create_buffer(k, is_descending);

    // Batch add: inner product scores and ids
    std::vector<float> scores = {0.1, 0.4, 0.3};
    std::vector<int64_t> ids = {100, 200, 300};
    buffer->batch_add(scores.data(), ids.data(), scores.size());

    // Expected top-k: 0.4, 0.3
    std::vector<float> expected_dists = {0.4, 0.3};
    std::vector<int64_t> expected_ids = {200, 300};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test flushing the buffer
TEST_F(TypedTopKBufferTest, FlushBuffer) {
    int k = 2;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Add more elements than capacity to force flushing
    std::vector<float> distances = {5.0, 1.0, 3.0, 2.0, 4.0};
    std::vector<int64_t> ids = {50, 10, 30, 20, 40};
    for (size_t i = 0; i < distances.size(); i++) {
        buffer->add(distances[i], ids[i]);
    }

    // Expected top-k: 1.0, 2.0
    std::vector<float> expected_dists = {1.0f, 2.0f};
    std::vector<int64_t> expected_ids = {10, 20};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test buffer with less than k elements
TEST_F(TypedTopKBufferTest, LessThanKElements) {
    int k = 3;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Add only two elements
    buffer->add(2.0, 20);
    buffer->add(1.0, 10);

    // Expected top-k: 1.0, 2.0, +infinity
    std::vector<float> expected_dists = {1.0f, 2.0f, std::numeric_limits<float>::infinity()};
    std::vector<int64_t> expected_ids = {10, 20, -1};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();



    ASSERT_EQ(topk_dists.size(), 2);
    ASSERT_EQ(topk_ids.size(), 2);

    for (int i = 0; i < 2; i++) {
        std::cout << "i: " << i << " topk_dists[i]: " << topk_dists[i] << " expected_dists[i]: " << expected_dists[i] << std::endl;
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test TopkBuffer reset
TEST_F(TypedTopKBufferTest, ResetBuffer) {
    int k = 2;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Add some elements
    buffer->add(3.0, 30);
    buffer->add(1.0, 10);
    buffer->add(2.0, 20);

    // Reset the buffer
    buffer->reset();

    // After reset, topk should be initialized to +infinity for L2
    std::vector<float> expected_dists = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    std::vector<int64_t> expected_ids = {-1, -1};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), 0);
    ASSERT_EQ(topk_ids.size(), 0);
}

// Test that TopkBuffer maintains correct order for L2
TEST_F(TypedTopKBufferTest, Order_L2) {
    int k = 3;
    bool is_descending = false; // L2
    auto buffer = create_buffer(k, is_descending);

    // Add unsorted distances
    buffer->add(4.0, 40);
    buffer->add(1.0, 10);
    buffer->add(3.0, 30);
    buffer->add(2.0, 20);
    buffer->add(5.0, 50);

    // Expected top-k: 1.0, 2.0, 3.0
    std::vector<float> expected_dists = {1.0, 2.0, 3.0};
    std::vector<int64_t> expected_ids = {10, 20, 30};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}

// Test that TopkBuffer maintains correct order for Inner Product
TEST_F(TypedTopKBufferTest, Order_InnerProduct) {
    int k = 2;
    bool is_descending = true; // Inner Product
    auto buffer = create_buffer(k, is_descending);

    // Add unsorted inner product scores
    buffer->add(0.2, 200);
    buffer->add(0.5, 500);
    buffer->add(0.3, 300);
    buffer->add(0.6, 600);

    // Expected top-k: 0.6, 0.5
    std::vector<float> expected_dists = {0.6, 0.5};
    std::vector<int64_t> expected_ids = {600, 500};

    auto topk_dists = buffer->get_topk();
    auto topk_ids = buffer->get_topk_indices();

    ASSERT_EQ(topk_dists.size(), expected_dists.size());
    ASSERT_EQ(topk_ids.size(), expected_ids.size());

    for (int i = 0; i < k; i++) {
        EXPECT_FLOAT_EQ(topk_dists[i], expected_dists[i]);
        EXPECT_EQ(topk_ids[i], expected_ids[i]);
    }
}