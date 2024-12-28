// query_coordinator_test.cpp

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <stdexcept>

#include "query_coordinator.h"
#include "partition_manager.h"
#include "quake_index.h"


// Test fixture
class QueryCoordinatorTest : public ::testing::Test {
protected:
    int64_t dimension_ = 8;
    int64_t total_vectors_ = 40;
    int64_t nlist_ = 4;
    int64_t k_ = 3;
    int64_t num_queries_ = 5;
    shared_ptr<QuakeIndex> index_;
    Tensor queries_;

    void SetUp() override {
        // Create a dummy PartitionManager

        Tensor vectors = torch::randn({total_vectors_, dimension_}, torch::kCPU);
        Tensor ids = torch::arange(0, total_vectors_, torch::kInt64);

        index_ = std::make_shared<QuakeIndex>();
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = nlist_;
        build_params->metric = faiss::METRIC_L2;
        index_->build(vectors, ids, build_params);

        // We'll create 5 random queries
        queries_ = torch::randn({num_queries_, dimension_}, torch::kCPU);
    }
};

TEST_F(QueryCoordinatorTest, NullParentBatchedScanTest) {
    // Parent is null => QueryCoordinator scans all partitions (flat index scenario)
    auto coordinator = std::make_shared<QueryCoordinator>(
        nullptr /* parent */,
        index_->partition_manager_,
        faiss::METRIC_L2);

    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 3;
    // Because parent is null, the code sets batched_scan = true for us.

    auto result = coordinator->search(queries_, search_params);

    // We do not do an exact recall test here, but let's do some basic sanity checks:
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(result->ids.sizes(), (std::vector<int64_t>{queries_.size(0), search_params->k}));
    ASSERT_EQ(result->distances.sizes(), (std::vector<int64_t>{queries_.size(0), search_params->k}));

    // Ensure no -1s remain if data_size >= k
    // (In your actual test, you might compare to brute force results.)
    for (int64_t i = 0; i < num_queries_; i++) {
        for (int64_t j = 0; j < k_; j++) {
            ASSERT_NE(result->ids[i][j].item<int64_t>(), -1);
        }
    }
}

TEST_F(QueryCoordinatorTest, NullParentSerialScanTest) {
    // Force "serial_scan" by disabling batched_scan
    auto coordinator = std::make_shared<QueryCoordinator>(
        nullptr,
        index_->partition_manager_,
        faiss::METRIC_L2);

    // We'll set batched_scan = false artificially to see if the serial path works
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 3;
    search_params->batched_scan = false; // Force serial scan

    auto result = coordinator->search(queries_, search_params);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(result->ids.size(0), queries_.size(0));
    ASSERT_EQ(result->ids.size(1), search_params->k);
}

TEST_F(QueryCoordinatorTest, NonNullParentTest) {
    auto coordinator = std::make_shared<QueryCoordinator>(
        index_->parent_,
        index_->partition_manager_,
        faiss::METRIC_L2);

    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 2; // final K
    search_params->nprobe = 1; // parent's search does a top-1

    auto result = coordinator->search(queries_, search_params);
    ASSERT_TRUE(result != nullptr);

    // The parent's result says "scan partition #0 only", so the coordinator
    // should have results from partition #0 only, for all queries.
    // Just do a basic check:
    ASSERT_EQ(result->ids.size(0), queries_.size(0));
    ASSERT_EQ(result->ids.size(1), 2); // k=2
}

TEST_F(QueryCoordinatorTest, EmptyQueryTest) {
    // If queries is empty, we expect an empty result
    auto coordinator = std::make_shared<QueryCoordinator>(
        nullptr,
        index_->partition_manager_,
        faiss::METRIC_L2);

    auto empty_queries = torch::empty({0, dimension_}, torch::kCPU);

    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 2;

    auto result = coordinator->search(empty_queries, search_params);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(result->ids.numel(), 0);
    ASSERT_EQ(result->distances.numel(), 0);
}

TEST_F(QueryCoordinatorTest, NullPartitionManagerThrows) {
    // If PartitionManager is null, we expect the coordinator to throw
    auto coordinator = std::make_shared<QueryCoordinator>(
        nullptr,
        nullptr /* partition_manager_ = null */,
        faiss::METRIC_L2);

    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 2;

    // The code in serial_scan or batched_serial_scan throws if partition_manager_ is null
    EXPECT_THROW({
                 coordinator->search(queries_, search_params);
                 }, std::runtime_error);
}

TEST_F(QueryCoordinatorTest, WorkerInitializationTest) {
    auto coordinator = std::make_shared<QueryCoordinator>(
        index_->parent_,
        index_->partition_manager_,
        faiss::METRIC_L2
    );

    // check that the workers are not initialized
    ASSERT_FALSE(coordinator->workers_initialized_);

    coordinator = std::make_shared<QueryCoordinator>(
        index_->parent_,
        index_->partition_manager_,
        faiss::METRIC_L2,
        4 /* num_workers */
    );

    ASSERT_TRUE(coordinator->workers_initialized_);
}
