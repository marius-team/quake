//
// quake_index_test.cpp
//
// Unit tests for the QuakeIndex class using Google Test (GTest).
//

#include <gtest/gtest.h>
#include "quake_index.h"
#include <torch/torch.h>
#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/chunked_array.h>
#include <random>
#include <arrow/compute/api_vector.h>

// Helper functions for random data
static torch::Tensor generate_random_data(int64_t num_vectors, int64_t dim) {
    return torch::randn({num_vectors, dim}, torch::kFloat32);
}

static torch::Tensor generate_sequential_ids(int64_t count, int64_t start = 0) {
    return torch::arange(start, start + count, torch::kInt64);
}

static std::shared_ptr<arrow::Table> generate_data_frame(int64_t num_vectors, torch::Tensor ids) {
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    // Builders for the "price" and "id" columns
    arrow::DoubleBuilder price_builder(pool);
    arrow::Int64Builder id_builder(pool);

    // Append values to the builders
    for (int64_t i = 0; i < num_vectors; i++) {
        price_builder.Append(static_cast<double>(i) * 1.5); // Price column
        id_builder.Append(ids[i].item<int64_t>());          // ID column from the input tensor
    }

    // Finalize the arrays
    std::shared_ptr<arrow::Array> price_array;
    std::shared_ptr<arrow::Array> id_array;
    price_builder.Finish(&price_array);
    id_builder.Finish(&id_array);

    // Define the schema with two fields: "price" and "id"
    std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
        arrow::field("id", arrow::int64()),
        arrow::field("price", arrow::float64()),
    };
    auto schema = std::make_shared<arrow::Schema>(schema_vector);

    // Create and return the table with both columns
    return arrow::Table::Make(schema, {id_array, price_array});
}

class QuakeIndexTest : public ::testing::Test {
protected:
    // Example parameters
    int64_t dimension_ = 16;
    int64_t nlist_ = 8;
    int64_t num_vectors_ = 100;
    int64_t num_queries_ = 5;

    // Data & IDs
    torch::Tensor data_vectors_;
    torch::Tensor data_ids_;

    // Query vectors
    torch::Tensor query_vectors_;

    // Arrow data
    std::shared_ptr<arrow::Table> attributes_table;

    void SetUp() override {
        // Generate random data
        data_vectors_ = generate_random_data(num_vectors_, dimension_);
        // Generate sequential IDs
        data_ids_ = generate_sequential_ids(num_vectors_, 0);

        // Queries
        query_vectors_ = generate_random_data(num_queries_, dimension_);

        // Arrow data
        attributes_table = generate_data_frame(num_vectors_, data_ids_);
    }
};

// Basic constructor test
TEST_F(QuakeIndexTest, ConstructorTest) {
    QuakeIndex index;
    // We can confirm default fields are null
    EXPECT_EQ(index.parent_, nullptr);
    EXPECT_EQ(index.partition_manager_, nullptr);
    EXPECT_EQ(index.query_coordinator_, nullptr);
    EXPECT_EQ(index.build_params_, nullptr);
    EXPECT_EQ(index.maintenance_policy_params_, nullptr);
}

// Building the index with nlist > 1 => multi-partition scenario
TEST_F(QuakeIndexTest, BuildTest) {
    QuakeIndex index;

    // create build_params
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;       // Use multi-partition
    build_params->metric = "l2";
    build_params->niter = 5;           // small kmeans iteration

    auto timing_info = index.build(data_vectors_, data_ids_, build_params, attributes_table);

    // Check that we created partition_manager_, parent_, etc.
    EXPECT_NE(index.partition_manager_, nullptr);
    EXPECT_NE(index.query_coordinator_, nullptr);
    EXPECT_NE(index.build_params_, nullptr);

    // For a multi-partition scenario, we expect a parent_ that indexes centroids
    EXPECT_NE(index.parent_, nullptr);

    // Check that the timing_info fields look valid
    EXPECT_EQ(timing_info->n_vectors, data_vectors_.size(0));
    EXPECT_EQ(timing_info->d, data_vectors_.size(1));
}

// Building the index with nlist <= 1 => a "flat" scenario
TEST_F(QuakeIndexTest, BuildFlatTest) {
    QuakeIndex index;

    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1;  // "Flat"
    build_params->metric = "ip";
    build_params->niter = 3;

    auto timing_info = index.build(data_vectors_, data_ids_, build_params);

    // parent_ should be nullptr because we only have a single partition
    EXPECT_EQ(index.parent_, nullptr);

    // partition_manager_ & query_coordinator_ should exist
    EXPECT_NE(index.partition_manager_, nullptr);
    EXPECT_NE(index.query_coordinator_, nullptr);
}

// Test searching the index
TEST_F(QuakeIndexTest, SearchPartitionedTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;
    build_params->metric = "l2";
    index.build(data_vectors_, data_ids_, build_params, attributes_table);

    // Create a search_params object (if you need special fields, set them up)
    auto search_params = std::make_shared<SearchParams>();
    // Example: if your QuakeIndex respects nprobe, set it in search_params
    search_params->nprobe = 4;
    search_params->k = 5;

    // Perform the search
    auto search_result = index.search(query_vectors_, search_params);
    Tensor ret_ids = search_result->ids;
    Tensor ret_dis = search_result->distances;
    shared_ptr<SearchTimingInfo> timing_info = search_result->timing_info;

    // Basic shape checks
    ASSERT_EQ(ret_ids.size(0), query_vectors_.size(0));
    ASSERT_EQ(ret_ids.size(1), search_params->k);
    ASSERT_EQ(ret_dis.size(0), query_vectors_.size(0));
    ASSERT_EQ(ret_dis.size(1), search_params->k);
}

TEST_F(QuakeIndexTest, SearchFlatTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->metric = "l2";
    index.build(data_vectors_, data_ids_, build_params, attributes_table);

    // Create a search_params object (if you need special fields, set them up)
    auto search_params = std::make_shared<SearchParams>();
    // Example: if your QuakeIndex respects nprobe, set it in search_params
    search_params->nprobe = 4;
    search_params->k = 5;

    // Perform the search
    auto search_result = index.search(query_vectors_, search_params);
    Tensor ret_ids = search_result->ids;
    Tensor ret_dis = search_result->distances;
    shared_ptr<SearchTimingInfo> timing_info = search_result->timing_info;

    // Basic shape checks
    ASSERT_EQ(ret_ids.size(0), query_vectors_.size(0));
    ASSERT_EQ(ret_ids.size(1), search_params->k);
    ASSERT_EQ(ret_dis.size(0), query_vectors_.size(0));
    ASSERT_EQ(ret_dis.size(1), search_params->k);
}

// Test the get(...) method
TEST_F(QuakeIndexTest, GetVectorsTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 1; // Simple
    index.build(data_vectors_, data_ids_, build_params);

    // Suppose we retrieve the first 10 IDs
    int64_t get_size = 10;
    auto some_ids = data_ids_.slice(/*dim=*/0, /*start=*/0, /*end=*/get_size);
    torch::Tensor retrieved_vectors = index.get(some_ids);

    // Because the default implementation is incomplete, we might only check shape
    // or that it doesn't throw. If you had a real "get" that returns the correct data,
    // you could compare them directly.
    EXPECT_EQ(retrieved_vectors.size(0), get_size);
}

// Test add method
TEST_F(QuakeIndexTest, AddTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;
    index.build(data_vectors_, data_ids_, build_params);

    Tensor add_vectors = generate_random_data(10, dimension_);
    Tensor add_ids = generate_sequential_ids(10, 1000);
    auto attr_table = generate_data_frame(10,add_ids);

    auto modify_info = index.add(add_vectors, add_ids, attr_table);
    EXPECT_EQ(modify_info->n_vectors, 10);
    EXPECT_GE(modify_info->modify_time_us, 0);
}

// Test remove method
TEST_F(QuakeIndexTest, RemoveTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;
    index.build(data_vectors_, data_ids_, build_params, attributes_table);

    // remove half of them
    int64_t remove_count = num_vectors_ / 2;
    auto remove_ids = data_ids_.slice(0, 0, remove_count);
    auto modify_info = index.remove(remove_ids);

    // Basic checks
    EXPECT_EQ(modify_info->n_vectors, remove_count);
    EXPECT_GT(modify_info->modify_time_us, 0);
}

// Test ntotal() and nlist()
TEST_F(QuakeIndexTest, NTotalNListTest) {
    QuakeIndex index;

    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;
    index.build(data_vectors_, data_ids_, build_params);

    // We don't know exactly how QuakeIndex stores the total,
    // but we at least expect ntotal() to be ~ num_vectors
    int64_t total = index.ntotal();
    EXPECT_EQ(total, num_vectors_);

    // nlist should be ~ nlist_ in a multi-part scenario
    int64_t actual_nlist = index.nlist();
    EXPECT_EQ(actual_nlist, nlist_);
}

// Test saving and loading
TEST_F(QuakeIndexTest, SaveLoadTest) {
    QuakeIndex index;

    // Build
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = nlist_;
    index.build(data_vectors_, data_ids_, build_params);

    // Save the index
    std::string path = "quake_test_index.qidx";
    index.save(path);

    // Load into a new index
    QuakeIndex loaded_index;
    loaded_index.load(path);

    // minimal checks
    EXPECT_EQ(loaded_index.ntotal(), index.ntotal());
    EXPECT_EQ(loaded_index.nlist(), index.nlist());
}

// -------------------------------------------------------------------------
// LARGE BUILD STRESS TEST
// -------------------------------------------------------------------------
TEST(QuakeIndexStressTest, LargeBuildTest) {
    // Attempt to build an index with a large number of vectors.
    // Adjust these numbers based on your available memory/compute.
    int64_t dimension = 128;     // Medium-high dimension
    int64_t num_vectors = 1e6;   // 1 million vectors
    auto data_vectors = generate_random_data(num_vectors, dimension);
    auto data_ids = generate_sequential_ids(num_vectors, 0);
    auto data_frames = generate_data_frame(num_vectors, data_ids);

    QuakeIndex index;

    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 512;
    build_params->metric = "l2";
    // Keep the iteration count modest to avoid overly long tests
    build_params->niter = 5;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto timing_info = index.build(data_vectors, data_ids, build_params, data_frames);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Check that the build completed and that we didn't crash
    ASSERT_NE(timing_info, nullptr);
    // Basic correctness
    EXPECT_EQ(index.ntotal(), num_vectors);

    auto build_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "[LargeBuildTest] Building " << num_vectors
              << " vectors took " << build_duration_ms << " ms.\n";
}

// -------------------------------------------------------------------------
// REPEATED BUILD-SEARCH STRESS TEST
// -------------------------------------------------------------------------
TEST(QuakeIndexStressTest, RepeatedBuildSearchTest) {
    // Build, search, teardown repeatedly, looking for memory leaks or instability
    int64_t iteration_count = 5;
    int64_t dimension = 32;
    int64_t nlist = 16;
    int64_t num_vectors = 10000;
    int64_t num_queries = 100;

    // Pre-generate data
    auto data_vectors = generate_random_data(num_vectors, dimension);
    auto data_ids = generate_sequential_ids(num_vectors, 1000);
    auto data_frames = generate_data_frame(num_vectors, data_ids);
    auto query_vectors = generate_random_data(num_queries, dimension);

    for (int i = 0; i < iteration_count; i++) {
        QuakeIndex index;
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = nlist;
        build_params->metric = "l2";
        build_params->niter = 3;

        // Build index
        index.build(data_vectors, data_ids, build_params, data_frames);

        // Query index
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = 10;
        search_params->nprobe = 4;

        // Search
        auto result = index.search(query_vectors, search_params);
        ASSERT_EQ(result->ids.size(0), query_vectors.size(0));
        ASSERT_EQ(result->ids.size(1), search_params->k);

        // Optionally print iteration stats
        // e.g., memory usage or timing
        // ...
    }

    // If we reach here without crashes or leaks, the test is considered passing.
    SUCCEED();
}

// // -------------------------------------------------------------------------
// // DIMENSION MISMATCH / INVALID INPUT STRESS TEST
// // -------------------------------------------------------------------------
// TEST(QuakeIndexStressTest, InvalidInputTest) {
//     QuakeIndex index;
//
//     // Build basic index
//     auto build_params = std::make_shared<IndexBuildParams>();
//     build_params->nlist = 2;
//     build_params->metric = faiss::METRIC_L2;
//
//     // Valid data
//     auto valid_vectors = generate_random_data(100, 8);
//     auto valid_ids = generate_sequential_ids(100);
//
//     index.build(valid_vectors, valid_ids, build_params);
//
//     // Try to add vectors with different dimensionality
//     auto invalid_vectors = generate_random_data(10, 16);
//     auto invalid_ids = generate_sequential_ids(10);
//
//     // We expect either an exception or an error code. Adjust depending on how your code handles it.
//     // If your QuakeIndex gracefully handles dimension mismatches, replace EXPECT_ANY_THROW with the relevant check.
//     EXPECT_ANY_THROW(index.add(invalid_vectors, invalid_ids));
//
//     // Similarly for a search dimension mismatch
//     auto invalid_query = generate_random_data(5, 16);
//     auto search_params = std::make_shared<SearchParams>();
//     search_params->k = 3;
//
//     EXPECT_ANY_THROW(index.search(invalid_query, search_params));
// }

// -------------------------------------------------------------------------
// RAPID ADD-REMOVE-ADD STRESS TEST
// -------------------------------------------------------------------------
TEST(QuakeIndexStressTest, RapidAddRemoveAddTest) {
    // Repeatedly add, remove, and re-add data to see if the index remains consistent.

    int64_t dimension = 16;
    int64_t batch_size = 1000;
    int64_t repeats = 10;

    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    // Alternatively, you can prime it with a small base set if your design demands it.
    build_params->nlist = 2;

    Tensor initial_vectors = generate_random_data(batch_size, dimension);
    Tensor initial_ids = generate_sequential_ids(batch_size);

    index.build(initial_vectors, initial_ids, build_params);

    for (int i = 1; i < repeats; ++i) {
        // Add
        auto add_vectors = generate_random_data(batch_size, dimension);
        auto add_ids = generate_sequential_ids(batch_size, i * batch_size);
        auto add_info = index.add(add_vectors, add_ids);
        ASSERT_EQ(add_info->n_vectors, batch_size);

        // Remove half
        auto remove_ids = add_ids.slice(0, 0, batch_size / 2);
        auto remove_info = index.remove(remove_ids);
        ASSERT_EQ(remove_info->n_vectors, batch_size / 2);

        // Add again
        auto add_info_2 = index.add(add_vectors.slice(0, 0, batch_size / 2), add_ids.slice(0, 0, batch_size / 2));
        ASSERT_EQ(add_info_2->n_vectors, batch_size / 2);

        // Optional: do a sanity check search
        auto query_vectors = generate_random_data(5, dimension);
        auto search_params = std::make_shared<SearchParams>();
        search_params->k = 2;
        auto res = index.search(query_vectors, search_params);
        ASSERT_EQ(res->ids.size(0), query_vectors.size(0));
        ASSERT_EQ(res->ids.size(1), 2);
    }

    SUCCEED();
}

// -------------------------------------------------------------------------
// LARGE DIMENSION STRESS TEST
// -------------------------------------------------------------------------
TEST(QuakeIndexStressTest, HighDimensionTest) {
    // Build an index with very large dimension to see if it handles memory pressure or
    // indexing logic correctly.

    int64_t dimension = 1024;  // Large dimension
    int64_t num_vectors = 5000;
    auto data_vectors = generate_random_data(num_vectors, dimension);
    auto data_ids = generate_sequential_ids(num_vectors);
    auto data_frames = generate_data_frame(num_vectors, data_ids);

    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    // For extremely large dimension, consider a small nlist to avoid enormous memory usage
    build_params->nlist = 2;
    build_params->metric = "l2";
    build_params->niter = 3;

    // If your system doesn’t have enough memory for bigger tests, reduce num_vectors or dimension.
    auto timing_info = index.build(data_vectors, data_ids, build_params, data_frames);
    ASSERT_NE(timing_info, nullptr);
    EXPECT_EQ(index.ntotal(), num_vectors);

    // Basic search test to ensure the index is functional
    auto query_vectors = generate_random_data(10, dimension);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 5;
    auto result = index.search(query_vectors, search_params);

    ASSERT_EQ(result->ids.size(0), query_vectors.size(0));
    ASSERT_EQ(result->ids.size(1), search_params->k);
}

// -------------------------------------------------------------------------
// SEARCH, ADD, REMOVE, AND MAINTENANCE TEST
// -------------------------------------------------------------------------
TEST(QuakeIndexStressTest, SearchAddRemoveMaintenanceTest) {
    // Repeatedly search, add, remove, and perform maintenance to see if the index remains consistent.

    int64_t dimension = 16;
    int64_t num_vectors = 100000;
    int64_t num_queries = 100;
    int64_t batch_size = 10;

    QuakeIndex index;
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 100;
    build_params->metric = "l2";
    build_params->niter = 3;

    Tensor data_vectors = generate_random_data(num_vectors, dimension);
    Tensor data_ids = generate_sequential_ids(num_vectors, 0);

    index.build(data_vectors, data_ids, build_params);

    for (int i = 0; i < 100; i++) {
        // Search
        std::cout << "Iteration " << i << std::endl;
        auto query_vectors = generate_random_data(num_queries, dimension) * .1;
        auto search_params = std::make_shared<SearchParams>();
        search_params->nprobe = 1;
        search_params->k = 5;
        auto search_result = index.search(query_vectors, search_params);
        ASSERT_EQ(search_result->ids.size(0), query_vectors.size(0));
        ASSERT_EQ(search_result->ids.size(1), search_params->k);

        // Add
        auto add_vectors = generate_random_data(batch_size, dimension);
        auto add_ids = generate_sequential_ids(batch_size, (i * batch_size) + num_vectors);
        auto add_info = index.add(add_vectors, add_ids);
        ASSERT_EQ(add_info->n_vectors, batch_size);

        // Remove
        auto remove_ids = add_ids.slice(0, 0, batch_size / 2);
        auto remove_info = index.remove(remove_ids);
        ASSERT_EQ(remove_info->n_vectors, batch_size / 2);

        index.maintenance();
    }

    SUCCEED();
}
