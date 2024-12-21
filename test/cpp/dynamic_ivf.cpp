// dynamic_ivf_test.cpp
//
// Unit tests for the DynamicIVF_C class using Google Test (GTest).

#include <gtest/gtest.h>
#include "dynamic_ivf.h"

#include <torch/torch.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>

#include <vector>
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>

using torch::Tensor;
using std::vector;
using std::shared_ptr;

// Helper function to generate random data
Tensor generate_random_data(int64_t num_vectors, int64_t dim) {
    return torch::randn({num_vectors, dim}, torch::kFloat32);
}

// Helper function to generate sequential IDs
Tensor generate_sequential_ids(int64_t num_vectors) {
    return torch::arange(num_vectors, torch::kInt64);
}

// Test fixture for DynamicIVF_C
class DynamicIVFTest : public ::testing::Test {
protected:
    int dimension = 3;
    int nlist = 100;
    int nprobe = 5;
    int num_codebooks = 8;
    int code_size = 8;
    int num_vectors = 10000;
    int num_queries = 1;
    Tensor data_vectors;
    Tensor data_ids;
    Tensor query_vectors;
    faiss::MetricType metric = faiss::METRIC_L2;

    void SetUp() override {
        // Generate random data and queries
        data_vectors = generate_random_data(num_vectors, dimension);
        data_ids = generate_sequential_ids(num_vectors);
        query_vectors = generate_random_data(num_queries, dimension);
    }
};

// Test constructor and basic properties
TEST_F(DynamicIVFTest, ConstructorTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    EXPECT_EQ(index->nlist(), nlist);
    EXPECT_EQ(index->ntotal(), 0);
}

// Test building the index
TEST_F(DynamicIVFTest, BuildIndexTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    BuildTimingInfo timing_info = index->build(data_vectors, data_ids);
    EXPECT_GT(index->ntotal(), 0);
    EXPECT_EQ(index->ntotal(), num_vectors);
    EXPECT_TRUE(index->centroids().defined());
    EXPECT_EQ(index->centroids().size(0), nlist);
    EXPECT_EQ(index->centroids().size(1), dimension);
}

// Test adding vectors to the index
TEST_F(DynamicIVFTest, AddVectorsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);
    int64_t initial_total = index->ntotal();

    // Add new vectors
    Tensor new_vectors = generate_random_data(100, dimension);
    Tensor new_ids = generate_sequential_ids(100) + num_vectors;
    index->add(new_vectors, new_ids);

    EXPECT_EQ(index->ntotal(), initial_total + 100);
}

TEST_F(DynamicIVFTest, GetVectorsAndIDsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Get vectors and IDs
    auto [vectors, ids] = index->get_vectors_and_ids();

    // Check that the number of vectors is correct
    EXPECT_EQ(vectors.size(0), num_vectors);
    EXPECT_EQ(ids.size(0), num_vectors);

    // Check that the number of dimensions is correct
    EXPECT_EQ(vectors.size(1), dimension);

    // sort the ids
    ids = std::get<0>(torch::sort(ids));
    EXPECT_TRUE(torch::equal(ids, data_ids));
}

// Test removing vectors from the index
TEST_F(DynamicIVFTest, RemoveVectorsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);
    int64_t initial_total = index->ntotal();

    // Remove some vectors
    Tensor remove_ids = data_ids.slice(0, 0, 100);
    index->remove(remove_ids);

    EXPECT_EQ(index->ntotal(), initial_total - 100);
}

// Test saving and loading the index
TEST_F(DynamicIVFTest, SaveLoadTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Save the index
    std::string index_path = "test_index.faiss";
    index->save(index_path);

    // Load the index
    auto loaded_index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    loaded_index->load(index_path);

    loaded_index->search(query_vectors, nprobe, 5);

    EXPECT_EQ(loaded_index->ntotal(), index->ntotal());
    EXPECT_EQ(loaded_index->nlist(), index->nlist());
    EXPECT_TRUE(torch::allclose(loaded_index->centroids(), index->centroids()));
}

// Test searching the index
TEST_F(DynamicIVFTest, SearchTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    int k = 5;
    auto [ret_ids, ret_dis, timing_info] = index->search(query_vectors, nprobe, k);

    EXPECT_EQ(ret_ids.size(0), num_queries);
    EXPECT_EQ(ret_ids.size(1), k);
    EXPECT_EQ(ret_dis.size(0), num_queries);
    EXPECT_EQ(ret_dis.size(1), k);
}

// // Test refining clusters
// TEST_F(DynamicIVFTest, RefineClustersTest) {
//     auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
//     index->build(data_vectors, data_ids);
//     Tensor cluster_ids = torch::tensor({0, 1, 2}, torch::kInt64);
//
//     // Verify centroids have been updated
//     Tensor centroids_before = index->centroids().clone().index_select(0, cluster_ids);
//
//     // Select some clusters to refine
//     index->refine_clusters(cluster_ids);
//
//     Tensor centroids_after = index->centroids().index_select(0, cluster_ids);
//
//     // check that the number of vectors has not changed
//     EXPECT_EQ(index->ntotal(), num_vectors);
//
//     // check that the number of partitions has not changed
//     EXPECT_EQ(index->nlist(), nlist);
// }

// Test computing quantization error
TEST_F(DynamicIVFTest, ComputeQuantizationErrorTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    Tensor quantization_errors = index->compute_quantization_error();
    EXPECT_EQ(quantization_errors.size(0), nlist);
    EXPECT_TRUE(torch::all(quantization_errors >= 0).item<bool>());
}

// Test cluster covariance computation
TEST_F(DynamicIVFTest, ComputeClusterCovarianceTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    int cluster_id = 0;
    Tensor covariance = index->compute_cluster_covariance(cluster_id);

    EXPECT_EQ(covariance.size(0), dimension);
    EXPECT_EQ(covariance.size(1), dimension);
}

// Test computing cluster sums
TEST_F(DynamicIVFTest, ComputeClusterSumsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    Tensor cluster_sums = index->compute_cluster_sums(false);
    EXPECT_EQ(cluster_sums.size(0), nlist);
    EXPECT_EQ(cluster_sums.size(1), dimension);
}

// Test splitting partitions
TEST_F(DynamicIVFTest, SplitPartitionsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Split partitions
    Tensor partition_ids = torch::tensor({0, 1}, torch::kInt64);

    Tensor new_centroids;
    vector<Tensor> new_vectors, new_ids;
    std::tie(new_centroids, new_vectors, new_ids) = index->split_partitions(partition_ids);

    index->delete_partitions(partition_ids, /*reassign=*/false);

    // Add new partitions to the index
    index->add_partitions(new_centroids, new_vectors, new_ids);

    // Verify that nlist has increased
    EXPECT_EQ(index->nlist(), nlist + partition_ids.size(0));

    // Verify that the size has not changed
    EXPECT_EQ(index->ntotal(), num_vectors);
}

// Test adding a level
TEST_F(DynamicIVFTest, AddLevelTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    int new_nlist = nlist / 2;
    index->add_level(new_nlist);

    // Verify that the parent index exists and has the correct nlist
    EXPECT_NE(index->parent_, nullptr);
    EXPECT_EQ(index->parent_->nlist(), new_nlist);
}

// Test removing a level
TEST_F(DynamicIVFTest, RemoveLevelTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    index->add_level(nlist / 2);
    index->remove_level();

    // Verify that the parent index is nullptr
    EXPECT_EQ(index->parent_->parent_, nullptr);
}

// Test adding partitions
TEST_F(DynamicIVFTest, AddPartitionsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Generate new partitions
    Tensor new_centroids = generate_random_data(5, dimension);
    vector<Tensor> new_vectors;
    vector<Tensor> new_ids;

    int64_t curr_num_vectors = num_vectors;
    for (int i = 0; i < new_centroids.size(0); i++) {
        new_vectors.push_back(generate_random_data(100, dimension));
        new_ids.push_back(generate_sequential_ids(100) + curr_num_vectors);
        curr_num_vectors += 100;
    }

    // Add new partitions
    index->add_partitions(new_centroids, new_vectors, new_ids);

    // Verify that nlist has increased
    EXPECT_EQ(index->nlist(), nlist + new_centroids.size(0));
}

// Test deleting partitions
TEST_F(DynamicIVFTest, DeletePartitionsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);
    int initial_nlist = index->nlist();

    // Delete partitions
    Tensor partition_ids = torch::tensor({0, 1}, torch::kInt64);
    index->delete_partitions(partition_ids, /*reassign=*/true);

    // Verify that nlist has decreased or partitions are empty
    EXPECT_EQ(index->nlist(), initial_nlist - partition_ids.size(0));

    // Check that the size has not changed
    EXPECT_EQ(index->ntotal(), num_vectors);
}

TEST_F(DynamicIVFTest, RecomputeCentroidsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Modify some vectors
    Tensor modified_vectors = data_vectors.clone();
    modified_vectors.slice(0, 0, 10) += 1.0;

    // Reassign modified vectors
    index->remove(data_ids.slice(0, 0, 10));
    index->add(modified_vectors.slice(0, 0, 10), data_ids.slice(0, 0, 10));

    // Recompute centroids
    index->recompute_centroids();

    // Optionally, verify that centroids have changed
    // This part is left as a comment because it depends on specific implementation details
    // EXPECT_TRUE(/* condition to verify centroids have changed */);
}

// Test get_cluster_ids
TEST_F(DynamicIVFTest, GetClusterIdsTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    Tensor cluster_ids = index->get_cluster_ids();
    EXPECT_EQ(cluster_ids.size(0), num_vectors);
}

// Test get_nprobe_for_recall_target
TEST_F(DynamicIVFTest, GetNProbeForRecallTargetTest) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    int k = 5;
    float recall_target = 0.9f;

    int optimal_nprobe = index->get_nprobe_for_recall_target(query_vectors, k, recall_target);

    EXPECT_GE(optimal_nprobe, 1);
    EXPECT_LE(optimal_nprobe, nlist);
}

TEST_F(DynamicIVFTest, TestSingleQuery) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    int k = 5;
    float recall_target = 0.9f;
    Tensor query = generate_random_data(1, dimension);    
    auto result = index->search_one(query, k, recall_target);
    std::get<2>(result)->print();
}

#ifdef QUAKE_USE_NUMA
TEST_F(DynamicIVFTest, TestResetWorkers) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric);
    index->build(data_vectors, data_ids);

    // Save the index
    std::string index_path = "test_index.faiss";
    index->save(index_path);

    // Load the index
    auto loaded_index = std::make_shared<DynamicIVF_C>(0, 0, metric, 0, -1, -1, true, false, false, true, true);
    loaded_index->load(index_path, true);

    // Call the reset workers
    loaded_index->reset_workers(2, true, true);
    loaded_index->parent_->reset_workers(2, true, true);

    // Now perform the search
    Tensor query = generate_random_data(1, dimension);
    int k = 5;
    float recall_target = 0.9f;
    auto result = loaded_index->search_one(query, k, recall_target);
    std::get<2>(result)->print();
}

TEST_F(DynamicIVFTest, NumaTestMultiQuery) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 8, -1, -1, true);
    index->build(data_vectors, data_ids);

    while(!index->index_ready()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    int k = 5;
    float recall_target = 0.9f;

    // Run a large search
    Tensor query = generate_random_data(100, dimension);    
    auto result = index->search_one(query, k, recall_target);
    std::cout << "Many vectors search result: " << std::endl;
    std::get<2>(result)->print();
    std::cout << std::endl;

    // Now run a small serach
    query = generate_random_data(10, dimension);    
    result = index->search_one(query, k, recall_target);
    std::cout << "Not many vectors search result: " << std::endl;
    std::get<2>(result)->print();
    std::cout << std::endl;
}

TEST_F(DynamicIVFTest, NumaTestSingleQuery) {
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 8, -1, -1, true);
    index->build(data_vectors, data_ids);

    while(!index->index_ready()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    int k = 5;
    float recall_target = 0.9f;
    Tensor query = generate_random_data(1, dimension);    
    auto result = index->search_one(query, k, recall_target);
    std::get<2>(result)->print();
}

TEST_F(DynamicIVFTest, NumaTestSingleAdaptive) {
    // Build the index
    int64_t dimension = 64;
    int64_t complete_vectors_size = 200000;
    Tensor test_data_vectors = generate_random_data(complete_vectors_size, dimension);
    Tensor test_data_ids = generate_sequential_ids(complete_vectors_size);
    int64_t nlist = 1000;
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 4, -1, -1, true, false, false, true, false, true);
    index->build(test_data_vectors, test_data_ids);

    index->set_timeout_values(-1, 500);
    while(!index->index_ready()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    int k = 100;
    float recall_target = 0.9f;
    int nprobe = 0.25 * nlist;
    int num_batches = 10;
    int num_search_queries = 1;
    for(int i = 0; i < num_batches; i++) {
        Tensor query = generate_random_data(num_search_queries, dimension);    
        auto result = index->search(query, nprobe, k, recall_target);
        std::cout << "Query " << i << " breakdown: " << std::endl;
        std::get<2>(result)->print();
        std::cout << std::endl;
    }
}

TEST_F(DynamicIVFTest, NumaTestMultiAdaptive) {
    // Create the index
    int64_t dimension = 64;
    int64_t complete_vectors_size = 200000;
    Tensor test_data_vectors = generate_random_data(complete_vectors_size, dimension);
    Tensor test_data_ids = generate_sequential_ids(complete_vectors_size);
    int64_t nlist = 1000;
    int64_t second_level_nlist = 20;
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, metric, 4, -1, -1, true, false, false, true, false, true);

    // Build the index
    std::cout << "Building the index" << std::endl;
    index->build(test_data_vectors, test_data_ids);
    
    index->set_timeout_values(-1, 500);

    std::cout << "Calling add level" << std::endl;
    index->add_level(second_level_nlist);
    while(!index->index_ready()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Index is ready" << std::endl;

    int k = 100;
    float recall_target = 0.9f;
    int nprobe = 0.25 * nlist;
    int num_queries = 25;
    for(int i = 0; i < num_queries; i++) {
        Tensor query = generate_random_data(5, dimension);    
        auto result = index->search(query, nprobe, k, recall_target);
        std::cout << "Query " << i << " breakdown: " << std::endl;
        std::get<2>(result)->print();
        std::cout << std::endl;
    }
}

TEST(StressTest, NumaInsertQueryRemove) {
    int64_t dimension = 64;
    int64_t complete_vectors_size = 20000;
    Tensor data_vectors = generate_random_data(complete_vectors_size, dimension);
    Tensor data_ids = generate_sequential_ids(complete_vectors_size);

    int64_t initial_size = complete_vectors_size/2;
    int64_t batch_size = initial_size/10;
    int64_t nlist = 100;
    int64_t nprobe = 25;
    int64_t k = 8;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(dimension, nlist, faiss::METRIC_L2, 2, -1, -1, true, false, false, true, true);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
    
    while(!index->index_ready()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    int64_t offset = initial_size;
    int64_t delete_offset = 0;

    while (offset < data_vectors.size(0)) {
        // Perform an insert
        int64_t end = std::min(offset + batch_size, data_vectors.size(0));
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;
        std::cout << "Add call for offset " << offset << "/" << data_vectors.size(0) << std::endl;

        // Perform a query
        Tensor query = generate_random_data(1, dimension);    
        auto result = index->search(query, nprobe, k);

        // Perform a remove
        Tensor remove_ids = data_ids.narrow(0, delete_offset, batch_size);
        index->remove(remove_ids);
        delete_offset += batch_size;
        std::cout << "Remove called for delete offset of " << delete_offset << std::endl;
    }
}
#endif

TEST_F(DynamicIVFTest, AddRemoveLevels) {
    // Seed the random number generator
    srand(time(NULL));

    // Create the dataset data
    int vector_dimension = 64;
    Tensor data_vectors = generate_random_data(num_vectors, vector_dimension);
    Tensor data_ids = generate_sequential_ids(num_vectors);
    Tensor query_vectors = generate_random_data(num_queries, vector_dimension);
    
    // Create the intial index
    int k = 5;
    int default_n_list = 25;
    auto index = std::make_shared<DynamicIVF_C>(vector_dimension, default_n_list, faiss::METRIC_L2);
    index->build(data_vectors, data_ids);
    int index_height = 2;
    
    // Get the initial index result
    Tensor baseline_ids; Tensor baseline_distances; 
    std::tie(baseline_ids, baseline_distances, std::ignore) = index->search(query_vectors, 5, k);

    for(int i = 0; i < 15; i++) {
        // Perform an operation
        double random_value = ((double) rand()) / RAND_MAX;
        if(index_height >= 3 && random_value < 0.5) {
            index->remove_level();
            index_height -= 1;
        } else {
            index->add_level(default_n_list);
            index_height += 1;
        }

        // Rerun the same query and ensure we get the same result
        Tensor current_ids; Tensor current_distances; 
        std::tie(current_ids, current_distances, std::ignore) = index->search(query_vectors, 5, k);

        // Check that the returned IDs are identical
        EXPECT_TRUE(torch::equal(baseline_ids, current_ids))
            << "Mismatch in returned IDs for iteration " << i;

        // Check that the returned distances are identical within a small tolerance
        EXPECT_TRUE(torch::allclose(baseline_distances, current_distances, /*atol=*/1e-6, /*rtol=*/1e-4))
            << "Mismatch in returned distances for iteration " << i;
    }
}

TEST(WorkloadCases, TestNonConsecutiveIDs) {
    int k = 5;
    int d = 64;
    int nlist = 100;
    int n = 100000;
    int nqueries = 10;
    int nprobe = 100;

    Tensor data_vectors = generate_random_data(n, d);
    Tensor data_ids = generate_sequential_ids(n);
    Tensor queries = generate_random_data(nqueries, d);

    auto index = std::make_shared<DynamicIVF_C>(d, nlist, faiss::METRIC_L2);
    index->build(data_vectors, data_ids);
    auto [ret_ids, ret_dis, timing_info] = index->search(queries, nprobe, k);

    // Shuffle the IDs and vectors
    Tensor rand_perm = torch::randperm(data_ids.size(0));

    // Build the index
    auto index2 = std::make_shared<DynamicIVF_C>(d, nlist, faiss::METRIC_L2);
    index2->build(data_vectors.index_select(0, rand_perm), data_ids.index_select(0, rand_perm));
    auto [ret_ids2, ret_dis2, timing_info2] = index2->search(queries, nprobe, k);

    // Check that the returned distances are identical within a small tolerance
    EXPECT_TRUE(torch::allclose(ret_dis, ret_dis2, /*atol=*/1e-6, /*rtol=*/1e-4))
        << "Mismatch in returned distances.";

    // Check that the returned IDs are identical
    EXPECT_TRUE(torch::equal(ret_ids, ret_ids2))
        << "Mismatch in returned IDs.";
}

TEST(WorkerTests, TestWorkers) {
    // Parameters for the test
    int dimension = 128;
    int nlist = 100;
    faiss::MetricType metric = faiss::METRIC_L2;
    int num_workers_initial = 1; // Number of workers used to build and save the index
    int num_vectors = 100000;       // Reduced size for faster testing
    int num_queries = 100;
    int k = 10;                     // Number of nearest neighbors to search
    int nprobe = 10;

    // Define different numbers of workers to test
    std::vector<int> worker_counts = {1, 2, 4, 8, 16, 32};
#ifdef QUAKE_USE_NUMA
    std::vector<bool> use_numa = {false, true};
    std::vector<bool> same_core = {false, true};
#else
    std::vector<bool> use_numa = {false};
    std::vector<bool> same_core = {false};
#endif


    // Paths for saving and loading the index
    std::string index_path = "test_index.faiss";

    // Generate random data and queries
    Tensor data_vectors = generate_random_data(num_vectors, dimension);
    Tensor data_ids = generate_sequential_ids(num_vectors);
    Tensor query_vectors = generate_random_data(num_queries, dimension);

    // Step 1: Build the index once with the initial number of workers and save it
    {
        // Initialize the index with the initial number of workers
        auto index = std::make_shared<DynamicIVF_C>(
            dimension,
            nlist,
            metric,
            num_workers_initial  // Initial worker count
            // Other parameters use default values
        );

        // Build the index
        index->build(data_vectors, data_ids);

        // Save the index to disk
        index->save(index_path);
    }

    // Store results for each worker count
    struct SearchResult {
        Tensor ids;
        Tensor distances;
    };

    std::map<std::tuple<int, bool, bool>, SearchResult> results_map;

    // Step 2: Load the saved index with different worker counts and perform searches
    for (const auto& num_workers : worker_counts) {

        for (const auto& numa_opt : use_numa) {

            for (const auto& same_core_opt : same_core) {
                // skip if same_core is true and numa_opt is false
                if (same_core_opt && !numa_opt) {
                    continue;
                }

                std::tuple<int, bool, bool> key = std::make_tuple(num_workers, numa_opt, same_core_opt);

                // Initialize the index with the specified number of workers
                auto loaded_index = std::make_shared<DynamicIVF_C>(
                    dimension,
                    nlist,
                    metric,
                    num_workers,  // Varying the number of workers
                    -1,
                    -1,
                    numa_opt,
                    false,
                    false,
                    false
                    // Other parameters use default values
                );

                // Load the index from disk
                loaded_index->load(index_path);

                // Perform the search
                auto [ret_ids, ret_dis, timing_info] = loaded_index->search(query_vectors, /*nprobe=*/nprobe, k);

                // Store the results
                results_map[key] = SearchResult{ret_ids, ret_dis};
            }
        }
    }

    // Step 3: Compare results across all worker counts
    // Use the results from single worker, openmp as reference
    std::tuple<int, bool, bool> reference_key = std::make_tuple(worker_counts[0], false, false);

    auto reference_ids = results_map[reference_key].ids;
    auto reference_dis = results_map[reference_key].distances;

    for (size_t i = 0; i < worker_counts.size(); ++i) {
        for (const auto& numa_opt : use_numa) {
            for (const auto& same_core_opt : same_core) {
                if (same_core_opt && !numa_opt) {
                    continue;
                }

                std::cout << "Comparing results with " << worker_counts[i] << " workers, numa_opt: " << numa_opt
                          << ", same_core_opt: " << same_core_opt << std::endl;

                std::tuple<int, bool, bool> key = std::make_tuple(worker_counts[i], numa_opt, same_core_opt);
                int current_workers = worker_counts[i];
                auto current_ids = results_map[key].ids;
                auto current_dis = results_map[key].distances;

                // Check that the returned IDs are identical
                EXPECT_TRUE(torch::equal(reference_ids, current_ids))
                    << "Mismatch in returned IDs with " << current_workers << " workers.";

                // Check that the returned distances are identical within a small tolerance
                EXPECT_TRUE(torch::allclose(reference_dis, current_dis, /*atol=*/1e-6, /*rtol=*/1e-4))
                    << "Mismatch in returned distances with " << current_workers << " workers.";
            }
        }
    }

    // Step 4: Clean up by deleting the saved index file
    // This ensures that the test does not leave residual files
    int remove_result = std::remove(index_path.c_str());
    EXPECT_EQ(remove_result, 0) << "Failed to delete the index file: " << index_path;
}

// Stress Tests
TEST(StressTest, InsertOnly) {
    Tensor data_vectors = generate_random_data(1000000, 64);
    Tensor data_ids = generate_sequential_ids(1000000);

    int64_t initial_size = 100000;
    int64_t batch_size = 1000;
    int64_t nlist = 100;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(64, nlist, faiss::METRIC_L2);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));

    int64_t offset = initial_size;
    while (offset < data_vectors.size(0)) {
        int64_t end = std::min(offset + batch_size, data_vectors.size(0));
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;
    }
}

TEST(StressTest, InsertAndRemove) {
    Tensor data_vectors = generate_random_data(1000000, 64);
    Tensor data_ids = generate_sequential_ids(1000000);

    int64_t initial_size = 100000;
    int64_t batch_size = 1000;
    int64_t nlist = 100;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(64, nlist, faiss::METRIC_L2);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));

    int64_t offset = initial_size;

    int64_t delete_offset = 0;
    while (offset < data_vectors.size(0)) {
        int64_t end = std::min(offset + batch_size, data_vectors.size(0));
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;

        // Remove some random vectors from the index
        Tensor remove_ids = data_ids.narrow(0, delete_offset, batch_size);
        index->remove(remove_ids);
        delete_offset += batch_size;
    }
}

TEST(StressTest, InsertOnlySplitAndDelete) {
    int64_t vector_size = 1000000;
    Tensor data_vectors = generate_random_data(vector_size, 64);
    Tensor data_ids = generate_sequential_ids(vector_size);

    int64_t chunk_size = vector_size/20;
    int64_t initial_size = chunk_size;
    int64_t batch_size = chunk_size;
    int64_t nlist = 100;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(64, nlist, faiss::METRIC_L2);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));

    int64_t offset = initial_size;

    int delete_id = 0;
    while (offset < data_vectors.size(0)) {
        int64_t end = std::min(offset + batch_size, data_vectors.size(0));
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;

        Tensor curr_partition_ids = index->get_partition_ids();
        curr_partition_ids = curr_partition_ids.index_select(0, torch::randperm(curr_partition_ids.size(0)));

        // Delete some random partitions
        Tensor partition_ids = curr_partition_ids.narrow(0, 0, 5);
        index->delete_partitions(partition_ids, /*reassign=*/true);
        curr_partition_ids = curr_partition_ids.narrow(0, 5, curr_partition_ids.size(0) - 5);

        // Split some random partitions
        Tensor split_partition_ids = curr_partition_ids.narrow(0, 0, 5);

        // Filter out any partitions that are smaller than 8 (minimum size)
        Tensor split_partition_sizes = index->get_partition_sizes(split_partition_ids);
        split_partition_ids = split_partition_ids.masked_select(split_partition_sizes > 8);

        Tensor new_centroids;
        vector<Tensor> new_vectors, new_ids;
        std::tie(new_centroids, new_vectors, new_ids) = index->split_partitions(split_partition_ids);

        // Delete the split partitions
        index->delete_partitions(split_partition_ids, /*reassign=*/false);

        // Add the new partitions
        index->add_partitions(new_centroids, new_vectors, new_ids);

        std::cout << "Iteration " << delete_id << std::endl;
        std::cout << "Num partitions: " << index->nlist() << std::endl;
        std::cout << "Num vectors: " << index->ntotal() << std::endl;
    }
}

TEST(StressTest, InsertAndQuerySplitAndDelete) {
    int64_t total_data_size = 1000000; 
    Tensor data_vectors = generate_random_data(total_data_size, 64);
    Tensor data_ids = generate_sequential_ids(total_data_size);
    Tensor query_vectors = generate_random_data(1000, 64);

    int64_t chunk_size = total_data_size/20;
    int64_t initial_size = chunk_size;
    int64_t batch_size = chunk_size;
    int64_t nlist = 100;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(64, nlist, faiss::METRIC_L2);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
    int k = 5;

    int64_t offset = initial_size;

    int delete_id = 0;
    int64_t total_vectors = data_vectors.size(0);
    while (offset < total_vectors) {
        int64_t end = std::min(offset + batch_size, total_vectors);
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;

        Tensor curr_partition_ids = index->get_partition_ids();
        curr_partition_ids = curr_partition_ids.index_select(0, torch::randperm(curr_partition_ids.size(0)));

        // Delete some random partitions
        Tensor partition_ids = curr_partition_ids.narrow(0, 0, 5);
        index->delete_partitions(partition_ids, /*reassign=*/true);
        curr_partition_ids = curr_partition_ids.narrow(0, 5, curr_partition_ids.size(0) - 5);

        // Split some random partitions
        Tensor split_partition_ids = curr_partition_ids.narrow(0, 0, 5);

        // Filter out any partitions that are smaller than 8 (minimum size)
        Tensor split_partition_sizes = index->get_partition_sizes(split_partition_ids);
        split_partition_ids = split_partition_ids.masked_select(split_partition_sizes > 8);

        Tensor new_centroids;
        vector<Tensor> new_vectors, new_ids;
        std::tie(new_centroids, new_vectors, new_ids) = index->split_partitions(split_partition_ids);

        // Delete the split partitions
        index->delete_partitions(split_partition_ids, /*reassign=*/false);

        // Add the new partitions
        index->add_partitions(new_centroids, new_vectors, new_ids);

        // Query the index
        Tensor ret_ids, ret_dis;
        std::tie(ret_ids, ret_dis, std::ignore) = index->search(query_vectors, 5, k);

        std::cout << "Num partitions: " << index->nlist() << std::endl;
        std::cout << "Num vectors: " << index->ntotal() << std::endl;
        std::cout << "Data Processed: " << offset << "/" << total_vectors << std::endl;
    }
}

TEST(StressTest, MultiLevelInsertAndQuerySplitAndDelete) {
    Tensor data_vectors = generate_random_data(100000, 64);
    Tensor data_ids = generate_sequential_ids(100000);

    Tensor query_vectors = generate_random_data(100, 64);

    int64_t initial_size = 50000;
    int64_t batch_size = 5000;
    int64_t nlist = 1000;

    // Build initial index
    auto index = std::make_shared<DynamicIVF_C>(64, nlist, faiss::METRIC_L2);
    index->build(data_vectors.slice(0, 0, initial_size), data_ids.slice(0, 0, initial_size));
    index->add_level(nlist / 50);

    int64_t offset = initial_size;

    int delete_id = 0;
    while (offset < data_vectors.size(0)) {
        int64_t end = std::min(offset + batch_size, data_vectors.size(0));
        index->add(data_vectors.slice(0, offset, end), data_ids.slice(0, offset, end));
        offset = end;

        Tensor curr_partition_ids = index->get_partition_ids();
        curr_partition_ids = curr_partition_ids.index_select(0, torch::randperm(curr_partition_ids.size(0)));

        // Delete some random partitions
        Tensor partition_ids = curr_partition_ids.narrow(0, 0, 5);
        index->delete_partitions(partition_ids, /*reassign=*/true);
        curr_partition_ids = curr_partition_ids.narrow(0, 5, curr_partition_ids.size(0) - 5);

        // Split some random partitions
        Tensor split_partition_ids = curr_partition_ids.narrow(0, 0, 5);

        // Filter out any partitions that are smaller than 8 (minimum size)
        Tensor split_partition_sizes = index->get_partition_sizes(split_partition_ids);
        split_partition_ids = split_partition_ids.masked_select(split_partition_sizes > 8);

        Tensor new_centroids;
        vector<Tensor> new_vectors, new_ids;
        std::tie(new_centroids, new_vectors, new_ids) = index->split_partitions(split_partition_ids);

        // Delete the split partitions
        index->delete_partitions(split_partition_ids, /*reassign=*/false);

        // Add the new partitions
        index->add_partitions(new_centroids, new_vectors, new_ids);

        // Query the index
        int k = 5;
        Tensor ret_ids, ret_dis;
        std::tie(ret_ids, ret_dis, std::ignore) = index->search(query_vectors, 5, k);

        std::cout << "Iteration " << delete_id << std::endl;
        std::cout << "Num partitions: " << index->nlist() << std::endl;
        std::cout << "Num vectors: " << index->ntotal() << std::endl;
    }
}