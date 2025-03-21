// benchmark.cpp
//
// This file benchmarks four main operations (build, search, add, remove)
// for two types of indexes (Flat and IVF) using both Quake and Faiss.
//
// For Quake, a flat index is built with build_params->nlist == 1,
// and an IVF index is built with nlist > 1.
// For Faiss, we use IndexFlatL2 for flat and IndexIVFFlat for IVF.

#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>

#include <torch/torch.h>
#include "quake_index.h"  // Quake API header

// Faiss headers
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/Index.h>

using namespace std::chrono;
using torch::Tensor;

// Global benchmark parameters
static const int64_t DIM = 128;
static const int64_t NUM_VECTORS = 100000;   // number of database vectors
static const int64_t N_LIST = 100;           // number of clusters for IVF
static const int64_t NUM_QUERIES = 1000;     // number of queries for search benchmark
static const int64_t K = 10;                  // top-K neighbors
static const int64_t N_PROBE = 32;             // number of probes for IVF
static const int64_t N_WORKERS = 12;           // number of workers for parallel query coordinator

// Helper functions to generate random data and sequential IDs
static Tensor generate_data(int64_t num, int64_t dim) {
    return torch::randn({num, dim}, torch::kFloat32);
}

static Tensor generate_ids(int64_t num, int64_t start = 0) {
    return torch::arange(start, start + num, torch::kInt64);
}

//
// ===== Quake BENCHMARK FIXTURES =====
//

// Quake Flat: using build_params->nlist == 1
class QuakeSerialFlatBenchmark : public ::testing::Test {
protected:
    std::shared_ptr<QuakeIndex> index_;
    Tensor data_;
    Tensor ids_;
    void SetUp() override {
        data_ = generate_data(NUM_VECTORS, DIM);
        ids_ = generate_ids(NUM_VECTORS);
        index_ = std::make_shared<QuakeIndex>();
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = 1;      // flat index
        build_params->metric = "l2";
        index_->build(data_, ids_, build_params);
    }
};

// Quake Flat with workers (parallel query coordinator)
class QuakeWorkerFlatBenchmark : public ::testing::Test {
    protected:
    std::shared_ptr<QuakeIndex> index_;
    Tensor data_;
    Tensor ids_;
    void SetUp() override {
        data_ = generate_data(NUM_VECTORS, DIM);
        ids_ = generate_ids(NUM_VECTORS);
        index_ = std::make_shared<QuakeIndex>();
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = 1;      // flat index
        build_params->metric = "l2";
        // Use as many workers as hardware concurrency
        build_params->num_workers = N_WORKERS;
        index_->build(data_, ids_, build_params);
    }
};


// Quake IVF (serial): using build_params->nlist > 1 and no workers.
class QuakeSerialIVFBenchmark : public ::testing::Test {
protected:
    std::shared_ptr<QuakeIndex> index_;
    Tensor data_;
    Tensor ids_;
    void SetUp() override {
        data_ = generate_data(NUM_VECTORS, DIM);
        ids_ = generate_ids(NUM_VECTORS);
        index_ = std::make_shared<QuakeIndex>();
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = N_LIST;     // IVF index
        build_params->metric = "l2";
        build_params->niter = 3;
        index_->build(data_, ids_, build_params);
    }
};

// Quake IVF with workers (parallel query coordinator)
class QuakeWorkerIVFBenchmark : public ::testing::Test {
protected:
    std::shared_ptr<QuakeIndex> index_;
    Tensor data_;
    Tensor ids_;
    void SetUp() override {
        data_ = generate_data(NUM_VECTORS, DIM);
        ids_ = generate_ids(NUM_VECTORS);
        index_ = std::make_shared<QuakeIndex>();
        auto build_params = std::make_shared<IndexBuildParams>();
        build_params->nlist = N_LIST;     // IVF index
        build_params->metric = "l2";
        build_params->niter = 3;
        build_params->num_workers = N_WORKERS;
        index_->build(data_, ids_, build_params);
    }
};

//
// ===== Faiss BENCHMARK FIXTURES =====
//

// For Faiss Flat we use IndexFlatL2
class FaissFlatBenchmark : public ::testing::Test {
protected:
    std::unique_ptr<faiss::IndexFlatL2> index_;
    Tensor data_;
    int64_t dim_;
    void SetUp() override {
        dim_ = DIM;
        data_ = generate_data(NUM_VECTORS, dim_);
        index_ = std::make_unique<faiss::IndexFlatL2>(dim_);
        index_->add(NUM_VECTORS, data_.data_ptr<float>());
    }
};

// For Faiss IVF we use IndexIVFFlat; note that IVF requires training.
class FaissIVFBenchmark : public ::testing::Test {
protected:
    std::unique_ptr<faiss::IndexIVFFlat> index_;
    Tensor data_;
    int64_t dim_;
    void SetUp() override {
        dim_ = DIM;
        data_ = generate_data(NUM_VECTORS, dim_);
        // Create a quantizer index (flat L2)
        auto quantizer = new faiss::IndexFlatL2(dim_);
        int nlist = N_LIST;
        index_.reset(new faiss::IndexIVFFlat(quantizer, dim_, nlist, faiss::METRIC_L2));
        index_->train(NUM_VECTORS, data_.data_ptr<float>());
        index_->add(NUM_VECTORS, data_.data_ptr<float>());
    }
};

//
// ===== Quake BENCHMARK TESTS =====
//

TEST_F(QuakeSerialFlatBenchmark, Search) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = 1;  // not used for flat index
    search_params->batched_scan = false;

    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat Serial] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeSerialFlatBenchmark, SearchBatch) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = 1;  // not used for flat index
    search_params->batched_scan = true;

    auto start = high_resolution_clock::now();
    auto result = index_->search(queries, search_params);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat Serial] Batched search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeWorkerFlatBenchmark, Search) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = 1;  // not used for flat index
    search_params->batched_scan = false;

    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat Worker] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeWorkerFlatBenchmark, SearchBatch) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = 1;  // not used for flat index
    search_params->batched_scan = true;

    index_->search(queries, search_params);

    auto start = high_resolution_clock::now();
    auto result = index_->search(queries, search_params);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat Worker] Batched search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeSerialIVFBenchmark, Search) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = N_PROBE;
    search_params->batched_scan = false;

    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF Serial] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeSerialIVFBenchmark, SearchBatch) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = N_PROBE;
    search_params->batched_scan = true;

    index_->search(queries, search_params);
    auto start = high_resolution_clock::now();
    auto result = index_->search(queries, search_params);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF Serial] Batched search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeWorkerIVFBenchmark, Search) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = N_PROBE;
    // search_params->recall_target = .75;
    // search_params->aps_flush_period_us = 1;
    // For worker-based search, batched_scan can be false (or true) depending on your implementation.
    search_params->batched_scan = false;

    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        auto result = index_->search(queries[i].unsqueeze(0), search_params);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF Worker] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeWorkerIVFBenchmark, SearchBatch) {
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = K;
    search_params->nprobe = N_PROBE;
    search_params->batched_scan = true;

    index_->search(queries, search_params);

    auto start = high_resolution_clock::now();
    auto result = index_->search(queries, search_params);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF Worker] Batched search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(QuakeSerialFlatBenchmark, Add) {
    int64_t num_add = NUM_VECTORS / 10;
    Tensor add_data = generate_data(num_add, DIM);
    Tensor add_ids = generate_ids(num_add, NUM_VECTORS);

    auto start = high_resolution_clock::now();
    auto modify_info = index_->add(add_data, add_ids);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat] Add time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(modify_info->modify_time_us, 0);
}

TEST_F(QuakeSerialIVFBenchmark, Add) {
    int64_t num_add = NUM_VECTORS / 10;
    Tensor add_data = generate_data(num_add, DIM);
    Tensor add_ids = generate_ids(num_add, NUM_VECTORS);

    auto start = high_resolution_clock::now();
    auto modify_info = index_->add(add_data, add_ids);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF] Add time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(modify_info->modify_time_us, 0);
}

TEST_F(QuakeSerialFlatBenchmark, Remove) {
    Tensor remove_ids = ids_.slice(0, 0, NUM_VECTORS / 2);
    auto start = high_resolution_clock::now();
    auto modify_info = index_->remove(remove_ids);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake Flat] Remove time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(modify_info->modify_time_us, 0);
}

TEST_F(QuakeSerialIVFBenchmark, Remove) {
    Tensor remove_ids = ids_.slice(0, 0, NUM_VECTORS / 2);
    auto start = high_resolution_clock::now();
    auto modify_info = index_->remove(remove_ids);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "[Quake IVF] Remove time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(modify_info->modify_time_us, 0);
}

//
// ===== Faiss BENCHMARK TESTS =====
//
TEST_F(FaissFlatBenchmark, Search) {
    int64_t k = K;
    std::vector<float> distances(NUM_QUERIES * k);
    std::vector<faiss::idx_t> labels(NUM_QUERIES * k);
    Tensor queries = generate_data(NUM_QUERIES, DIM);

    for (int i = 0; i < queries.size(0); i++) {
        index_->search(1, queries[i].data_ptr<float>(), k, distances.data() + i * k, labels.data() + i * k);
    }
    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        index_->search(1, queries[i].data_ptr<float>(), k, distances.data() + i * k, labels.data() + i * k);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss Flat] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissFlatBenchmark, SearchBatch) {
    int64_t k = K;
    std::vector<float> distances(NUM_QUERIES * k);
    std::vector<faiss::idx_t> labels(NUM_QUERIES * k);
    Tensor queries = generate_data(NUM_QUERIES, DIM);

    index_->search(NUM_QUERIES, queries.data_ptr<float>(), k, distances.data(), labels.data());
    auto start = high_resolution_clock::now();
    index_->search(NUM_QUERIES, queries.data_ptr<float>(), k, distances.data(), labels.data());
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss Flat] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissFlatBenchmark, Add) {
    int64_t num_add = NUM_VECTORS / 10;
    Tensor add_data = generate_data(num_add, DIM);
    auto start = high_resolution_clock::now();
    index_->add(num_add, add_data.data_ptr<float>());
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss Flat] Add time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissFlatBenchmark, Remove) {
    int64_t num_remove = NUM_VECTORS / 2;
    Tensor remove_ids = generate_ids(num_remove);
    std::vector<faiss::idx_t> ids_to_remove(remove_ids.data_ptr<faiss::idx_t>(),
                                             remove_ids.data_ptr<faiss::idx_t>() + num_remove);
    auto start = high_resolution_clock::now();
    auto sel = faiss::IDSelectorBatch(num_remove, ids_to_remove.data());
    index_->remove_ids(sel);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss Flat] Remove time: " << elapsed << " ms" << std::endl;
    SUCCEED();
}

TEST_F(FaissIVFBenchmark, Search) {
    int64_t k = K;
    std::vector<float> distances(NUM_QUERIES * k);
    std::vector<faiss::idx_t> labels(NUM_QUERIES * k);
    index_->nprobe = N_PROBE;
    Tensor queries = generate_data(NUM_QUERIES, DIM);

    for (int i = 0; i < queries.size(0); i++) {
        index_->search(1, queries[i].data_ptr<float>(), k, distances.data() + i * k, labels.data() + i * k);
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < queries.size(0); i++) {
        index_->search(1, queries[i].data_ptr<float>(), k, distances.data() + i * k, labels.data() + i * k);
    }
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss IVF] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissIVFBenchmark, SearchBatch) {
    int64_t k = K;
    std::vector<float> distances(NUM_QUERIES * k);
    std::vector<faiss::idx_t> labels(NUM_QUERIES * k);
    index_->nprobe = N_PROBE;
    Tensor queries = generate_data(NUM_QUERIES, DIM);
    auto start = high_resolution_clock::now();
    index_->search(NUM_QUERIES, queries.data_ptr<float>(), k, distances.data(), labels.data());
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss IVF] Search time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissIVFBenchmark, Add) {
    int64_t num_add = NUM_VECTORS / 10;
    Tensor add_data = generate_data(num_add, DIM);
    auto start = high_resolution_clock::now();
    index_->add(num_add, add_data.data_ptr<float>());
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss IVF] Add time: " << elapsed << " ms" << std::endl;
    ASSERT_GT(elapsed, 0);
}

TEST_F(FaissIVFBenchmark, Remove) {
    int64_t num_remove = NUM_VECTORS / 2;
    Tensor remove_ids = generate_ids(num_remove);
    std::vector<faiss::idx_t> ids_to_remove(remove_ids.data_ptr<faiss::idx_t>(),
                                             remove_ids.data_ptr<faiss::idx_t>() + num_remove);
    auto start = high_resolution_clock::now();
    auto sel = faiss::IDSelectorBatch(num_remove, ids_to_remove.data());
    index_->remove_ids(sel);
    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<milliseconds>(end - start).count();
    std::cout << "[Faiss IVF] Remove time: " << elapsed << " ms" << std::endl;
    SUCCEED();
}