#include <faiss/impl/FaissAssert.h>
// #include <omp.h> // OpenMP no longer used for the primary parallelization
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include <mutex>     // For std::mutex and std::call_once
#include <thread>    // For std::thread::hardware_concurrency
#include <future>    // For std::async, std::future (used by parallel.h)
#include <gtest/gtest.h>
#include "faiss/utils/distances.h"
#include "parallel.h" // Include your custom parallel_for header

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
static constexpr int64_t LIST_SIZE   = 1'000'000;
static constexpr int      D          = 128;
static constexpr int      NQ         = 100; // Number of queries

// Forward declaration of the fixture
class ListScanBenchmarkF;

// ------------------------------------------------------------------
// Timing helper
// ------------------------------------------------------------------
struct Timer {
    using clk = std::chrono::steady_clock;
    clk::time_point t0;
    void   start() { t0 = clk::now();               }
    double stop()  { return std::chrono::duration<double>(clk::now()-t0).count(); }
};

// ------------------------------------------------------------------
// Pretty printer
// ------------------------------------------------------------------
// Takes requested threads (nth_req) and actual threads used by parallel_for (pfor_threads)
static void print_stats(const char* label, int nth_req, double sec)
{
    const double evaluations   = static_cast<double>(LIST_SIZE) * NQ;
    const double bytes_scanned = evaluations * D * sizeof(float);
    const double gb_scanned    = bytes_scanned / (1ULL << 30);
    const double gb_per_sec    = gb_scanned / sec;

    std::cout << std::setw(24) << std::left  << label
              << " | " << std::setw(2) << nth_req
              << " | " << std::fixed << std::setprecision(3)
              << sec  << " s"
              << " | " << std::setprecision(2)
              << gb_per_sec << " GB/s"
              << std::endl;
}

// ==================================================================
// Google‑test Test Fixture
// ==================================================================
class ListScanBenchmarkF : public ::testing::Test {
protected:
    // Data members are static members of the fixture
    static std::vector<float>   S_base_vectors_data;
    static std::vector<float>   S_queries_data;
    static std::vector<int64_t> S_seq_order_data;
    static std::vector<int64_t> S_rnd_order_data;

    // Flags for std::call_once, also static members
    static std::once_flag base_vectors_flag;
    static std::once_flag queries_flag;
    static std::once_flag seq_order_flag;
    static std::once_flag rnd_order_flag;

    static void initialize_base_vectors() {
        std::call_once(base_vectors_flag, []{
            S_base_vectors_data.resize(LIST_SIZE * D);
            std::mt19937 rng(123);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float& v : S_base_vectors_data) v = nd(rng);
        });
    }

    static void initialize_queries() {
        std::call_once(queries_flag, []{
            S_queries_data.resize(NQ * D);
            std::mt19937 rng(321);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float& v : S_queries_data) v = nd(rng);
        });
    }

    static void initialize_seq_order() {
        std::call_once(seq_order_flag, []{
            S_seq_order_data.resize(LIST_SIZE);
            std::iota(S_seq_order_data.begin(), S_seq_order_data.end(), 0);
        });
    }

    static void initialize_rnd_order() {
        std::call_once(rnd_order_flag, []{
            S_rnd_order_data.resize(LIST_SIZE);
            std::iota(S_rnd_order_data.begin(), S_rnd_order_data.end(), 0);
            std::mt19937 rng(999);
            std::shuffle(S_rnd_order_data.begin(), S_rnd_order_data.end(), rng);
        });
    }

public:
    // Kernel now uses the custom parallel_for for the inner loop (scanning list_size)
    // It returns the sum of minimum distances found for each query (for anti-optimization)
    static float static_scan_kernel(const std::vector<int64_t>& order, int n_threads_requested)
    {
        const float* base_ptr = S_base_vectors_data.data();
        const float* query_ptr    = S_queries_data.data();

        volatile float total_min_sum = 0.f; // Accumulates min distance for each query

        // Determine the actual number of threads parallel_for will aim to use
        int threads_for_pfor = n_threads_requested;
        if (threads_for_pfor <= 0) {
            threads_for_pfor = static_cast<int>(std::thread::hardware_concurrency());
        }
        if (threads_for_pfor == 0) { // Safety for hardware_concurrency returning 0
             threads_for_pfor = 1;
        }

        float distances[LIST_SIZE]; // Distances for the current query

        // Outer loop over queries (sequential)
        for (int qi = 0; qi < NQ; ++qi) {
            const float* current_query_vector = query_ptr + qi * D;

            // Inner loop (over base vectors for the current query) is parallelized
            // The 'order' vector contains indices into 'S_base_vectors_data'
            // 'k' will be an index from 0 to order.size()-1
            parallel_for(static_cast<int64_t>(0), static_cast<int64_t>(order.size()),
                [&](int64_t k) { // k is the index in the range [0, order.size())
                    int64_t base_vector_original_idx = order[k];
                    const float* current_base_vector = base_ptr + base_vector_original_idx * D;

                    distances[k] = faiss::fvec_L2sqr(current_query_vector, current_base_vector, static_cast<size_t>(D));
                },
                threads_for_pfor // Pass the determined number of threads to parallel_for
            );
        }
        return total_min_sum; // Return sum for anti-optimization
    }

    // Wrappers call the kernel
    static float static_sequential_scan(int nth = 1) {
        return static_scan_kernel(S_seq_order_data, nth);
    }
    static float static_random_scan(int nth = 1) {
        return static_scan_kernel(S_rnd_order_data, nth);
    }

protected:
    static void SetUpTestSuite() {
        std::cout << "Using custom parallel_for for inner loop parallelization." << std::endl;
        unsigned int hw_threads = std::thread::hardware_concurrency();
        std::cout << "std::thread::hardware_concurrency(): " << hw_threads << std::endl;

        std::cout << "Initializing benchmark data (once)..." << std::endl;
        auto overall_start_init = std::chrono::steady_clock::now();

        initialize_base_vectors();
        initialize_queries();
        initialize_seq_order();
        initialize_rnd_order();

        auto overall_end_init = std::chrono::steady_clock::now();
        std::chrono::duration<double> init_duration = overall_end_init - overall_start_init;
        std::cout << "VECTOR SCAN BENCHMARK" << std::endl;
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        std::cout << "Benchmark data initialized in " << std::fixed << std::setprecision(3)
                  << init_duration.count() << " s." << std::endl;
        std::cout << "Benchmark data size: " << LIST_SIZE << "vectors, "
                  << NQ << " queries, " << D << " dimensions." << std::endl;
        std::cout << "Throughput averaged over all queries." << std::endl;
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        std::cout << std::setw(24) << std::left << "Test Case"
                  << " | " << "N_Threads" // Requested threads
                  << " | " << "Time (s)"
                  << " | " << "Throughput" << std::endl;
        std::cout << "--------------------------------------------------------------------------" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "--------------------------------------------------------------------------" << std::endl;
        std::cout << "Benchmark suite finished." << std::endl;
    }
};

// Static member definitions
std::vector<float>   ListScanBenchmarkF::S_base_vectors_data;
std::vector<float>   ListScanBenchmarkF::S_queries_data;
std::vector<int64_t> ListScanBenchmarkF::S_seq_order_data;
std::vector<int64_t> ListScanBenchmarkF::S_rnd_order_data;

std::once_flag ListScanBenchmarkF::base_vectors_flag;
std::once_flag ListScanBenchmarkF::queries_flag;
std::once_flag ListScanBenchmarkF::seq_order_flag;
std::once_flag ListScanBenchmarkF::rnd_order_flag;



TEST_F(ListScanBenchmarkF, ParallelSequentialScan)
{
    for (int nth : {1, 2, 4, 8, 12}) { // Test with different numbers of threads
        Timer t;
        t.start();
        volatile float dummy_sum = static_sequential_scan(nth);
        double elapsed_seconds = t.stop();
        (void)dummy_sum;

        int actual_pfor_threads = (nth <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : nth;
        if (actual_pfor_threads == 0) actual_pfor_threads = 1;
         if (nth == 1) actual_pfor_threads = 1; // parallel_for handles num_threads == 1 specifically


        print_stats("Parallel-Sequential", nth, elapsed_seconds);
    }
}

TEST_F(ListScanBenchmarkF, ParallelRandomScan)
{
    for (int nth : {1, 2, 4, 8, 12}) { // Test with different numbers of threads
        Timer t;
        t.start();
        volatile float dummy_sum = static_random_scan(nth);
        double elapsed_seconds = t.stop();
        (void)dummy_sum;

        int actual_pfor_threads = (nth <= 0) ? static_cast<int>(std::thread::hardware_concurrency()) : nth;
        if (actual_pfor_threads == 0) actual_pfor_threads = 1;
        if (nth == 1) actual_pfor_threads = 1; // parallel_for handles num_threads == 1 specifically

        print_stats("Parallel-Random", nth, elapsed_seconds);
    }
}
