#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <numeric>   // For std::iota
#include <iomanip>   // For std::fixed, std::setprecision
#include <thread>    // For std::thread
#include <atomic>    // For std::atomic<bool>

#include <torch/torch.h>     // Assuming torch is available
#include "concurrentqueue.h" // Assuming moodycamel's queue header is findable

using namespace std::chrono;
using torch::Tensor;

// Namespace for this specific microbenchmark's helpers
namespace JobEnqueueMicrobenchmarkWithContention {

// --- Struct Definitions (ScanJob, CoreResources, MockPartitionManager - same as before) ---
    struct ScanJob {
        int64_t partition_id;
        int k_val;
        const float* query_vector;
        int query_id;
        bool is_batched = false;
        int64_t num_queries_in_job = 0;
        int rank = 0;

        ScanJob() = default;
        ScanJob(ScanJob&& other) noexcept = default;
        ScanJob& operator=(ScanJob&& other) noexcept = default;
        ScanJob(const ScanJob&) = delete;
        ScanJob& operator=(const ScanJob&) = delete;
    };

    struct CoreResources {
        int core_id_val;
        moodycamel::ConcurrentQueue<int64_t> job_queue;
        // For worker thread to count its dequeued items:
        std::atomic<long> dequeued_items_by_worker{0};


        // Explicitly define constructors for clarity if needed,
        // or rely on aggregate initialization / default members.
        CoreResources() : core_id_val(-1), job_queue(), dequeued_items_by_worker(0) {} // Default
        CoreResources(int id) : core_id_val(id), job_queue(), dequeued_items_by_worker(0) {} // With ID

        // Since moodycamel::ConcurrentQueue is not copyable,
        // CoreResources becomes non-copyable. We need move semantics.
        CoreResources(CoreResources&& other) noexcept
                : core_id_val(other.core_id_val),
                  job_queue(std::move(other.job_queue)), // Moodycamel queue is movable
                  dequeued_items_by_worker(other.dequeued_items_by_worker.load())
        {}

        CoreResources& operator=(CoreResources&& other) noexcept {
            if (this != &other) {
                core_id_val = other.core_id_val;
                job_queue = std::move(other.job_queue);
                dequeued_items_by_worker.store(other.dequeued_items_by_worker.load());
            }
            return *this;
        }
        // Delete copy operations
        CoreResources(const CoreResources&) = delete;
        CoreResources& operator=(const CoreResources&) = delete;
    };


    class MockPartitionManager {
    public:
        int num_cores_manager_ = 1;
        MockPartitionManager(int cores) : num_cores_manager_(cores) {
            if (cores <= 0) num_cores_manager_ = 1;
        }
        int get_partition_core_id(int64_t pid) const {
            return static_cast<int>(pid % num_cores_manager_);
        }
    };

// --- Parameters ---
    static const int64_t BENCH_DIMENSION = 128;
    static const int BENCH_K = 10;
    static const int BENCH_NUM_CORES = 4; // Number of worker threads and queues

// --- Worker Thread Function ---
    void dequeuing_worker_function(CoreResources& core_res, std::atomic<bool>& stop_signal) {
        long items_this_worker = 0;
        int64_t job_id_val; // Renamed to avoid conflict

        while (!stop_signal.load(std::memory_order_acquire)) {
            if (core_res.job_queue.try_dequeue(job_id_val)) {
                items_this_worker++;
                // Simulate minimal work with the job_id_val if needed
                volatile int64_t sink = job_id_val; // "Use" job_id_val
                (void)sink;
            } else {
                // Yield if the queue is often empty to reduce busy-waiting,
                // though try_dequeue is quite efficient.
                std::this_thread::yield();
            }
        }
        // After stop_signal is true, drain any remaining items from the queue
        while (core_res.job_queue.try_dequeue(job_id_val)) {
            items_this_worker++;
            volatile int64_t sink = job_id_val;
            (void)sink;
        }
        core_res.dequeued_items_by_worker.fetch_add(items_this_worker, std::memory_order_relaxed);
    }


// --- Function with the snippet to benchmark (Updated jid logic) ---
    int run_job_creation_and_enqueue_snippet(
            std::vector<ScanJob>& job_buffer,
            std::vector<CoreResources>& core_resources_list,
            const MockPartitionManager& partition_manager,
            const Tensor& partition_ids_tensor,
            const float* x_ptr,
            long num_queries,
            long num_partitions_per_query,
            int k_param,
            int dimension_param)
    {
        auto partition_ids_accessor = partition_ids_tensor.accessor<int64_t, 2>();
        job_buffer.resize(num_queries * num_partitions_per_query); // Max possible
        int jid = 0; // Current index for job_buffer, also basis for enqueued ID

        for (long q = 0; q < num_queries; q++) {
            for (long p = 0; p < num_partitions_per_query; p++) {
                int64_t pid_val = partition_ids_accessor[q][p];
                if (pid_val == -1) continue;

                ScanJob job_item;
                job_item.is_batched = false;
                job_item.query_id = static_cast<int>(q);
                job_item.partition_id = pid_val;
                job_item.k_val = k_param;
                job_item.query_vector = x_ptr + q * dimension_param;
                job_item.num_queries_in_job = 1;
                job_item.rank = static_cast<int>(p);

                int core_id = partition_manager.get_partition_core_id(pid_val);

                // Original snippet logic: place job, THEN increment jid, THEN enqueue the NEW jid.
                job_buffer[jid] = std::move(job_item);
                jid++; // jid is now the 1-based count of jobs created, or index for the *next* job.
                core_resources_list[core_id].job_queue.enqueue(jid); // Enqueue this 1-based count.
            }
        }
        return jid; // Returns the total count of jobs created (which is the last enqueued jid value)
    }


// --- GTest TEST Case with Contention ---
    TEST(JobEnqueueMicrobenchmark, SnippetPerformanceWithContention) {
        long num_queries_test = 100; // Increase for more work: e.g., 1000
        long num_partitions_test = 1000; // Increase for more work: e.g., 1000
        int test_iterations = 5;
        int warmup_iterations = 1;

        std::cout << std::endl << "[JobEnqueueMicrobenchmark::SnippetPerformanceWithContention]" << std::endl;
        std::cout << "  Config: Queries=" << num_queries_test
                  << ", Partitions/Query=" << num_partitions_test
                  << ", Iterations=" << test_iterations
                  << ", Worker Threads=" << BENCH_NUM_CORES << std::endl;

        // 1. Setup data structures
        std::vector<JobEnqueueMicrobenchmarkWithContention::ScanJob> local_job_buffer;
        std::vector<JobEnqueueMicrobenchmarkWithContention::CoreResources> local_core_resources;
        local_core_resources.reserve(BENCH_NUM_CORES);
        for (int i = 0; i < BENCH_NUM_CORES; ++i) {
            local_core_resources.emplace_back(i); // Use constructor CoreResources(int id)
        }
        JobEnqueueMicrobenchmarkWithContention::MockPartitionManager local_partition_manager(BENCH_NUM_CORES);

        std::vector<float> local_queries_data_storage(num_queries_test * BENCH_DIMENSION);
        float* local_x_ptr = local_queries_data_storage.data();
        Tensor local_partition_ids_tensor = torch::empty({num_queries_test, num_partitions_test}, torch::TensorOptions().dtype(torch::kInt64));
        auto accessor = local_partition_ids_tensor.accessor<int64_t, 2>();
        for (long i = 0; i < num_queries_test; ++i) {
            for (long j = 0; j < num_partitions_test; ++j) {
                accessor[i][j] = i * num_partitions_test + j;
            }
        }

        std::vector<double> durations_ms;
        durations_ms.reserve(test_iterations);
        long total_jobs_actually_created = 0;


        // 2. Run benchmark loop
        for (int iter = 0; iter < test_iterations + warmup_iterations; ++iter) {
            // Reset/drain queues and worker counts for each iteration
            for (auto& cr : local_core_resources) {
                int64_t dummy_val;
                while (cr.job_queue.try_dequeue(dummy_val));
                cr.dequeued_items_by_worker.store(0, std::memory_order_relaxed);
            }
            local_job_buffer.clear();

            std::atomic<bool> stop_worker_threads_signal{false};
            std::vector<std::thread> worker_threads;
            worker_threads.reserve(BENCH_NUM_CORES);

            // Launch worker threads
            for (int i = 0; i < BENCH_NUM_CORES; ++i) {
                worker_threads.emplace_back(dequeuing_worker_function, std::ref(local_core_resources[i]), std::ref(stop_worker_threads_signal));
            }

            auto iter_start_time = high_resolution_clock::now();

            // This is the timed operation (producer)
            int jobs_created_this_iteration = run_job_creation_and_enqueue_snippet(
                    local_job_buffer, local_core_resources, local_partition_manager,
                    local_partition_ids_tensor, local_x_ptr,
                    num_queries_test, num_partitions_test,
                    BENCH_K, BENCH_DIMENSION);

            auto iter_end_time = high_resolution_clock::now();

            // Signal workers to stop and wait for them to finish
            stop_worker_threads_signal.store(true, std::memory_order_release);
            for (auto& t : worker_threads) {
                if (t.joinable()) {
                    t.join();
                }
            }

            if (iter >= warmup_iterations) {
                durations_ms.push_back(duration_cast<microseconds>(iter_end_time - iter_start_time).count() / 1000.0);
                total_jobs_actually_created += jobs_created_this_iteration;
            }

            // Sanity check: total dequeued items should match jobs created
            long total_dequeued_this_iteration = 0;
            for (const auto& cr : local_core_resources) {
                total_dequeued_this_iteration += cr.dequeued_items_by_worker.load();
            }
            ASSERT_EQ(total_dequeued_this_iteration, jobs_created_this_iteration)
                                        << "Mismatch in created vs dequeued jobs for iteration " << iter;
        }

        // 3. Report results
        double total_duration_ms = 0;
        for (double d : durations_ms) total_duration_ms += d;
        double avg_duration_ms = durations_ms.empty() ? 0 : total_duration_ms / durations_ms.size();

        std::cout << "    Average time for snippet iteration (with contention): "
                  << std::fixed << std::setprecision(3) << avg_duration_ms << " ms" << std::endl;

        if (test_iterations > 0 && total_jobs_actually_created > 0) {
            long avg_jobs_per_iteration = total_jobs_actually_created / test_iterations;
            if (avg_jobs_per_iteration > 0 && avg_duration_ms > 0) {
                std::cout << "    Average jobs created per iteration: " << avg_jobs_per_iteration << std::endl;
                std::cout << "    Average time per job (snippet part, with contention): "
                          << (avg_duration_ms * 1000.0 / avg_jobs_per_iteration) << " us" << std::endl;
            }
        }
        ASSERT_GT(avg_duration_ms, -0.00001);
    }

} // namespace JobEnqueueMicrobenchmarkWithContention