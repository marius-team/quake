#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <vector>
#include <pthread.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <ctime>
#include <numeric>
#include <functional>
#include <cmath>

#include "blockingconcurrentqueue.h"

inline int64_t getCurrentTime() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

void worker_thread(int thread_id, moodycamel::BlockingConcurrentQueue<int>& job_queue, int64_t* result_times, int* result_workers) {
    // Pin thread to a CPU
    int num_cpus = std::thread::hardware_concurrency();
    int worker_cpu = thread_id % num_cpus;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(worker_cpu, &cpuset);
    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::string err_msg = "Unable to bind worker " + std::to_string(thread_id) + " to cpu " + std::to_string(worker_cpu);
        throw std::runtime_error(err_msg);
    }

    // Verify the thread is running on the correct CPU
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::runtime_error("Failed to get thread affinity");
    }

    if (!CPU_ISSET(worker_cpu, &cpuset)) {
        throw std::runtime_error("Thread not running on the specified CPU");
    }
    
    int job_id;
    while(true) {
       job_queue.wait_dequeue(job_id);
        if(job_id == -1) {
            break;
        }

        // Save the job finish time
        result_workers[job_id] = thread_id;
        result_times[2 * job_id + 1] = getCurrentTime();
    }
}

std::tuple<float, float> get_summary(std::vector<float> timings) {
    float sum = std::accumulate(timings.begin(), timings.end(), 0.0);
    float mean = sum / (1.0 * timings.size());

    std::vector<float> diff(timings.size());
    std::transform(timings.begin(), timings.end(), diff.begin(), [mean](float x) { return x - mean; });
    float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float std_dev = std::sqrt(sq_sum / (1.0 * timings.size()));
    return std::make_tuple(mean, std_dev);
}

int main() {
    // Create the job queue
    int num_workers = 8;
    int num_jobs = 8;
    moodycamel::BlockingConcurrentQueue<int> job_queue;
    std::cout << "Running benchmark with " << num_jobs << " with " << num_workers << " workers" << std::endl;

    // Launch the workers
    omp_set_num_threads(1);
    int64_t* result_times = new int64_t[2 * num_jobs];
    int* result_workers = new int[num_jobs];
    std::vector<std::thread> queue_reader_workers;
    for(int i = 0; i < num_workers; i++) {
        queue_reader_workers.emplace_back(&worker_thread, i, std::ref(job_queue), result_times, result_workers);
    }

    // Sleep for a bit
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Push the jobs
    for(int i = 0; i < num_jobs; i++) {
        job_queue.enqueue(i);
        result_times[2 * i] = getCurrentTime();
    }

    // Submit the finish job
    for(int i = 0; i < num_workers; i++) {
        job_queue.enqueue(-1);
    }

    // Join the workers
    for (std::thread &worker: queue_reader_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    // Log the summary
    std::vector<float> all_times_passed;
    for(int i = 0; i < num_jobs; i++) {
        int64_t job_submit_time = result_times[2 * i];
        int64_t job_receive_time = result_times[2 * i + 1];
        float time_passed_us = (job_receive_time - job_submit_time)/1000.0;
        all_times_passed.push_back(time_passed_us);
        std::cout << "Job " << i << " details: Submit Time - " << job_submit_time << ", Receive Time - " << job_receive_time << ", Queue Time - " << time_passed_us << ", Worker - " << result_workers[i] << std::endl;
    }

    std::tuple<float, float> times_summary = get_summary(all_times_passed);
    std::cout << "Times passed distribution of " << std::get<0>(times_summary) << " +/- " << std::get<1>(times_summary) << " uS" << std::endl;

    delete[] result_times;
}
