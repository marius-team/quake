//
// Created by Jason on 2/28/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARALLEL_H
#define PARALLEL_H

#include <future>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <iostream>


inline bool set_affinity_linux(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    return ret == 0;
}

#endif

inline bool set_thread_affinity(int core_id) {
#ifdef __APPLE__
    return false; // Not supported on macOS
#elif defined(__linux__)
    return set_affinity_linux(core_id);
#else
    std::cerr << "Platform not supported for setting thread affinity" << std::endl;
    return false;
#endif
}

template <typename IndexType, typename Function>
void parallel_for(IndexType start, IndexType end, Function func, int num_threads = -1) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    } else if (num_threads == 1) {
        for (IndexType i = start; i < end; i++) {
            func(i);
        }
    } else {
        IndexType total = end - start;
        IndexType chunk = (total + num_threads - 1) / num_threads;
        std::vector<std::future<void>> futures;
        for (IndexType t = 0; t < num_threads; ++t) {
            IndexType chunk_start = start + t * chunk;
            IndexType chunk_end = std::min(end, chunk_start + chunk);
            if (chunk_start >= chunk_end)
                break;
            futures.push_back(std::async(std::launch::async, [=]() {
                for (IndexType i = chunk_start; i < chunk_end; i++) {
                    func(i);
                }
            }));
        }
        for (auto &f : futures) {
            f.get();
        }
    }
}

static uint8_t* allocate_numa_buffer(size_t bytes, int target_numa_node, bool attempt_numa_allocation) {
    uint8_t* buffer = nullptr;
#ifdef QUAKE_USE_NUMA
    if (attempt_numa_allocation && target_numa_node != -1 && numa_available() != -1) {
        buffer = static_cast<uint8_t*>(numa_alloc_onnode(bytes, target_numa_node));
        if (!buffer) {
             // Fallback to standard allocation if numa_alloc_onnode fails (e.g. out of memory on that node)
            // std::cerr << "Warning: numa_alloc_onnode failed for node " << target_numa_node << ", falling back to std::malloc." << std::endl;
            buffer = static_cast<uint8_t*>(std::malloc(bytes));
        }
    } else {
        buffer = static_cast<uint8_t*>(std::malloc(bytes));
    }
#else
    (void)target_numa_node; // Suppress unused parameter warning
    (void)attempt_numa_allocation;
    buffer = static_cast<uint8_t*>(std::malloc(bytes));
#endif
    if (!buffer) {
        throw std::bad_alloc();
    }
    return buffer;
}

static void free_numa_buffer(uint8_t* buffer, size_t bytes, int allocated_on_node, bool was_numa_allocation) {
    if (!buffer) return;
#ifdef QUAKE_USE_NUMA
    if (was_numa_allocation && allocated_on_node != -1 && numa_available() != -1) {
        // Assuming numa_free doesn't strictly need the original node if it was allocated via numa_alloc_*
        numa_free(buffer, bytes);
    } else {
        std::free(buffer);
    }
#else
    (void)allocated_on_node; // Suppress unused parameter warning
    (void)was_numa_allocation;
    (void)bytes; // Not always needed by std::free
    std::free(buffer);
#endif
}

#endif //PARALLEL_H
