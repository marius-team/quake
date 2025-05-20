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


#ifdef QUAKE_USE_NUMA
#include <numa.h>
  #include <sys/mman.h>
  #include <new>

  inline void* quake_alloc(size_t sz, int node) {
    void* ptr = numa_alloc_onnode(sz, node);
    if (ptr == nullptr || ptr == MAP_FAILED) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  inline void quake_free(void* ptr, size_t sz) noexcept {
    if (ptr) {
      numa_free(ptr, sz);
    }
  }

    inline int cpu_numa_node(int cpu) {
        return numa_node_of_cpu(cpu);
    }

    inline int get_num_numa_nodes() {
        return numa_num_configured_nodes();
    }

#else
#include <cstdlib>
#include <new>

inline void* quake_alloc(size_t sz, int /*node*/) {
    void* ptr = std::malloc(sz);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

inline void quake_free(void* ptr, size_t /*sz*/) noexcept {
    std::free(ptr);
}

inline int cpu_numa_node(int /*cpu*/) {
    return 0; // Not applicable
}

inline int get_num_numa_nodes() {
    return 1; // Not applicable
}
#endif


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

#endif //PARALLEL_H
