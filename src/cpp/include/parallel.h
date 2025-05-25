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

    inline int get_current_cpu_numa_node() {
        if (numa_available() < 0) return 0; // NUMA not available or error
        int current_cpu = sched_getcpu();
        if (current_cpu < 0) {
            // perror("sched_getcpu failed"); // User prefers less output
            return -1; // Error
        }
        return numa_node_of_cpu(current_cpu);
    }

    // Helper to get the NUMA node of a memory address
    inline int get_memory_numa_node(const void* ptr) {
        if (numa_available() < 0) return 0; // NUMA not available or error
        if (!ptr) {
            return -1; // Invalid pointer (e.g. NULL)
        }

        // Align pointer to page boundary for numa_move_pages query
        void* page_address = (void*)((uintptr_t)ptr & ~((uintptr_t)getpagesize() - 1));

        void *pages_to_query[] = { page_address };
        int status_array[] = { -1 }; // To store the node of the page

        errno = 0; // Clear errno before system call
        // pid = 0 for current process's address space
        // nodes = NULL to query current node
        // flags = 0 for query
        if (numa_move_pages(0, 1, pages_to_query, NULL, status_array, 0) == 0) {
            // Success. status_array[0] contains the node.
            // It can be -1 if the page has a policy like MPOL_INTERLEAVE or is not mapped to a specific node.
            return status_array[0];
        } else {
            // numa_move_pages failed, return negative errno.
            // errno is set by numa_move_pages.
            // Common errors: EFAULT (-14), EINVAL (-22), EPERM (-1), ESRCH (-3), ENOMEM (-12)
            return -errno;
        }
}

    // Verifies if the memory at memory_address is on the same NUMA node as the current CPU.
    // variable_name is for logging/debugging by the caller if verification fails.
    inline bool verify_numa_locality(const void* memory_address, const char* variable_name) {
        if (numa_available() < 0) return true; // NUMA not available, assume locality
        if (get_num_numa_nodes() <= 1) return true; // Single NUMA node system

        if (!memory_address) {
            std::cout << "Warning: Cannot verify NUMA locality for null pointer: " << variable_name << std::endl;
            return true; // Or false, based on desired strictness for nullptrs. True avoids false positives.
        }

        int current_cpu_node = get_current_cpu_numa_node();
        if (current_cpu_node < 0) {
            std::cout << "Error: Could not determine current CPU NUMA node for " << variable_name << std::endl;
            return false; // Cannot verify
        }

        int memory_page_node = get_memory_numa_node(memory_address);
        if (memory_page_node < -1) { // -2 indicates query error
            std::cout << "memory_page_node = " << memory_page_node << std::endl;
            std::cout << "Error: Could not determine NUMA node for memory of " << variable_name << std::endl;
            return false; // Cannot verify due to error
        }
        if (memory_page_node == -1) {
            // This typically means the page is not mapped or has a default/interleaved policy
            // that doesn't map to a single specific node in a way numa_move_pages can report.
            // For strict checking, this could be considered non-local if cpu_node is specific.
            // Caller can log: "Warning: Memory for " << variable_name << " has undetermined/interleaved policy (node -1). CPU node: " << current_cpu_node
            // A common case for MPOL_DEFAULT is allocation on the node of first touch.
            // If it's truly interleaved, it's not "local" to any single node.
            // If it's default and first touched by current_cpu_node, it would be local.
            // This check is tricky. For now, if we can't determine a specific node, assume it might not be local.
            std::cout << "Warning: Memory for " << variable_name << " has undetermined/interleaved policy (node -1). CPU node: " << current_cpu_node << std::endl;
            return false;
        }

        bool is_local = (current_cpu_node == memory_page_node);

        if (is_local) {
            // std::cout << "NUMA locality verified for " << variable_name << std::endl;
        } else {
            std::cout << "Warning: NUMA locality mismatch for " << variable_name
                      << ": CPU is on node " << current_cpu_node
                      << ", but memory is on node " << memory_page_node << std::endl;
        }

        return is_local;
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

inline bool verify_numa_locality(const void* /*memory_address*/, const char* /*variable_name*/) {
    return true;
}
inline int get_current_cpu_numa_node() { return 0; }
inline int get_memory_numa_node(const void* /*ptr*/) { return 0; }

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
