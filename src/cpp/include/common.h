//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COMMON_H
#define COMMON_H

#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cassert>
#include <mutex>
#include <atomic>
#include <utility>
#include <unordered_map>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <faiss/MetricType.h>
#include <filesystem>
#include <unordered_set>
#include <sstream>
#include <thread>
#include <pthread.h>
#include <ctime>

#ifdef QUAKE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#ifdef FAISS_ENABLE_GPU
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#endif

#ifdef QUAKE_OMP
#include <omp.h>
#endif

using torch::Tensor;
using std::vector;
using std::shared_ptr;
using std::chrono::high_resolution_clock;
using faiss::idx_t;
using std::size_t;
using faiss::MetricType;

#endif //COMMON_H
