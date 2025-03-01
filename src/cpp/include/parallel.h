//
// Created by Jason on 2/28/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARALLEL_H
#define PARALLEL_H

#include <future>
#include <vector>
#include <algorithm>

template <typename IndexType, typename Function>
void parallel_for(IndexType start, IndexType end, Function func) {
    const auto num_threads = std::thread::hardware_concurrency();
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

#endif //PARALLEL_H
