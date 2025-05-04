#include "lru_policy.h"

LRUPolicy::LRUPolicy() {
    std::cout << "LRU Policy constructor" << std::endl;
}

LRUPolicy::~LRUPolicy() = default;

void LRUPolicy::insert(int pid) {
    // Remove if already present
    auto it = lru_map_.find(pid);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
    }

    // Insert to back (most recently used)
    lru_list_.push_back(pid);
    lru_map_[pid] = std::prev(lru_list_.end());

    if (debug_) {
        std::cout << "[LRUPolicy::insert] Current buffer: ";
        for (int id : lru_list_) std::cout << id << " ";
        std::cout << std::endl;
    }
}

std::vector<int> LRUPolicy::findVictims() {
    if (lru_list_.empty()) return {};
    int victim = lru_list_.front();  // Least recently used
    remove(victim);
    return {victim};
}

void LRUPolicy::remove(int pid) {
    auto it = lru_map_.find(pid);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
        lru_map_.erase(it);
    }
}
