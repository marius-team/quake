#ifndef LRU_POLICY_H
#define LRU_POLICY_H

#include <policy.h>
#include <unordered_map>
#include <list>
#include <vector>
#include <iostream>

class LRUPolicy : public Policy {
public:
    LRUPolicy();
    ~LRUPolicy();

    std::vector<int> findVictims() override;
    void insert(int pid) override;

private:
    void remove(int pid) override;

    std::list<int> lru_list_;  // most recent at back, least recent at front
    std::unordered_map<int, std::list<int>::iterator> lru_map_;
    bool debug_ = true;
};

#endif  // LRU_POLICY_H
