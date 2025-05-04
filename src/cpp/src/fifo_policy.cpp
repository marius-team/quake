#include "fifo_policy.h"
#include <iostream>

FIFOPolicy::FIFOPolicy() {
    std::cout << "FIFO Policy constructor" << std::endl;
};

std::vector<int> FIFOPolicy::findVictims() {
    auto q_front = fifo_q.front();
    fifo_q.pop();
    return std::vector<int>(1, q_front);
}

void FIFOPolicy::insert(int pid) {
    fifo_q.push(pid);
}

FIFOPolicy::~FIFOPolicy() {
}

void FIFOPolicy::remove(int pid)
{
    std::cout << "Remove implementation goes here" << std::endl;
}