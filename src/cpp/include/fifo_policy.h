#ifndef FIFO_POLICY_H
#define FIFO_POLICY_H

#include <policy.h>
#include <queue>

class FIFOPolicy : public Policy {
    public:
        std::queue<int> fifo_q; // who to evict
        bool debug_ = true;


        FIFOPolicy();
        ~FIFOPolicy();
        std::vector<int> findVictims();
        void insert(int pid);

    private:
        void remove(int pid); // called by findVictim
};

#endif