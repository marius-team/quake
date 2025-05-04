#ifndef POLICY_H
#define POLICY_H

#include <vector>

class Policy {
    public:
        virtual std::vector<int> findVictims() = 0;
        virtual void insert(int pid) = 0;
        virtual ~Policy() {};
    private:
        virtual void remove(int pid) = 0; // called by findVictim
};

#endif