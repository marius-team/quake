#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include <unordered_set>
#include <queue>
#include <partition_manager.h>
#include <file_index_partition.h>
#include <policy.h>

// decide who to evict
// virtual class Policy {
//     public:
//         std::queue<int> q; // who to evict
//         std::vector findVictims();
//         void insert(int pid);
//     private:
//         void remove(int pid); // called by findVictim
// }

// actually putting and removing things in buffer
class BufferManager {
    public:
        std::unordered_set<int> u; // who is in the buffer (needs a better name - Atharva)
        shared_ptr<Policy> policy;
        int bufSize; // number of partitions in the memory
        int curSize;
        bool debug_ = true;
        

        BufferManager();
        ~BufferManager();

        void put(int pid, shared_ptr<FileIndexPartition> fip, shared_ptr<PartitionManager> partition_manager_); // load the vectors and ids of a partition from disk
        void flush(int pid, shared_ptr<FileIndexPartition> fip); // flush the vectors and ids of a partition to disk, while still keeping in memory
        
    private:
        void evict(); // evict a partition based on the eviction policy, this will be called by putBuf if the buffer is full
};

#endif