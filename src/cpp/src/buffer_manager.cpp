#include <query_coordinator.h>
#include <buffer_manager.h>
#include <fifo_policy.h>
#include <lru_policy.h>

// for destructor, flush all pages


BufferManager::BufferManager() {
    std::cout << "Constructing buffer manager" << std::endl;
    bufSize = 1;
    policy = make_shared<LRUPolicy>();
    curSize = 0;
}

BufferManager::~BufferManager()
{
    std::cout << "Flush contents? " << std::endl;
}

void BufferManager::put(int pid, shared_ptr<FileIndexPartition> fip, shared_ptr<PartitionManager> partition_manager_) {
    std::cout << "[BufferManager::put] Called with partition id: " << pid << std::endl;
    if(curSize == bufSize) { // Buffer is full
        auto victims = policy->findVictims();
        for(auto victim_pid : victims) {
            if(debug_) {
                std::cout << "Evicting partition ID: " << victim_pid << std::endl;
            }
            u.erase(victim_pid);
            shared_ptr<FileIndexPartition> victim_fip = nullptr;
            auto it = partition_manager_->partitions_->partitions_.find(victim_pid);
            if (it == partition_manager_->partitions_->partitions_.end()) {
                throw std::runtime_error("victim pid does not exist");
            }
            victim_fip = std::dynamic_pointer_cast<FileIndexPartition>(partition_manager_->partitions_->partitions_[victim_pid]);
            if(victim_fip) victim_fip->save();
        }
    } else {
        curSize ++; // Probably need locks around all these variables
    }
    policy->insert(pid);
    std::cout << "Inserted partion ID: " << pid << " into the buffer" << std::endl;
    u.insert(pid);
    fip->load();
    if(debug_) {
        std::cout << "Curent buffer state (not ordered): { ";
        for (const int& element : u) {
            std::cout << element << " ";
        }
        std::cout << " }" << std::endl;
    }
}

void BufferManager::flush(int pid, shared_ptr<FileIndexPartition> fip) {
    std::cout << "[BufferManager] flush implementation goes here" << std::endl;
}

void BufferManager::evict() {
    std::cout << "[BufferManager] evict implementation goes here" << std::endl;
}