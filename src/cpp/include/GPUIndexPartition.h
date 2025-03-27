#ifndef GPU_INDEX_PARTITION_H
#define GPU_INDEX_PARTITION_H

#include "index_partition.h"
#include <cuda_runtime.h>

class GPUIndexPartition : public IndexPartition {
public:
    GPUIndexPartition() : IndexPartition() {}

    // helper: allocate memory on GPU
    template <typename T>
    T* allocate_gpu_memory(size_t num_elements) {
        T* ptr = nullptr;
        size_t total_bytes = num_elements * sizeof(T);
        cudaError_t err = cudaMalloc(&ptr, total_bytes);
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    // override reallocate_memory to use cudaMalloc and cudaMemcpy
    virtual void reallocate_memory(int64_t new_capacity) override {
        if (new_capacity < num_vectors_) {
            num_vectors_ = new_capacity;
        }
        const size_t code_bytes = static_cast<size_t>(code_size_);
        int64_t curr_count = num_vectors_;

        uint8_t* new_codes = allocate_gpu_memory<uint8_t>(new_capacity * code_bytes);
        idx_t* new_ids = allocate_gpu_memory<idx_t>(new_capacity);

        if (codes_ && ids_) {
            cudaMemcpy(new_codes, codes_, curr_count * code_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(new_ids, ids_, curr_count * sizeof(idx_t), cudaMemcpyDeviceToDevice);
        }

        free_memory();

        codes_ = new_codes;
        ids_ = new_ids;
        buffer_size_ = new_capacity;
    }

    // override free_memory to use cudaFree
    virtual void free_memory() override {
        if (codes_) {
            cudaFree(codes_);
            codes_ = nullptr;
        }
        if (ids_) {
            cudaFree(ids_);
            ids_ = nullptr;
        }
        buffer_size_ = 0;
        num_vectors_ = 0;
    }
};

#endif
