#include "file_index_partition.h"
#include <fstream>

// Parameterized constructor
FileIndexPartition::FileIndexPartition(int64_t num_vectors,
                                       uint8_t* codes,
                                       idx_t* ids,
                                       int64_t code_size) {
    // Implementation here
}

// Move constructor
FileIndexPartition::FileIndexPartition(FileIndexPartition&& other) noexcept {
    // Implementation here
}

// Move assignment operator
FileIndexPartition& FileIndexPartition::operator=(FileIndexPartition&& other) noexcept {
    // Implementation here
    return *this;
}

// Destructor
FileIndexPartition::~FileIndexPartition() {
    // munmap stuff needs to happen here?
}

void FileIndexPartition::append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) {
    // std::cout << "Appending to FileIndexPartition file (" << file_path_ << ") goes here" << std::endl;
    if (n_entry <= 0) return;

    std::ofstream ofs(file_path_, std::ios::binary | std::ios::app); // append mode
    if (!ofs) {
        throw std::runtime_error("Failed to open file for appending: " + file_path_);
    }

    const size_t code_bytes = static_cast<size_t>(code_size_);
    for (int64_t i = 0; i < n_entry; ++i) {
        ofs.write(reinterpret_cast<const char*>(new_codes + i * code_bytes), code_bytes);
        ofs.write(reinterpret_cast<const char*>(&new_ids[i]), sizeof(idx_t));
    }
    num_vectors_ += n_entry;

    ofs.close();
}

void FileIndexPartition::update(int64_t offset, int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) {
    // Implementation here
}

void FileIndexPartition::remove(int64_t index) {
    // Implementation here
    // if (index < 0 || index >= num_vectors_) {
    //     throw std::runtime_error("Index out of range in remove");
    // }
    // if (index == num_vectors_ - 1) {
    //     num_vectors_--;
    //     return;
    // }

    // int64_t last_idx = num_vectors_ - 1;
    // const size_t code_bytes = static_cast<size_t>(code_size_);

    // if (is_in_memory) {
    //     std::lock_guard<std::mutex> lock(ref_mutex);
    //     std::cout << "[File_index_partition] remove : Removing index " << index << " of partition ID (in memory) " << file_path_ << std::endl; 
    //     std::memcpy(codes_ + index * code_bytes, codes_ + last_idx * code_bytes, code_bytes);
    //     ids_[index] = ids_[last_idx];
    //     is_dirty = true;
    //     num_vectors_--;
    // }
    // else {
    //     std::cout << "[File_index_partition] remove : Removing index " << index << " of file path " << file_path_ << std::endl; 
    //     std::fstream file(file_path_, std::ios::in | std::ios::out | std::ios::binary);
    //     if (!file) {
    //         throw std::runtime_error("Failed to open file for remove: " + file_path_);
    //     }

    //     std::vector<uint8_t> last_code(code_bytes);
    //     idx_t last_id;
        
    //     // seek the last entry
    //     file.seekg(last_idx * (code_bytes + sizeof(idx_t)), std::ios::beg);
    //     file.read(reinterpret_cast<char*>(last_code.data()), code_bytes);
    //     file.read(reinterpret_cast<char*>(&last_id), sizeof(idx_t));
    
    //     // seek the deleted entry and replace with the last entry
    //     file.seekp(index * (code_bytes + sizeof(idx_t)), std::ios::beg);
    //     file.write(reinterpret_cast<const char*>(last_code.data()), code_bytes);
    //     file.write(reinterpret_cast<const char*>(&last_id), sizeof(idx_t));
    
    //     file.close();
    //     num_vectors_--;
    // }

}

void FileIndexPartition::resize(int64_t new_capacity) {
    // Implementation here
}

void FileIndexPartition::clear() {
    // Implementation here
}

int64_t FileIndexPartition::find_id(idx_t id) const {
    // Implementation here
    return -1; // Placeholder return
}

void FileIndexPartition::reallocate(int64_t new_capacity) {
    // Implementation here
}

// for testing
void FileIndexPartition::load() {
    // std::cout << "[FileIndexPartition] load" << std::endl;
    std::ifstream in(file_path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Unable to open file for reading: " + file_path_);
    }

    std::lock_guard<std::mutex> lock(ref_mutex);
    
    ref_cnt ++;
    if (is_in_memory) return;

    ensure_capacity(num_vectors_); // allocate memory for codes_ and ids_

    for (int64_t i = 0; i < num_vectors_; ++i) {
        in.read(reinterpret_cast<char*>(codes_ + i * code_size_), code_size_);
        in.read(reinterpret_cast<char*>(ids_ + i), sizeof(idx_t));
    }

    in.close();
    is_in_memory = true;
}

// decrement the reference bit, if its zero called the buffer manager to flush the file
// the buffer manager should loop over the buffer pool and evict all buffers that belongs to the file
void FileIndexPartition::save() {
    std::lock_guard<std::mutex> lock(ref_mutex);
    ref_cnt --;
    if (ref_cnt == 0) {
        buffer_size_ = 0;
        free_memory(); // for now, we don't have a dedicated buffer pool so just free the memory
        is_in_memory = false;
    }

    if (is_dirty) {
        std::ofstream out(file_path_, std::ios::binary); // if num_vectors_ is maintained correctly, no need to clear all vectors before writing back
        if (!out) {
            throw std::runtime_error("Unable to open file for writing");
        }
    
        const size_t code_bytes = static_cast<size_t>(code_size_);
        for (int64_t i = 0; i < num_vectors_; ++i) {
            out.write(reinterpret_cast<const char*>(codes_ + i * code_bytes), code_bytes);
            out.write(reinterpret_cast<const char*>(&ids_[i]), sizeof(idx_t));
        }

        out.close();
    }
}

void FileIndexPartition::set_file_path(std::string file_path) {
    file_path_ = file_path;
}

// void FileIndexPartition::free_memory() {
//     if (codes_ == nullptr && ids_ == nullptr) {
//         return;
//     }
//     std::free(codes_);
//     std::free(ids_);
    
//     codes_ = nullptr;
//     ids_ = nullptr;
// }

#ifdef QUAKE_USE_NUMA
void FileIndexPartition::set_numa_node(int new_numa_node) {
    // Implementation here
}
#endif
