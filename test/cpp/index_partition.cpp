// index_partition_test.cpp

#include <gtest/gtest.h>
#include "index_partition.h"  // Include the IndexPartition header
#include <vector>
#include <cstring>

using namespace faiss;

class IndexPartitionTest : public ::testing::Test {
protected:
    int64_t initial_num_vectors = 10;
    int64_t code_size = 16; // bytes per code
    IndexPartition* partition;

    // Vectors to hold initial codes and ids for verification
    std::vector<uint8_t> initial_codes_vec_;
    std::vector<idx_t> initial_ids_vec_;

    virtual void SetUp() {
        // Initialize initial_ids with sequential IDs starting from 1000
        generate_sequential_ids(initial_num_vectors, initial_ids_vec_, 1000);

        // Initialize initial_codes with sequential codes starting from 0
        generate_sequential_codes(initial_num_vectors, initial_codes_vec_, 0);

        // Allocate memory and copy initial data
        uint8_t* initial_codes = static_cast<uint8_t*>(std::malloc(initial_num_vectors * code_size));
        idx_t* initial_ids = static_cast<idx_t*>(std::malloc(initial_num_vectors * sizeof(idx_t)));
        std::memcpy(initial_codes, initial_codes_vec_.data(), initial_num_vectors * code_size);
        std::memcpy(initial_ids, initial_ids_vec_.data(), initial_num_vectors * sizeof(idx_t));

        // Initialize an IndexPartition with initial data
        partition = new IndexPartition(initial_num_vectors, initial_codes, initial_ids, code_size);

        // Free temporary allocations as IndexPartition has its own copies
        std::free(initial_codes);
        std::free(initial_ids);
    }

    virtual void TearDown() {
        delete partition;
    }

    // Helper function to generate sequential codes
    void generate_sequential_codes(size_t n, std::vector<uint8_t>& codes, unsigned int start_val = 0) {
        codes.resize(n * code_size);
        for (size_t i = 0; i < n * code_size; ++i) {
            codes[i] = static_cast<uint8_t>((start_val + i) % 256);
        }
    }

    // Helper function to generate sequential IDs
    void generate_sequential_ids(size_t n, std::vector<idx_t>& ids, idx_t start_id = 0) {
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i] = start_id + i;
        }
    }

    // Helper functions for verification
    void verify_ids(const idx_t* actual_ids, const std::vector<idx_t>& expected_ids, size_t start_idx = 0) {
        for (size_t i = 0; i < expected_ids.size(); ++i) {
            EXPECT_EQ(actual_ids[start_idx + i], expected_ids[i]) << "Mismatch at index " << (start_idx + i);
        }
    }

    void verify_codes(const uint8_t* actual_codes, const std::vector<uint8_t>& expected_codes, size_t start_idx = 0) {
        size_t code_bytes = code_size;
        for (size_t i = 0; i < expected_codes.size() / code_bytes; ++i) {
            EXPECT_EQ(std::memcmp(actual_codes + (start_idx + i) * code_bytes,
                                  expected_codes.data() + i * code_bytes,
                                  code_bytes), 0)
                << "Code mismatch at vector " << (start_idx + i);
        }
    }
};

// Test default constructor
TEST_F(IndexPartitionTest, DefaultConstructorTest) {
    IndexPartition default_partition;
    EXPECT_EQ(default_partition.buffer_size_, 0);
    EXPECT_EQ(default_partition.num_vectors_, 0);
    EXPECT_EQ(default_partition.code_size_, 0);
    EXPECT_EQ(default_partition.codes_, nullptr);
    EXPECT_EQ(default_partition.ids_, nullptr);
    EXPECT_EQ(default_partition.numa_node_, -1);
    EXPECT_EQ(default_partition.core_id_, -1);
}

// Test parameterized constructor
TEST_F(IndexPartitionTest, ParameterizedConstructorTest) {
    size_t num_vectors = 5;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_sequential_codes(num_vectors, codes, 10);
    generate_sequential_ids(num_vectors, ids, 5000);

    IndexPartition param_partition(num_vectors, codes.data(), ids.data(), code_size);

    EXPECT_EQ(param_partition.num_vectors_, num_vectors);
    EXPECT_EQ(param_partition.code_size_, code_size);
    EXPECT_NE(param_partition.codes_, nullptr);
    EXPECT_NE(param_partition.ids_, nullptr);

    // Verify initial data
    verify_ids(param_partition.ids_, ids, 0);
    verify_codes(param_partition.codes_, codes, 0);
}

// Test append method
TEST_F(IndexPartitionTest, AppendTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(n_entry, new_codes, 500);
    generate_sequential_ids(n_entry, new_ids, 6000);

    // Append entries
    partition->append(n_entry, new_ids.data(), new_codes.data());

    EXPECT_EQ(partition->num_vectors_, initial_num_vectors + n_entry);

    // Verify initial data remains unchanged
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);

    // Verify appended data
    size_t append_start = initial_num_vectors;
    verify_ids(partition->ids_, new_ids, append_start);
    verify_codes(partition->codes_, new_codes, append_start);
}

// Test append with exceeding initial buffer size
TEST_F(IndexPartitionTest, AppendExceedBufferTest) {
    size_t n_entry = initial_num_vectors + 5; // Intentionally large to test resizing
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(n_entry, new_codes, 600);
    generate_sequential_ids(n_entry, new_ids, 7000);

    // Append entries
    partition->append(n_entry, new_ids.data(), new_codes.data());

    EXPECT_EQ(partition->num_vectors_, initial_num_vectors + n_entry);

    // Verify initial data remains unchanged
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);

    // Verify appended data
    size_t append_start = initial_num_vectors;
    verify_ids(partition->ids_, new_ids, append_start);
    verify_codes(partition->codes_, new_codes, append_start);
}

// Test update method
TEST_F(IndexPartitionTest, UpdateTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 700);
    generate_sequential_ids(n_entry, append_ids, 8000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Prepare new codes and ids for update
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(n_entry, new_codes, 900);
    generate_sequential_ids(n_entry, new_ids, 9000);

    // Update entries at offset = initial_num_vectors (10)
    size_t offset = initial_num_vectors;
    partition->update(offset, n_entry, new_ids.data(), new_codes.data());

    // Verify updated data
    verify_ids(partition->ids_, new_ids, offset);
    verify_codes(partition->codes_, new_codes, offset);

    // Verify initial data remains unchanged
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);
}

// Test update with out-of-range offset
TEST_F(IndexPartitionTest, UpdateOutOfRangeTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1000);
    generate_sequential_ids(n_entry, append_ids, 10000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Prepare new codes and ids
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_sequential_codes(n_entry, new_codes, 1100);
    generate_sequential_ids(n_entry, new_ids, 11000);

    // Attempt to update with offset + n_entry > num_vectors_
    size_t offset = partition->num_vectors_ - 2; // 15 -2=13
    size_t invalid_n_entry = 3; // 13 +3=16 >15

    EXPECT_THROW(partition->update(offset, invalid_n_entry, new_ids.data(), new_codes.data()), std::runtime_error);
}

// Test remove method
TEST_F(IndexPartitionTest, RemoveTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1100);
    generate_sequential_ids(n_entry, append_ids, 12000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    size_t expected_num_vectors = initial_num_vectors + n_entry;
    EXPECT_EQ(partition->num_vectors_, expected_num_vectors);

    // Remove element at index = initial_num_vectors + 2 (12)
    size_t remove_idx = initial_num_vectors + 2;
    partition->remove(remove_idx);

    EXPECT_EQ(partition->num_vectors_, expected_num_vectors - 1);

    // Verify that the last element was swapped into position remove_idx
    size_t last_idx = expected_num_vectors - 1;
    EXPECT_EQ(partition->ids_[remove_idx], append_ids[n_entry - 1]);
    EXPECT_EQ(std::memcmp(partition->codes_ + remove_idx * code_size, append_codes.data() + (n_entry - 1) * code_size, code_size), 0);

    // Verify that the last element has been removed
    for (size_t i = 0; i < partition->num_vectors_; ++i) {
        if (i == remove_idx) continue;
        EXPECT_NE(partition->ids_[i], append_ids[n_entry - 1]);
    }
}

// Test remove with out-of-range index
TEST_F(IndexPartitionTest, RemoveOutOfRangeTest) {
    size_t n_entry = 3;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1200);
    generate_sequential_ids(n_entry, append_ids, 13000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Attempt to remove with invalid index
    size_t invalid_idx = initial_num_vectors + n_entry; // Out of range
    EXPECT_THROW(partition->remove(invalid_idx), std::runtime_error);
}

// Test resize method
TEST_F(IndexPartitionTest, ResizeTest) {
    size_t new_capacity = 20;
    partition->resize(new_capacity);

    EXPECT_GE(partition->buffer_size_, new_capacity);

    // Verify existing data is intact
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);
}

// Test resize to smaller than num_vectors_
TEST_F(IndexPartitionTest, ResizeSmallerThanNumVectorsTest) {
    size_t n_entry = 10;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1300);
    generate_sequential_ids(n_entry, append_ids, 14000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Resize to smaller capacity
    size_t new_capacity = initial_num_vectors + 5; // Less than current num_vectors (10+10=20 >15)
    partition->resize(new_capacity);

    EXPECT_EQ(partition->buffer_size_, new_capacity);
    EXPECT_EQ(partition->num_vectors_, new_capacity);

    // Verify first `new_capacity` entries are correct
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);

    // Since `resize` truncates, appended entries beyond `new_capacity` are lost
    for (size_t i = 0; i < 5; ++i) { // Only first 5 appended entries are kept
        EXPECT_EQ(partition->ids_[initial_num_vectors + i], append_ids[i]);
        EXPECT_EQ(std::memcmp(partition->codes_ + (initial_num_vectors + i) * code_size,
                              append_codes.data() + i * code_size,
                              code_size), 0)
            << "Code mismatch at index " << (initial_num_vectors + i);
    }
}

// Test clear method
TEST_F(IndexPartitionTest, ClearTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1400);
    generate_sequential_ids(n_entry, append_ids, 15000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Clear the partition
    partition->clear();

    EXPECT_EQ(partition->buffer_size_, 0);
    EXPECT_EQ(partition->num_vectors_, 0);
    EXPECT_EQ(partition->code_size_, 0);
    EXPECT_EQ(partition->codes_, nullptr);
    EXPECT_EQ(partition->ids_, nullptr);
    EXPECT_EQ(partition->numa_node_, -1);
    EXPECT_EQ(partition->core_id_, -1);
}

// Test find_id method
TEST_F(IndexPartitionTest, FindIdTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1500);
    generate_sequential_ids(n_entry, append_ids, 16000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Find existing IDs
    for (size_t i = 0; i < initial_num_vectors; ++i) {
        EXPECT_EQ(partition->find_id(initial_ids_vec_[i]), static_cast<int64_t>(i));
    }
    for (size_t i = 0; i < n_entry; ++i) {
        EXPECT_EQ(partition->find_id(append_ids[i]), static_cast<int64_t>(initial_num_vectors + i));
    }

    // Find non-existing ID
    EXPECT_EQ(partition->find_id(99999), -1);
}

// Test set_code_size method before adding vectors
TEST_F(IndexPartitionTest, SetCodeSizeBeforeAddingVectorsTest) {
    // Create a new partition with no vectors
    IndexPartition empty_partition;

    // Set code_size_
    int64_t new_code_size = 32;
    empty_partition.set_code_size(new_code_size);

    EXPECT_EQ(empty_partition.code_size_, new_code_size);
}

// Test set_code_size method after adding vectors (should throw)
TEST_F(IndexPartitionTest, SetCodeSizeAfterAddingVectorsTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1500);
    generate_sequential_ids(n_entry, append_ids, 16000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Attempt to set_code_size after adding vectors
    int64_t new_code_size = 32;
    EXPECT_THROW(partition->set_code_size(new_code_size), std::runtime_error);
}

// Test set_code_size with invalid size
TEST_F(IndexPartitionTest, SetCodeSizeInvalidTest) {
    int64_t invalid_code_size = 0;
    EXPECT_THROW(partition->set_code_size(invalid_code_size), std::runtime_error);
}

// Test move semantics do not leak or double free
TEST_F(IndexPartitionTest, MoveSemanticsTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1600);
    generate_sequential_ids(n_entry, append_ids, 17000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Total vectors: initial_num_vectors + n_entry = 10 + 5 = 15
    EXPECT_EQ(partition->num_vectors_, 15);

    // Move construct a new partition
    IndexPartition moved_partition(std::move(*partition));

    // Original partition should now be empty
    EXPECT_EQ(partition->buffer_size_, 0);
    EXPECT_EQ(partition->num_vectors_, 0);
    EXPECT_EQ(partition->codes_, nullptr);
    EXPECT_EQ(partition->ids_, nullptr);

    // Moved partition should have all 15 vectors
    EXPECT_EQ(moved_partition.num_vectors_, 15);

    // Verify initial data remains unchanged
    verify_ids(moved_partition.ids_, initial_ids_vec_, 0);
    verify_codes(moved_partition.codes_, initial_codes_vec_, 0);

    // Verify appended data
    size_t append_start = initial_num_vectors;
    verify_ids(moved_partition.ids_, append_ids, append_start);
    verify_codes(moved_partition.codes_, append_codes, append_start);

    // Move assign back to original
    *partition = std::move(moved_partition);

    // Moved_partition should now be empty
    EXPECT_EQ(moved_partition.buffer_size_, 0);
    EXPECT_EQ(moved_partition.num_vectors_, 0);
    EXPECT_EQ(moved_partition.codes_, nullptr);
    EXPECT_EQ(moved_partition.ids_, nullptr);

    // Original partition should have all 15 vectors again
    EXPECT_EQ(partition->num_vectors_, 15);
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);
    verify_ids(partition->ids_, append_ids, append_start);
    verify_codes(partition->codes_, append_codes, append_start);
}

// Test that append properly resizes multiple times
TEST_F(IndexPartitionTest, AppendMultipleTimesTest) {
    size_t n_entry = 5;
    size_t append_times = 3;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;

    // Separate vectors to accumulate expected data
    std::vector<uint8_t> expected_codes;
    std::vector<idx_t> expected_ids;

    for (size_t t = 0; t < append_times; ++t) {
        // Generate current batch of codes and ids
        std::vector<uint8_t> current_codes;
        std::vector<idx_t> current_ids;
        generate_sequential_codes(n_entry, current_codes, static_cast<uint8_t>(t * 100));
        generate_sequential_ids(n_entry, current_ids, static_cast<idx_t>(t * 1000));

        // Append to the partition
        partition->append(n_entry, current_ids.data(), current_codes.data());

        // Accumulate expected data
        expected_ids.insert(expected_ids.end(), current_ids.begin(), current_ids.end());
        expected_codes.insert(expected_codes.end(), current_codes.begin(), current_codes.end());
    }

    // Total vectors: initial_num_vectors + n_entry * append_times
    EXPECT_EQ(partition->num_vectors_, initial_num_vectors + n_entry * append_times);

    // Verify all appended data
    size_t append_start = initial_num_vectors;
    for (size_t t = 0; t < append_times; ++t) {
        for (size_t i = 0; i < n_entry; ++i) {
            size_t idx = append_start + t * n_entry + i;
            EXPECT_EQ(partition->ids_[idx], expected_ids[t * n_entry + i])
                << "Mismatch at index " << idx;
            EXPECT_EQ(std::memcmp(partition->codes_ + idx * code_size,
                                  expected_codes.data() + t * n_entry * code_size + i * code_size,
                                  code_size), 0)
                << "Code mismatch at index " << idx;
        }
    }
}

// Test duplicate IDs
// Test duplicate IDs
TEST_F(IndexPartitionTest, DuplicateIdsTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids = {100, 101, 102, 100, 104}; // Duplicate ID 100
    generate_sequential_codes(n_entry, append_codes, 1700);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Find first occurrence
    size_t first_occurrence_idx = partition->find_id(100);
    EXPECT_EQ(first_occurrence_idx, initial_num_vectors) << "First occurrence of ID 100 should be at index " << initial_num_vectors;

    // Remove first occurrence
    partition->remove(first_occurrence_idx);

    // Now, find_id should return the new position of 100, which was at index initial_num_vectors + 3 after removal
    size_t expected_new_index = initial_num_vectors + 3; // Original index of second 100 is 13
    size_t found_index = partition->find_id(100);
    EXPECT_EQ(found_index, expected_new_index) << "After removal, ID 100 should be found at index " << expected_new_index;

    // Verify the updated ID at the removed index now holds 104
    EXPECT_EQ(partition->ids_[first_occurrence_idx], 104) << "After removal, index " << first_occurrence_idx << " should hold ID 104";
}

// Test that update does not affect other entries
TEST_F(IndexPartitionTest, UpdateDoesNotAffectOthersTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1800);
    generate_sequential_ids(n_entry, append_ids, 18000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Update the third entry
    size_t update_idx = initial_num_vectors + 2;
    std::vector<uint8_t> new_codes(code_size, 255);
    std::vector<idx_t> new_ids = {99999};
    partition->update(update_idx, 1, new_ids.data(), new_codes.data());

    // Verify updated entry
    EXPECT_EQ(partition->ids_[update_idx], new_ids[0]);
    EXPECT_EQ(std::memcmp(partition->codes_ + update_idx * code_size, new_codes.data(), code_size), 0);

    // Verify that other entries remain unchanged
    for (size_t i = 0; i < partition->num_vectors_; ++i) {
        if (i == update_idx) continue;
        if (i < initial_num_vectors) {
            EXPECT_EQ(partition->ids_[i], initial_ids_vec_[i]);
            EXPECT_EQ(std::memcmp(partition->codes_ + i * code_size, initial_codes_vec_.data() + i * code_size, code_size), 0);
        } else {
            size_t appended_idx = i - initial_num_vectors;
            EXPECT_EQ(partition->ids_[i], append_ids[appended_idx]);
            EXPECT_EQ(std::memcmp(partition->codes_ + i * code_size, append_codes.data() + appended_idx * code_size, code_size), 0);
        }
    }
}

// Test that removing all entries leads to an empty partition
TEST_F(IndexPartitionTest, RemoveAllEntriesTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 1900);
    generate_sequential_ids(n_entry, append_ids, 19000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Remove all entries
    size_t initial_total = partition->num_vectors_;
    for (size_t i = 0; i < initial_total; ++i) {
        partition->remove(0); // Always remove the first element
    }

    EXPECT_EQ(partition->num_vectors_, 0);
}

// Test that append with zero entries does not change the partition
TEST_F(IndexPartitionTest, AppendWithZeroEntriesTest) {
    size_t n_entry = 0;
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;

    // Append zero entries
    partition->append(n_entry, new_ids.data(), new_codes.data());

    // Verify that partition remains unchanged
    EXPECT_EQ(partition->num_vectors_, initial_num_vectors);
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);
}

// Test updating with zero entries throws an exception
TEST_F(IndexPartitionTest, UpdateWithZeroEntriesTest) {
    size_t n_entry = 5;
    std::vector<uint8_t> append_codes;
    std::vector<idx_t> append_ids;
    generate_sequential_codes(n_entry, append_codes, 2000);
    generate_sequential_ids(n_entry, append_ids, 20000);

    partition->append(n_entry, append_ids.data(), append_codes.data());

    // Attempt to update with zero entries
    size_t offset = initial_num_vectors;
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;

    // Expect an exception when n_entry is zero
    EXPECT_THROW(partition->update(offset, 0, new_ids.data(), new_codes.data()), std::runtime_error);

    // Verify that data remains unchanged
    EXPECT_EQ(partition->num_vectors_, initial_num_vectors + n_entry);
    verify_ids(partition->ids_, initial_ids_vec_, 0);
    verify_codes(partition->codes_, initial_codes_vec_, 0);

    verify_ids(partition->ids_, append_ids, initial_num_vectors);
    verify_codes(partition->codes_, append_codes, initial_num_vectors);
}

TEST_F(IndexPartitionTest, AppendStressTest) {
    const size_t stress_count = 10000;
    std::vector<uint8_t> stress_codes;
    std::vector<idx_t> stress_ids;
    generate_sequential_codes(stress_count, stress_codes, 200);
    generate_sequential_ids(stress_count, stress_ids, 50000);

    partition->append(stress_count, stress_ids.data(), stress_codes.data());

    EXPECT_EQ(partition->num_vectors_, initial_num_vectors + stress_count);
    // Verify that the very first appended entry is correct.
    EXPECT_EQ(partition->ids_[initial_num_vectors], stress_ids[0]);
}

TEST_F(IndexPartitionTest, ConcurrentFindIdTest) {
    const size_t thread_count = 8;
    std::atomic<bool> error_found{false};

    auto worker = [this, &error_found]() {
        for (int iter = 0; iter < 1000; iter++) {
            // For each initial ID, verify that find_id returns the expected index.
            for (size_t j = 0; j < initial_ids_vec_.size(); j++) {
                int64_t idx = partition->find_id(initial_ids_vec_[j]);
                if (idx != static_cast<int64_t>(j)) {
                    error_found = true;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < thread_count; i++) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_FALSE(error_found);
}

#ifdef QUAKE_USE_NUMA
#include <numa.h>

bool verify_numa(void *ptr, int target_node) {
    int node = -1;
    int ret = get_mempolicy(&node, nullptr, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    if (ret < 0) {
        return false;
    }
    return (node == target_node);
}

class IndexPartitionNumaTest : public ::testing::Test {
protected:
    int64_t initial_num_vectors = 10;
    int64_t code_size = 16;  // bytes per code
    IndexPartition* partition;

    std::vector<uint8_t> initial_codes_vec_;
    std::vector<idx_t> initial_ids_vec_;

    virtual void SetUp() {
        if (numa_available() == -1) {
            GTEST_SKIP() << "NUMA not available on this system.";
        }
        // Prepare initial sequential IDs and codes.
        generate_sequential_ids(initial_num_vectors, initial_ids_vec_, 1000);
        generate_sequential_codes(initial_num_vectors, initial_codes_vec_, 0);

        // Allocate temporary buffers and copy the data.
        uint8_t* initial_codes = static_cast<uint8_t*>(std::malloc(initial_num_vectors * code_size));
        idx_t* initial_ids = static_cast<idx_t*>(std::malloc(initial_num_vectors * sizeof(idx_t)));
        std::memcpy(initial_codes, initial_codes_vec_.data(), initial_num_vectors * code_size);
        std::memcpy(initial_ids, initial_ids_vec_.data(), initial_num_vectors * sizeof(idx_t));

        // Construct the partition.
        partition = new IndexPartition(initial_num_vectors, initial_codes, initial_ids, code_size);
        std::free(initial_codes);
        std::free(initial_ids);
    }

    virtual void TearDown() {
        delete partition;
    }

    // Helper: generate sequential codes.
    void generate_sequential_codes(size_t n, std::vector<uint8_t>& codes, unsigned int start_val = 0) {
        codes.resize(n * code_size);
        for (size_t i = 0; i < n * code_size; ++i) {
            codes[i] = static_cast<uint8_t>((start_val + i) % 256);
        }
    }

    // Helper: generate sequential IDs.
    void generate_sequential_ids(size_t n, std::vector<idx_t>& ids, idx_t start_id = 0) {
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i] = start_id + i;
        }
    }
};

// Verify that set_numa_node re-allocates memory on the target node while preserving data.
// We also check that the memory is actually bound to the target node via verify_numa.
TEST_F(IndexPartitionNumaTest, SetNumaNodeContentPreservation) {
    EXPECT_EQ(partition->numa_node_, -1) << "Initial numa_node_ should be -1.";

    // Save original pointer addresses.
    uint8_t* orig_codes = partition->codes_;
    idx_t* orig_ids = partition->ids_;

    // Choose a target node (e.g. node 0).
    int target_node = 0;
    partition->set_numa_node(target_node);

    EXPECT_EQ(partition->numa_node_, target_node);
    EXPECT_NE(partition->codes_, orig_codes) << "Memory for codes should be reallocated on new NUMA node.";
    EXPECT_NE(partition->ids_, orig_ids) << "Memory for ids should be reallocated on new NUMA node.";

    // Verify that the allocated memory is bound to the target node.
    EXPECT_TRUE(verify_numa(partition->codes_, target_node))
        << "Codes memory not bound to target node " << target_node;
    EXPECT_TRUE(verify_numa(partition->ids_, target_node))
        << "IDs memory not bound to target node " << target_node;

    // Verify that the stored data remains intact.
    EXPECT_EQ(std::memcmp(partition->codes_, initial_codes_vec_.data(), partition->num_vectors_ * code_size), 0);
    EXPECT_EQ(std::memcmp(partition->ids_, initial_ids_vec_.data(), partition->num_vectors_ * sizeof(idx_t)), 0);
}

// Verify that calling set_numa_node with the same node value is a no-op.
// The pointers should remain unchanged, and the binding should still be correct.
TEST_F(IndexPartitionNumaTest, SetNumaNodeNoOp) {
    int target_node = 0;
    partition->set_numa_node(target_node);

    uint8_t* codes_after_first = partition->codes_;
    idx_t* ids_after_first = partition->ids_;

    // Calling with the same node should not change pointers.
    partition->set_numa_node(target_node);
    EXPECT_EQ(partition->codes_, codes_after_first);
    EXPECT_EQ(partition->ids_, ids_after_first);
    EXPECT_TRUE(verify_numa(partition->codes_, target_node));
    EXPECT_TRUE(verify_numa(partition->ids_, target_node));
}

//
// Verify that setting an invalid NUMA node value throws an exception.
TEST_F(IndexPartitionNumaTest, SetNumaNodeInvalid) {
    int max_node = numa_max_node();
    int invalid_node = max_node + 1;  // Should be invalid.
    EXPECT_THROW(partition->set_numa_node(invalid_node), std::runtime_error);
}

// After clearing the partition, setting the NUMA node should update the field
// without reallocating memory (since no data exists).
TEST_F(IndexPartitionNumaTest, SetNumaNodeAfterClear) {
    partition->clear();
    EXPECT_EQ(partition->codes_, nullptr);
    EXPECT_EQ(partition->ids_, nullptr);

    int target_node = 1;
    partition->set_numa_node(target_node);
    EXPECT_EQ(partition->numa_node_, target_node);
    EXPECT_EQ(partition->codes_, nullptr);
    EXPECT_EQ(partition->ids_, nullptr);
}

// Cycle through multiple NUMA nodes and verify that each call updates the binding.
// We iterate through several nodes (using target_node = i % (max_node+1)) and check that the pointers change
// (after the first iteration) and that the binding is correct.
TEST_F(IndexPartitionNumaTest, SetNumaNodeMultipleTest) {
    int max_node = numa_max_node();
    const int iterations = 3;  // Test with three different node values.
    for (int i = 0; i < iterations; ++i) {
        int target_node = i % (max_node + 1);
        // Save current pointer values to compare later.
        uint8_t* prev_codes = partition->codes_;
        idx_t* prev_ids = partition->ids_;

        partition->set_numa_node(target_node);

        EXPECT_EQ(partition->numa_node_, target_node);
        EXPECT_TRUE(verify_numa(partition->codes_, target_node))
            << "Codes memory not bound to target node " << target_node;
        EXPECT_TRUE(verify_numa(partition->ids_, target_node))
            << "IDs memory not bound to target node " << target_node;

        // For subsequent iterations, verify that new pointers are allocated.
        if (i > 0) {
            EXPECT_NE(prev_codes, partition->codes_) << "Codes pointer should change when switching NUMA nodes.";
            EXPECT_NE(prev_ids, partition->ids_) << "IDs pointer should change when switching NUMA nodes.";
        }
    }
}
#endif  // QUAKE_USE_NUMA