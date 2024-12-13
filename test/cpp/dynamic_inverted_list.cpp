// dynamic_inverted_list_test.cpp

#include <gtest/gtest.h>
#include "dynamic_inverted_list.h"
#include <vector>
#include <cstring> // for memcpy
#include <iostream>

using namespace faiss;

class DynamicInvertedListTest : public ::testing::Test {
protected:
    size_t nlist = 10;
    size_t code_size = 16; // bytes per code
    DynamicInvertedLists* invlists;

    virtual void SetUp() {
        // Initialize the DynamicInvertedLists
        invlists = new DynamicInvertedLists(nlist, code_size);
    }

    virtual void TearDown() {
        delete invlists;
    }

    // Helper function to generate random codes
    void generate_random_codes(size_t n, std::vector<uint8_t>& codes) {
        codes.resize(n * code_size);
        for (size_t i = 0; i < n * code_size; ++i) {
            codes[i] = rand() % 256;
        }
    }

    // Helper function to generate sequential IDs
    void generate_sequential_ids(size_t n, std::vector<idx_t>& ids, idx_t start_id = 0) {
        ids.resize(n);
        for (size_t i = 0; i < n; ++i) {
            ids[i] = start_id + i;
        }
    }
};

// Test constructor and basic properties
TEST_F(DynamicInvertedListTest, ConstructorTest) {
    EXPECT_EQ(invlists->nlist, nlist);
    EXPECT_EQ(invlists->code_size, code_size);
    // Check that codes_ and ids_ are initialized
    EXPECT_EQ(invlists->codes_.size(), nlist);
    EXPECT_EQ(invlists->ids_.size(), nlist);
}

// Test adding entries to a list
TEST_F(DynamicInvertedListTest, AddEntriesTest) {
    size_t list_no = 5;
    size_t n_entries = 10;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    // Add entries
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Check list size
    EXPECT_EQ(invlists->list_size(list_no), n_entries);

    // Check that get_codes and get_ids return the correct data
    const uint8_t* stored_codes = invlists->get_codes(list_no);
    const idx_t* stored_ids = invlists->get_ids(list_no);

    // Compare codes
    EXPECT_EQ(std::memcmp(stored_codes, codes.data(), n_entries * code_size), 0);

    // Compare ids
    for (size_t i = 0; i < n_entries; ++i) {
        EXPECT_EQ(stored_ids[i], ids[i]);
    }
}

// Test removing an entry
TEST_F(DynamicInvertedListTest, RemoveEntryTest) {
    size_t list_no = 3;
    size_t n_entries = 5;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    // Add entries
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Remove an entry
    idx_t id_to_remove = ids[2];
    invlists->remove_entry(list_no, id_to_remove);

    // Check that list size is reduced
    EXPECT_EQ(invlists->list_size(list_no), n_entries - 1);

    // Check that the id is no longer in the list
    const idx_t* stored_ids = invlists->get_ids(list_no);
    for (size_t i = 0; i < n_entries - 1; ++i) {
        EXPECT_NE(stored_ids[i], id_to_remove);
    }
}

// Test updating entries
TEST_F(DynamicInvertedListTest, UpdateEntriesTest) {
    size_t list_no = 2;
    size_t n_entries = 5;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    // Add entries
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Prepare new codes and ids for update
    std::vector<uint8_t> new_codes;
    std::vector<idx_t> new_ids;
    generate_random_codes(n_entries, new_codes);
    generate_sequential_ids(n_entries, new_ids, 100); // start from 100

    // Update entries
    invlists->update_entries(list_no, 0, n_entries, new_ids.data(), new_codes.data());

    // Check that data has been updated
    const uint8_t* stored_codes = invlists->get_codes(list_no);
    const idx_t* stored_ids = invlists->get_ids(list_no);

    EXPECT_EQ(std::memcmp(stored_codes, new_codes.data(), n_entries * code_size), 0);

    for (size_t i = 0; i < n_entries; ++i) {
        EXPECT_EQ(stored_ids[i], new_ids[i]);
    }
}

// Test removing a list
TEST_F(DynamicInvertedListTest, RemoveListTest) {
    size_t list_no = 4;
    // Ensure list exists by adding an entry
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(1, codes);
    generate_sequential_ids(1, ids);
    invlists->add_entries(list_no, 1, ids.data(), codes.data());

    // Remove the list
    invlists->remove_list(list_no);

    // Try accessing the list, should throw exception
    EXPECT_THROW(invlists->list_size(list_no), std::runtime_error);
    EXPECT_THROW(invlists->get_codes(list_no), std::runtime_error);
    EXPECT_THROW(invlists->get_ids(list_no), std::runtime_error);
}

// Test adding a list
TEST_F(DynamicInvertedListTest, AddListTest) {
    size_t new_list_no = 20; // beyond initial nlist
    invlists->add_list(new_list_no);

    // Now, add entries to the new list
    size_t n_entries = 3;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    invlists->add_entries(new_list_no, n_entries, ids.data(), codes.data());

    // Check that list_size returns correct value
    EXPECT_EQ(invlists->list_size(new_list_no), n_entries);
}

// Test reset
TEST_F(DynamicInvertedListTest, ResetTest) {
    // Add entries to multiple lists
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        size_t n_entries = list_no + 1;
        std::vector<uint8_t> codes;
        std::vector<idx_t> ids;
        generate_random_codes(n_entries, codes);
        generate_sequential_ids(n_entries, ids);
        invlists->add_entries(list_no, n_entries, ids.data(), codes.data());
    }

    // Reset the inverted lists
    invlists->reset();

    // Check that all lists are empty or do not exist
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        // Since reset clears the maps, accessing any list should throw exception
        EXPECT_THROW(invlists->list_size(list_no), std::runtime_error);
        EXPECT_THROW(invlists->get_codes(list_no), std::runtime_error);
        EXPECT_THROW(invlists->get_ids(list_no), std::runtime_error);
    }
}

// Test get_new_list_id
TEST_F(DynamicInvertedListTest, GetNewListIdTest) {
    size_t first_id = invlists->get_new_list_id();
    size_t second_id = invlists->get_new_list_id();
    EXPECT_EQ(second_id, first_id + 1);
}

// Test exception handling
TEST_F(DynamicInvertedListTest, ExceptionHandlingTest) {
    size_t invalid_list_no = 999;

    // Accessing non-existent list should throw exception
    EXPECT_THROW(invlists->list_size(invalid_list_no), std::runtime_error);
    EXPECT_THROW(invlists->get_codes(invalid_list_no), std::runtime_error);
    EXPECT_THROW(invlists->get_ids(invalid_list_no), std::runtime_error);

    // Removing an entry from non-existent list
    EXPECT_THROW(invlists->remove_entry(invalid_list_no, 0), std::runtime_error);

    // Adding entries to non-existent list
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(1, codes);
    generate_sequential_ids(1, ids);
    EXPECT_THROW(invlists->add_entries(invalid_list_no, 1, ids.data(), codes.data()), std::runtime_error);
}

// Test converting to and from ArrayInvertedLists
TEST_F(DynamicInvertedListTest, ConvertToFromArrayInvertedListsTest) {
    // Add entries to multiple lists
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        size_t n_entries = list_no + 1;
        std::vector<uint8_t> codes;
        std::vector<idx_t> ids;
        generate_random_codes(n_entries, codes);
        generate_sequential_ids(n_entries, ids);
        invlists->add_entries(list_no, n_entries, ids.data(), codes.data());
    }

    // Convert to ArrayInvertedLists
    std::unordered_map<size_t, size_t> old_to_new_ids; 
    ArrayInvertedLists* array_invlists = convert_to_array_invlists(invlists, old_to_new_ids);

    // Verify that array_invlists has the same data
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        size_t array_list_no = old_to_new_ids[list_no];
        size_t n_entries = invlists->list_size(list_no);
        ASSERT_EQ(array_invlists->list_size(array_list_no), n_entries);

        const uint8_t* inv_codes = invlists->get_codes(list_no);
        const idx_t* inv_ids = invlists->get_ids(list_no);

        const uint8_t* arr_codes = array_invlists->get_codes(array_list_no);
        const idx_t* arr_ids = array_invlists->get_ids(array_list_no);

        ASSERT_EQ(std::memcmp(inv_codes, arr_codes, n_entries * code_size), 0);
        for (size_t i = 0; i < n_entries; ++i) {
            ASSERT_EQ(inv_ids[i], arr_ids[i]);
        }
    }

    // Create the reverse map
    std::unordered_map<size_t, size_t> new_to_old_ids; 
    for(auto& pair : old_to_new_ids) {
        new_to_old_ids[pair.second] = pair.first;
    }

    // Now convert back to DynamicInvertedLists
    DynamicInvertedLists* new_invlists = convert_from_array_invlists(array_invlists);

    // Verify that new_invlists has the same data
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        size_t old_list_no = new_to_old_ids[list_no];
        size_t n_entries = invlists->list_size(list_no);
        ASSERT_EQ(new_invlists->list_size(old_list_no), n_entries);

        const uint8_t* orig_codes = invlists->get_codes(list_no);
        const idx_t* orig_ids = invlists->get_ids(list_no);

        const uint8_t* new_codes = new_invlists->get_codes(old_list_no);
        const idx_t* new_ids = new_invlists->get_ids(old_list_no);

        ASSERT_EQ(std::memcmp(orig_codes, new_codes, n_entries * code_size), 0);
        for (size_t i = 0; i < n_entries; ++i) {
            ASSERT_EQ(orig_ids[i], new_ids[i]);
        }
    }

    // Clean up
    delete array_invlists;
    delete new_invlists;
}

// Test removing and adding a list
TEST_F(DynamicInvertedListTest, RemoveAndAddListTest) {
    size_t list_no = 5;
    size_t n_entries = 10;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    // Add entries to list
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Remove the list
    invlists->remove_list(list_no);

    // Add a new list with the same list_no
    invlists->add_list(list_no);

    // Verify that the new list is empty
    EXPECT_EQ(invlists->list_size(list_no), 0);

    // Add entries to the new list
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Verify that the entries are added
    EXPECT_EQ(invlists->list_size(list_no), n_entries);
}

// Test adding entries to a new list after get_new_list_id
TEST_F(DynamicInvertedListTest, AddEntriesToNewListTest) {
    size_t new_list_no = invlists->get_new_list_id();

    // Before adding, the list should not exist
    EXPECT_THROW(invlists->list_size(new_list_no), std::runtime_error);

    // Add the list explicitly
    invlists->add_list(new_list_no);

    // Now add entries to it
    size_t n_entries = 5;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    invlists->add_entries(new_list_no, n_entries, ids.data(), codes.data());

    // Verify
    EXPECT_EQ(invlists->list_size(new_list_no), n_entries);
}

// Test removing multiple entries
TEST_F(DynamicInvertedListTest, RemoveMultipleEntriesTest) {
    size_t list_no = 7;
    size_t n_entries = 5;
    std::vector<uint8_t> codes;
    std::vector<idx_t> ids;
    generate_random_codes(n_entries, codes);
    generate_sequential_ids(n_entries, ids);

    // Add entries
    invlists->add_entries(list_no, n_entries, ids.data(), codes.data());

    // Remove multiple entries
    invlists->remove_entry(list_no, ids[0]);
    invlists->remove_entry(list_no, ids[3]);

    // Expected size is n_entries - 2
    EXPECT_EQ(invlists->list_size(list_no), n_entries - 2);

    // Check that the removed ids are no longer there
    const idx_t* stored_ids = invlists->get_ids(list_no);
    for (size_t i = 0; i < n_entries - 2; ++i) {
        EXPECT_NE(stored_ids[i], ids[0]);
        EXPECT_NE(stored_ids[i], ids[3]);
    }
}