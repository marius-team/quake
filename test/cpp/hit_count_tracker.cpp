#include "gtest/gtest.h"
#include "hit_count_tracker.h"

#include <random>
#include <numeric>
#include <cmath>

// Fixture for realistic HitCountTracker tests.
class HitCountTrackerTest : public ::testing::Test {
protected:
    // Parameters used in multiple tests.
    int window_size = 5;
    int total_vectors = 1000;

    // Utility: Given a vector of scanned sizes, compute fraction.
    float ComputeExpectedFraction(const std::vector<int64_t>& scanned_sizes) {
        int sum = std::accumulate(scanned_sizes.begin(), scanned_sizes.end(), 0);
        return static_cast<float>(sum) / total_vectors;
    }
};

TEST_F(HitCountTrackerTest, RandomQueriesTest) {
    // Simulate a realistic workload with random queries.
    // We'll generate 20 queries; each query has a random number of scanned partitions (between 1 and 5)
    // and random scanned sizes in [0,300]. After each query, we compare the expected average scan fraction.
    HitCountTracker tracker(window_size, total_vectors);

    // For expected simulation, store each query's fraction.
    std::vector<float> queryFractions;

    // Set up a random generator with fixed seed for reproducibility.
    std::default_random_engine rng(42);
    std::uniform_int_distribution<int> partitionsDist(1, 5);
    std::uniform_int_distribution<int> sizeDist(0, 300);

    const int numQueries = 20;
    for (int q = 0; q < numQueries; ++q) {
        int numPartitions = partitionsDist(rng);
        std::vector<int64_t> hit_ids;
        std::vector<int64_t> scanned_sizes;
        for (int i = 0; i < numPartitions; ++i) {
            // For simplicity, use partition id equal to i.
            hit_ids.push_back(i);
            scanned_sizes.push_back(sizeDist(rng));
        }
        tracker.add_query_data(hit_ids, scanned_sizes);
        float fraction = tracker.get_current_scan_fraction();
        // Manually compute the expected average over the effective window.
        queryFractions.push_back( ComputeExpectedFraction(scanned_sizes) );
        int effective_window = (q + 1 < window_size) ? q + 1 : window_size;
        float expectedAvg = 0.0f;
        for (int j = (q + 1 <= window_size ? 0 : q + 1 - window_size); j <= q; ++j) {
            expectedAvg += queryFractions[j];
        }
        expectedAvg /= effective_window;
        EXPECT_NEAR(fraction, expectedAvg, 1e-5f)
            << "Failure at query " << q << ": expected " << expectedAvg << ", got " << fraction;
    }
}

TEST_F(HitCountTrackerTest, MultipleWindowCyclesTest) {
    // Simulate many queries (e.g. 50 queries) to force multiple cycles through the sliding window.
    HitCountTracker tracker(window_size, total_vectors);
    std::vector<float> queryFractions;

    std::default_random_engine rng(123);
    std::uniform_int_distribution<int> partitionsDist(1, 4);
    std::uniform_int_distribution<int> sizeDist(0, 300);

    const int numQueries = 50;
    for (int q = 0; q < numQueries; ++q) {
        int numPartitions = partitionsDist(rng);
        std::vector<int64_t> hit_ids;
        std::vector<int64_t> scanned_sizes;
        for (int i = 0; i < numPartitions; ++i) {
            hit_ids.push_back(i);
            scanned_sizes.push_back(sizeDist(rng));
        }
        tracker.add_query_data(hit_ids, scanned_sizes);
        float fraction = tracker.get_current_scan_fraction();
        queryFractions.push_back( std::accumulate(scanned_sizes.begin(), scanned_sizes.end(), 0) / static_cast<float>(total_vectors) );

        int effective_window = (q + 1 < window_size) ? q + 1 : window_size;
        float expectedAvg = 0.0f;
        for (int j = (q + 1 <= window_size ? 0 : q + 1 - window_size); j <= q; ++j) {
            expectedAvg += queryFractions[j];
        }
        expectedAvg /= effective_window;
        EXPECT_NEAR(fraction, expectedAvg, 1e-5f)
            << "At query " << q << ", expected average " << expectedAvg << ", got " << fraction;
    }
}

TEST_F(HitCountTrackerTest, EdgeCaseZeroScannedTest) {
    // Test a query where all scanned sizes are zero.
    HitCountTracker tracker(window_size, total_vectors);
    std::vector<int64_t> hit_ids = {1, 2, 3};
    std::vector<int64_t> scanned_sizes = {0, 0, 0}; // Expect fraction 0.
    tracker.add_query_data(hit_ids, scanned_sizes);
    EXPECT_NEAR(tracker.get_current_scan_fraction(), 0.0f, 1e-5f);
}

TEST_F(HitCountTrackerTest, FullScanTest) {
    // Test a query where the scanned sizes sum equals total_vectors.
    HitCountTracker tracker(window_size, total_vectors);
    std::vector<int64_t> hit_ids = {0};
    std::vector<int64_t> scanned_sizes = {total_vectors}; // Fraction should be 1.0.
    tracker.add_query_data(hit_ids, scanned_sizes);
    EXPECT_NEAR(tracker.get_current_scan_fraction(), 1.0f, 1e-5f);
}

TEST_F(HitCountTrackerTest, HistoryAccumulationTest) {
    // Test that history accumulates correctly over multiple events.
    HitCountTracker tracker(window_size, total_vectors);
    // Record several split events.
    tracker.record_split(10, 100, 11, 60, 12, 40);
    tracker.record_split(20, 200, 21, 120, 22, 80);
    // Record several delete events.
    tracker.record_delete(30, 90);
    tracker.record_delete(40, 110);

    const vector<SplitRecord>& splits = tracker.get_split_history();
    const vector<DeleteRecord>& deletes = tracker.get_delete_history();

    EXPECT_EQ(splits.size(), 2);
    EXPECT_EQ(deletes.size(), 2);

    // Check first split record.
    EXPECT_EQ(splits[0].parent_id, 10);
    EXPECT_EQ(splits[0].parent_hits, 100);
    EXPECT_EQ(splits[0].left_id, 11);
    EXPECT_EQ(splits[0].left_hits, 60);
    EXPECT_EQ(splits[0].right_id, 12);
    EXPECT_EQ(splits[0].right_hits, 40);

    // Check second split record.
    EXPECT_EQ(splits[1].parent_id, 20);
    EXPECT_EQ(splits[1].parent_hits, 200);
    EXPECT_EQ(splits[1].left_id, 21);
    EXPECT_EQ(splits[1].left_hits, 120);
    EXPECT_EQ(splits[1].right_id, 22);
    EXPECT_EQ(splits[1].right_hits, 80);

    // Check delete records.
    EXPECT_EQ(deletes[0].partition_id, 30);
    EXPECT_EQ(deletes[0].hits, 90);
    EXPECT_EQ(deletes[1].partition_id, 40);
    EXPECT_EQ(deletes[1].hits, 110);
}

TEST_F(HitCountTrackerTest, InterleavedQueriesAndHistoryTest) {
    // Interleave queries with history events and ensure that history is independent of window state.
    HitCountTracker tracker(window_size, total_vectors);
    // Add some queries.
    vector<int64_t> hit_ids1 = {1, 2};
    vector<int64_t> scanned_sizes1 = {100, 50};   // 0.15
    tracker.add_query_data(hit_ids1, scanned_sizes1);

    vector<int64_t> hit_ids2 = {3};
    vector<int64_t> scanned_sizes2 = {300};         // 0.3
    tracker.add_query_data(hit_ids2, scanned_sizes2);

    // Record a history event.
    tracker.record_delete(50, 75);
    // Add more queries.
    vector<int64_t> hit_ids3 = {4, 5};
    vector<int64_t> scanned_sizes3 = {150, 150};      // 0.3
    tracker.add_query_data(hit_ids3, scanned_sizes3);

    // Record a split event.
    tracker.record_split(60, 120, 61, 70, 62, 50);

    // Check current scan fraction is computed from the last 3 queries.
    float expected1 = (0.15f + 0.3f + 0.3f) / 3.0f;
    EXPECT_NEAR(tracker.get_current_scan_fraction(), expected1, 1e-5f);

    // Verify history remains.
    const vector<DeleteRecord>& deletes = tracker.get_delete_history();
    const vector<SplitRecord>& splits = tracker.get_split_history();
    EXPECT_EQ(deletes.size(), 1);
    EXPECT_EQ(deletes[0].partition_id, 50);
    EXPECT_EQ(deletes[0].hits, 75);
    EXPECT_EQ(splits.size(), 1);
    EXPECT_EQ(splits[0].parent_id, 60);
    EXPECT_EQ(splits[0].parent_hits, 120);
}