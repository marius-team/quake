#include "gtest/gtest.h"
#include "maintenance_cost_estimator.h"
#include <memory>
#include <vector>
#include <stdexcept>

// For convenience.
using std::make_shared;
using std::shared_ptr;
using std::vector;

// Test fixture for MaintenanceCostEstimator tests.
class MaintenanceCostEstimatorTest : public ::testing::Test {
protected:
    int d = 128;         // dimension (not used in our fake, but required)
    float alpha = 0.9f;
    int k = 10;
    shared_ptr<ListScanLatencyEstimator> latency_estimator_;
    MaintenanceCostEstimator* estimator;

    virtual void SetUp() {
        estimator = new MaintenanceCostEstimator(d, alpha, k);
        latency_estimator_ = estimator->get_latency_estimator();
    }

    virtual void TearDown() {
        delete estimator;
    }
};

TEST_F(MaintenanceCostEstimatorTest, ComputeSplitDelta) {
    // Given: partition_size = 1000, hit_rate = 0.3, total_partitions = 100.
    int partition_size = 1000;
    float hit_rate = 0.3f;
    int total_partitions = 100;

    float expected_delta_overhead = latency_estimator_->estimate_scan_latency(total_partitions + 1, k) -
                           latency_estimator_->estimate_scan_latency(total_partitions, k);
    float expected_delta_split =  (latency_estimator_->estimate_scan_latency(partition_size / 2, k) * hit_rate * (2 * alpha) -
                           latency_estimator_->estimate_scan_latency(partition_size, k) * hit_rate);
    float expected_delta = expected_delta_overhead + expected_delta_split;

    float computed_delta = estimator->compute_split_delta(partition_size, hit_rate, total_partitions);

    std::cout << "Computed delta: " << computed_delta << std::endl;
    std::cout << "Expected delta: " << expected_delta << std::endl;

    EXPECT_NEAR(computed_delta, expected_delta, 1.0);
}

TEST_F(MaintenanceCostEstimatorTest, ComputeDeleteDelta) {
    // Given: partition_size = 1000, hit_rate = 0.3, total_partitions = 100, current_scan_fraction = 0.25.
    int partition_size = 1000;
    float hit_rate = 0.3f;
    int total_partitions = 100;
    float current_scan_fraction = 0.25f;
    int k = estimator->get_k();

    // Let T = total_partitions, n = partition_size, and p = hit_rate.
    // Compute the structural benefit: the reduction in overhead when one partition is removed.
    float latency_T = latency_estimator_->estimate_scan_latency(total_partitions, k);
    float latency_T_minus_1 = latency_estimator_->estimate_scan_latency(total_partitions - 1, k);
    float delta_overhead = latency_T_minus_1 - latency_T;

    // Compute the merging penalty.
    // After deletion, the n vectors of the deleted partition are redistributed among (T-1) partitions.
    // Under an even-distribution assumption, each remaining partition gets an extra n/(T-1) vectors.
    // The new cost for queries that originally hit this partition becomes:
    // L(n + n/(T-1), k) instead of L(n, k). The extra cost is:
    float merged_partition_size = partition_size + partition_size / static_cast<float>(total_partitions - 1);
    float latency_merged = latency_estimator_->estimate_scan_latency(merged_partition_size, k);
    float latency_original = latency_estimator_->estimate_scan_latency(partition_size, k);
    float delta_merge = latency_merged - latency_original;

    float delta_reassign = current_scan_fraction * latency_original;

    // Total delta cost is the sum of the structural benefit and the merging penalty scaled by the hit rate.
    float expected_delta = delta_overhead + hit_rate * delta_merge + delta_reassign;

    int average_partition_size = partition_size;

    float computed_delta = estimator->compute_delete_delta(partition_size,
        hit_rate,
        total_partitions,
        current_scan_fraction,
        average_partition_size);

    std::cout << "Computed delta: " << computed_delta << std::endl;
    std::cout << "Expected delta: " << expected_delta << std::endl;

    EXPECT_NEAR(computed_delta, expected_delta, 1.0);
}

TEST_F(MaintenanceCostEstimatorTest, InvalidParametersThrow) {
    // Test that invalid parameters cause exceptions in the constructor.
    EXPECT_THROW({
        MaintenanceCostEstimator est(d, -0.5f, k);
    }, std::invalid_argument);
    EXPECT_THROW({
        MaintenanceCostEstimator est(d, alpha, 0);
    }, std::invalid_argument);
}