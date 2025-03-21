#include "gtest/gtest.h"
#include "maintenance_cost_estimator.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <cmath>

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
    float avg_partition_hit_rate = 0.25f;
    int k = estimator->get_k();
    int avg_partition_size = partition_size;

    // Let T = total_partitions, n = partition_size, and p = hit_rate.
    // Compute the structural benefit: the reduction in overhead when one partition is removed.
    float latency_T = latency_estimator_->estimate_scan_latency(total_partitions, k);
    float latency_T_minus_1 = latency_estimator_->estimate_scan_latency(total_partitions - 1, k);
    float delta_overhead = latency_T_minus_1 - latency_T;

    float cost_old = (total_partitions - 1) * avg_partition_hit_rate
                     * latency_estimator_->estimate_scan_latency(avg_partition_size, k)
                     + hit_rate
                     * latency_estimator_->estimate_scan_latency(partition_size, k);

    // Compute the "new" size and scan fraction after merging
    float merged_size = avg_partition_size + static_cast<float>(partition_size) / (total_partitions - 1);
    float merged_hit_rate = avg_partition_hit_rate + hit_rate / static_cast<float>(total_partitions - 1);

    float cost_new;
    if (partition_size < total_partitions) {
        // assume at most partition_size partitions get the extra vectors
        cost_new = partition_size * merged_hit_rate * latency_estimator_->estimate_scan_latency(avg_partition_size + 1, k)
                   + (total_partitions - partition_size - 1) * merged_hit_rate * latency_estimator_->estimate_scan_latency(avg_partition_size, k);
    } else {
        cost_new = (total_partitions - 1) * merged_hit_rate * latency_estimator_->estimate_scan_latency(ceil(merged_size), k);
    }

    float delta_scanning = cost_new - cost_old;
    float expected_delta = delta_overhead + delta_scanning;

    int average_partition_size = partition_size;

    float computed_delta = estimator->compute_delete_delta(partition_size,
        hit_rate,
        total_partitions,
        avg_partition_hit_rate,
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