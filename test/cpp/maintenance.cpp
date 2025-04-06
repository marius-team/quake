// maintenance_policy_refactored_test.cpp
//
// Unit tests for the refactored MaintenancePolicy class.
//
// These tests use a helper function to create a parent QuakeIndex and PartitionManager,
// and then exercise the new MaintenancePolicy interface (recording hits, resetting,
// and performing maintenance operations based on cost-estimation).

#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include <vector>

#include "maintenance_policies.h"
#include "partition_manager.h"
#include "quake_index.h"
#include "list_scanning.h"  // Needed for latency estimator, etc.

using std::make_shared;
using std::shared_ptr;
using std::tuple;
using std::vector;
using torch::Tensor;

// Helper function to create a parent QuakeIndex and PartitionManager.
static tuple<shared_ptr<QuakeIndex>, shared_ptr<PartitionManager>> CreateParentAndManager(
    int64_t nlist, int dimension, int64_t ntotal) {
  auto clustering = make_shared<Clustering>();
  clustering->partition_ids = torch::arange(nlist, torch::kInt64);

  Tensor vectors = torch::randn({ntotal, dimension}, torch::kFloat32);
  Tensor ids = torch::arange(ntotal, torch::kInt64);
  Tensor assignments = torch::randint(nlist, {ntotal}, torch::kInt64);

  Tensor centroids = torch::empty({nlist, dimension}, torch::kFloat32);
  for (int i = 0; i < nlist; i++) {
    Tensor idx = torch::nonzero(assignments == i).squeeze(1);
    Tensor v = vectors.index_select(0, idx);
    Tensor id = ids.index_select(0, idx);
    clustering->vectors.push_back(v);
    clustering->vector_ids.push_back(id);
    centroids[i] = v.mean(0);
  }
  clustering->centroids = centroids;

  auto parent = make_shared<QuakeIndex>();
  auto build_params = make_shared<IndexBuildParams>();
  parent->build(clustering->centroids, clustering->partition_ids, build_params);

  auto manager = make_shared<PartitionManager>();
  manager->init_partitions(parent, clustering);

  return {parent, manager};
}

//
// Test that without any hit events, perform_maintenance() does nothing (i.e. no deletion or splitting).
//
TEST(MaintenancePolicyRefactoredTest, NoMaintenanceWithoutHits) {
  auto [parent, manager] = CreateParentAndManager(3, 4, 100);
  auto params = make_shared<MaintenancePolicyParams>();
  params->window_size = 3;
  params->alpha = 0.5f;
  // Set thresholds to 0 so that any deviation would trigger maintenance if there were hits.
  params->delete_threshold_ns = 0.0f;
  params->split_threshold_ns = 0.0f;
  // Assume params->latency_estimator is already set up (or set it to a dummy if needed).

  auto policy = make_shared<MaintenancePolicy>(manager, params);
  shared_ptr<MaintenanceTimingInfo> info = policy->perform_maintenance();

  // With no hit events recorded, no deletion or splitting should occur.
  EXPECT_EQ(info->delete_time_us, 0);
  EXPECT_EQ(info->split_time_us, 0);
}

//
// Test that hit events are recorded and then reset properly.
//
TEST(MaintenancePolicyRefactoredTest, RecordAndResetHitCount) {
  auto [parent, manager] = CreateParentAndManager(3, 4, 100);
  auto params = make_shared<MaintenancePolicyParams>();
  params->window_size = 3;
  params->alpha = 0.5f;
  // Use non-triggering thresholds.
  params->delete_threshold_ns = 1000.0f;
  params->split_threshold_ns = 1000.0f;

  auto policy = make_shared<MaintenancePolicy>(manager, params);

  // Record several hit events.
  policy->record_query_hits({1});
  policy->record_query_hits({2});
  policy->record_query_hits({1});

  // Perform maintenance. Since thresholds are high, no maintenance should occur.
  shared_ptr<MaintenanceTimingInfo> info1 = policy->perform_maintenance();
  EXPECT_EQ(info1->delete_time_us, 0);
  EXPECT_EQ(info1->split_time_us, 0);

  // Now reset and verify that subsequent maintenance has no effect.
  policy->reset();
  shared_ptr<MaintenanceTimingInfo> info2 = policy->perform_maintenance();
  EXPECT_EQ(info2->delete_time_us, 0);
  EXPECT_EQ(info2->split_time_us, 0);
}

//
// Test that underutilized partitions are selected for deletion.
// Here we record hits only on some partitions so that others remain underutilized.
//
TEST(MaintenancePolicyRefactoredTest, TriggerDeletion) {
  auto [parent, manager] = CreateParentAndManager(1000, 4, 100000);
  auto params = make_shared<MaintenancePolicyParams>();
  params->window_size = 999;
  params->alpha = 0.5f;
  // Set delete threshold low so that partitions with few hits are marked.
  params->delete_threshold_ns = 0.0f;
  // Set split threshold high so that splitting does not trigger.
  params->split_threshold_ns = 1000.0f;

  auto policy = make_shared<MaintenancePolicy>(manager, params);

  // set hits for all partitions besides 0
  for (int i = 1; i < 1000; i++) {
    policy->record_query_hits({i, i, i, i, i, i, i, i, i, i, i});
  }

  // make partition 0 small
  manager->partition_store_->partitions_[0]->resize(10);

  // Run maintenance. Partition 0, being unhit, should be deleted.
  shared_ptr<MaintenanceTimingInfo> info = policy->perform_maintenance();

  // Retrieve current partition IDs from the manager.
  Tensor pids = manager->get_partition_ids();
  auto pids_accessor = pids.accessor<int64_t, 1>();
  bool found0 = false;
  for (int i = 0; i < pids.size(0); i++) {
    if (pids_accessor[i] == 0) {
      found0 = true;
      break;
    }
  }
  EXPECT_FALSE(found0);
}

//
// Test that overutilized partitions are selected for splitting.
// Here we record many hits on a partition so that it is overutilized and triggers splitting.
//
TEST(MaintenancePolicyRefactoredTest, TriggerSplitting) {
  auto [parent, manager] = CreateParentAndManager(3, 4, 100);
  auto params = make_shared<MaintenancePolicyParams>();
  params->window_size = 3;
  params->alpha = 0.5f;
  // Set split threshold low to force splitting when hit count is high.
  params->split_threshold_ns = 0.0f;
  // Set delete threshold high so deletion will not be triggered.
  params->delete_threshold_ns = 1000.0f;
  // Ensure partitions are large enough to split.
  params->min_partition_size = 1;

  auto policy = make_shared<MaintenancePolicy>(manager, params);

  // Record multiple hits on partition 1.
  for (int i = 0; i < 5; i++) {
    policy->record_query_hits({1});
  }

  shared_ptr<MaintenanceTimingInfo> info = policy->perform_maintenance();

  // Check that partition 1 has been replaced by new partition IDs.
  Tensor pids = manager->get_partition_ids();
  auto pids_accessor = pids.accessor<int64_t, 1>();
  bool found1 = false;
  for (int i = 0; i < pids.size(0); i++) {
    if (pids_accessor[i] == 1) {
      found1 = true;
      break;
    }
  }
  EXPECT_FALSE(found1);
}

//
// Optionally, if you have implemented local refinement in PartitionManager,
// you can add a test that simulates a split and then verifies that refine_partitions()
// is called (perhaps by using a mock or a subclass of PartitionManager that records calls).
// For simplicity, this test only checks that perform_maintenance() runs without error
// when splits occur.
//
TEST(MaintenancePolicyRefactoredTest, MaintenanceRunsSuccessfully) {
  auto [parent, manager] = CreateParentAndManager(100, 4, 100000);
  auto params = make_shared<MaintenancePolicyParams>();
  params->window_size = 3;
  params->alpha = 0.5f;
  // Set thresholds to force both deletion and splitting if conditions are met.
  params->delete_threshold_ns = 0.0f;
  params->split_threshold_ns = 0.0f;
  params->min_partition_size = 1;
  params->refinement_radius = 10;
  params->refinement_iterations = 3;

  auto policy = make_shared<MaintenancePolicy>(manager, params);

  // Record some hit events to trigger maintenance operations.
  policy->record_query_hits({0});
  policy->record_query_hits({1});
  policy->record_query_hits({0});
  policy->record_query_hits({1});
  policy->record_query_hits({2});

  // Run maintenance and verify that timing info is returned.
  shared_ptr<MaintenanceTimingInfo> info = policy->perform_maintenance();
  // We expect non-negative timing values.
  EXPECT_GE(info->delete_time_us, 0);
  EXPECT_GE(info->split_time_us, 0);
  EXPECT_GE(info->total_time_us, 0);
}