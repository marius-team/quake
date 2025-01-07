//
// Created by Jason on 10/7/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names
//
// This file contains basic unit tests for MaintenancePolicy and
// its derived classes. Each test checks individual functionality
// within the maintenance code. If the approach for testing a
// particular method isn't obvious, the test is left blank.

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "maintenance_policies.h"
#include "partition_manager.h"
#include "list_scanning.h"
#include "quake_index.h"

using std::shared_ptr;
using std::vector;
using torch::Tensor;

// Helper function to create a QuakeIndex + PartitionManager:
static std::tuple<shared_ptr<QuakeIndex>, shared_ptr<PartitionManager>> CreateParentAndManager(
    int64_t nlist, int dimension, int64_t ntotal) {
  auto clustering = std::make_shared<Clustering>();
  clustering->partition_ids = torch::arange(nlist, torch::kInt64);

  Tensor vectors = torch::randn({ntotal, dimension}, torch::kFloat32);
  Tensor ids = torch::arange(ntotal, torch::kInt64);
  Tensor assignments = torch::randint(nlist, {ntotal}, torch::kInt64);

  Tensor centroids = torch::empty({nlist, dimension}, torch::kFloat32);
  for (int i = 0; i < nlist; i++) {
    Tensor v = vectors.index_select(0, torch::nonzero(assignments == i).squeeze(1));
    Tensor id = ids.index_select(0, torch::nonzero(assignments == i).squeeze(1));
    clustering->vectors.push_back(v);
    clustering->vector_ids.push_back(id);
    centroids[i] = v.mean(0);
  }
  clustering->centroids = centroids;

  auto parent = std::make_shared<QuakeIndex>();
  auto build_params = std::make_shared<IndexBuildParams>();
  parent->build(clustering->centroids, clustering->partition_ids, build_params);

  auto manager = std::make_shared<PartitionManager>();
  manager->init_partitions(parent, clustering);

  return {parent, manager};
}

TEST(MaintenancePolicyTest, IncrementAndCheckHitCount) {
  // Checks that increment_hit_count updates per_partition_hits_
  // and curr_query_id_ is incremented correctly.

  auto [p, pm] = CreateParentAndManager(3, 4, 100);
  auto params = std::make_shared<MaintenancePolicyParams>();
  params->window_size = 3;
  params->alpha = 0.5f;
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  EXPECT_EQ(policy->curr_query_id_, 0);
  int n = 5;
  vector<int64_t> hits = {1, 2};
  for (int i = 0; i < n; i++) {
    policy->increment_hit_count(hits);
  }
  EXPECT_EQ(policy->curr_query_id_, n);

  auto st = policy->get_partition_state(false);
  for (size_t i = 0; i < st->partition_ids.size(); i++) {
    auto pid = st->partition_ids[i];
    if (pid == 1 || pid == 2) {
      // With n queries, each incrementing partitions 1 & 2 once,
      // hit_rate => (n) / n = 1.0
      EXPECT_FLOAT_EQ(st->partition_hit_rate[i], 1.0f);
    } else {
      EXPECT_FLOAT_EQ(st->partition_hit_rate[i], 0.0f);
    }
  }
}

TEST(MaintenancePolicyTest, DecrementAndCheckHitCount) {
  // Checks that decrement_hit_count reduces the partition's per_partition_hits_.

  auto [p, pm] = CreateParentAndManager(2, 4, 50);
  auto params = std::make_shared<MaintenancePolicyParams>();
  params->window_size = 5;
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  vector<int64_t> hits = {0};
  for (int i = 0; i < 3; i++) policy->increment_hit_count(hits);

  policy->decrement_hit_count(0);
  auto st = policy->get_partition_state(false);
  for (size_t i = 0; i < st->partition_ids.size(); i++) {
    if (st->partition_ids[i] == 0) {
      // After 3 increments and 1 decrement => total hits = 2
      // The window size is effectively 3 queries,
      // so hit rate => 2/3
      EXPECT_NEAR(st->partition_hit_rate[i], (2.0f / 3.0f), 1e-6);
    }
  }
}

TEST(MaintenancePolicyTest, GetSplitHistory) {
  // Checks get_split_history after manually calling add_split.

  auto [p, pm] = CreateParentAndManager(3, 4, 30);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  // Pretend that partition 10 was split into 11,12. Provide some hits to partition 10.
  policy->add_partition(10, 5);
  policy->add_split(10, 11, 12);

  auto history = policy->get_split_history();
  // The tuple => (parent, parent_hits, left_id, left_hits, right_id, right_hits)
  // We expect one record: (10, 5, 11, 5, 12, 5)
  ASSERT_EQ(history.size(), 1u);
  auto rec = history[0];
  EXPECT_EQ(std::get<0>(rec), 10);
  EXPECT_EQ(std::get<1>(rec), 5);
  EXPECT_EQ(std::get<2>(rec), 11);
  EXPECT_EQ(std::get<3>(rec), 5);
  EXPECT_EQ(std::get<4>(rec), 12);
  EXPECT_EQ(std::get<5>(rec), 5);
}

TEST(MaintenancePolicyTest, GetPartitionState) {
  // Checks get_partition_state for only_modified = false.

  auto [p, pm] = CreateParentAndManager(4, 4, 100);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  auto st = policy->get_partition_state(false);
  // TODO: Confirm partition IDs, sizes, and hit rates if needed.
  // If it's not obvious, we leave it here.
}

TEST(MaintenancePolicyTest, SetPartitionModified) {
  // Checks that set_partition_modified updates the internal set.

  auto [p, pm] = CreateParentAndManager(2, 4, 20);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  policy->set_partition_modified(99);
  // TODO: verify internal sets if there's a direct way
}

TEST(MaintenancePolicyTest, SetPartitionUnmodified) {
  // Checks that set_partition_unmodified removes a partition from the modified set.

  auto [p, pm] = CreateParentAndManager(2, 4, 20);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  policy->set_partition_modified(42);
  policy->set_partition_unmodified(42);
  // TODO: verify internal sets if there's a direct way
}

TEST(MaintenancePolicyTest, EstimateSplitDelta) {
  // Checks estimate_split_delta for partitions in a state.
  // TODO: Provide a custom PartitionState or rely on real data.
}

TEST(MaintenancePolicyTest, EstimateDeleteDelta) {
  // Checks estimate_delete_delta for partitions in a state.
  // TODO: Provide a custom PartitionState or rely on real data.
}

TEST(MaintenancePolicyTest, EstimateAddLevelDelta) {
  // Checks estimate_add_level_delta is always returning 0.0 (as code indicates).
  auto [p, pm] = CreateParentAndManager(2, 4, 10);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);
  float val = policy->estimate_add_level_delta();
  EXPECT_FLOAT_EQ(val, 0.0f);
}

TEST(MaintenancePolicyTest, EstimateRemoveLevelDelta) {
  // Checks estimate_remove_level_delta is always returning 0.0 (as code indicates).
  auto [p, pm] = CreateParentAndManager(2, 4, 10);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);
  float val = policy->estimate_remove_level_delta();
  EXPECT_FLOAT_EQ(val, 0.0f);
}

TEST(MaintenancePolicyTest, Maintenance) {
  // Checks that maintenance runs without error and returns a timing info struct.
  auto [p, pm] = CreateParentAndManager(3, 4, 50);
  auto params = std::make_shared<MaintenancePolicyParams>();
  params->delete_threshold_ns = 10.0;
  params->split_threshold_ns = 10.0;
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);
  auto info = policy->maintenance();
  // TODO: verify info->n_deletes, info->n_splits, etc., if needed.
}

TEST(MaintenancePolicyTest, AddSplit) {
  // Checks that add_split correctly updates data structures.
  auto [p, pm] = CreateParentAndManager(2, 4, 20);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  policy->add_partition(50, 7);
  policy->add_split(50, 51, 52);
  // TODO: verify internal data
}

TEST(MaintenancePolicyTest, AddPartition) {
  // Checks that add_partition inserts new partition hits in the map.
  auto [p, pm] = CreateParentAndManager(2, 4, 20);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  policy->add_partition(1234, 99);
  // TODO: verify internal data
}

TEST(MaintenancePolicyTest, RemovePartition) {
  // Checks that remove_partition deletes from per_partition_hits_ and populates deleted_partition_hit_rate_.
  auto [p, pm] = CreateParentAndManager(2, 4, 20);
  auto params = std::make_shared<MaintenancePolicyParams>();
  auto policy = std::make_shared<QueryCostMaintenance>(pm, params);

  policy->add_partition(5, 10);
  policy->remove_partition(5);
  // TODO: verify it's in deleted_partition_hit_rate_, etc.
}

TEST(MaintenancePolicyTest, RefinePartitions) {
  // Not obvious how to test simply. Left blank.
}