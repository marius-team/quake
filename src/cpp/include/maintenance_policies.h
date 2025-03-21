#ifndef MAINTENANCE_POLICY_REFACTORED_H
#define MAINTENANCE_POLICY_REFACTORED_H

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>

#include "partition_manager.h"
#include "hit_count_tracker.h"
#include "maintenance_cost_estimator.h"

/**
 * @brief Maintenance policy that manages partition hit counts and
 * performs maintenance operations (such as deletion and splitting) in a single pass.
 *
 */
class MaintenancePolicy {
 public:
  /**
   * @brief Construct a new MaintenancePolicy object.
   *
   * @param partition_manager Shared pointer to the partition manager.
   * @param params Configuration parameters for the maintenance policy.
   */
  MaintenancePolicy(
      shared_ptr<PartitionManager> partition_manager,
      shared_ptr<MaintenancePolicyParams> params);

  /**
   * @brief Perform maintenance operations including deletion and splitting.
   *
   * @return MaintenanceTimingInfo with timing details.
   */
  shared_ptr<MaintenanceTimingInfo> perform_maintenance();

  /**
   * @brief Record a hit event for a given partition.
   *
   * @param partition_id Identifier of the partition.
   */
  void record_query_hits(vector<int64_t> partition_ids);

  /**
   * @brief Reset the internal maintenance state.
   */
  void reset();

 private:
  shared_ptr<PartitionManager> partition_manager_;  ///< Manages partition state.
  shared_ptr<MaintenancePolicyParams> params_;        ///< Maintenance parameters.
  shared_ptr<MaintenanceCostEstimator> cost_estimator_; ///< Cost estimator for maintenance actions.
  shared_ptr<HitCountTracker> hit_count_tracker_;       ///< Hit count tracker for partition hit rates.

  /**
   * @brief Perform local refinement on a set of partition IDs.
   *
   * @param partition_ids Tensor of partition IDs.
   */
  void local_refinement(const Tensor& partition_ids);
};

#endif  // MAINTENANCE_POLICY_REFACTORED_H