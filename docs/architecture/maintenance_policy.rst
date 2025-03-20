MaintenancePolicy
================================

Overview
--------
The **MaintenancePolicy** is a central component of the Quake indexing system that
automates index upkeep by monitoring query “hit” patterns and deciding when to
perform maintenance actions such as splitting or deleting partitions. It ties together
several key components:

- **PartitionManager**: Manages the set of index partitions. The maintenance policy
  uses it to query partition sizes, delete outdated partitions, and trigger splits.
- **MaintenanceCostEstimator**: Provides latency-based estimates for how much a
  particular maintenance action (split or delete) will cost. It uses measured or
  extrapolated scan latencies as a function of partition size and search parameters.
- **HitCountTracker**: Records per-query “hit” counts (i.e. which partitions were
  scanned) over a sliding window. It computes the average scan fraction and maintains
  history of split and delete events for later analysis.

Key Methods
-------------------------
1. **Record Query Hits:**
   Each time a query is processed, the `record_query_hits()` method is called
   with the partition IDs that were “hit” during the search. The *HitCountTracker*
   accumulates these events to compute the current scan fraction across the sliding window.

2. **Perform Maintenance:**
   When the window is full, `perform_maintenance()` is invoked. This method:

   - Aggregates hit counts from the *HitCountTracker*.
   - For each partition, uses the *MaintenanceCostEstimator* to compute a “delta”
     (change in cost) for both deletion and splitting. The decision is based on whether
     the estimated cost delta exceeds configured thresholds.
   - Determines which partitions should be split (to improve query efficiency)
     or deleted (if underutilized).
   - Triggers the corresponding operations through the *PartitionManager* and then,
     if needed, calls local refinement on newly split partitions.
   - Returns timing information via a *MaintenanceTimingInfo* structure.

3. **Reset:**
   After maintenance operations complete, the policy can be reset (via `reset()`)
   to clear the hit history and start a fresh monitoring window.

Configuration and Parameters
------------------------------
Maintenance behavior is governed by a set of parameters (encapsulated in the
**MaintenancePolicyParams** structure):

- **window_size**: Number of queries over which hits are aggregated.
- **refinement_radius** and **refinement_iterations**: Control local refinement of new partitions.
- **delete_threshold_ns** and **split_threshold_ns**: Latency thresholds (in nanoseconds)
  that trigger deletion or splitting.
- **alpha**: Scaling factor applied to cost estimates.
- **enable_split_rejection / enable_delete_rejection**: Flags to allow rejecting an
  otherwise triggered action if additional checks (such as vector reassignments) suggest it
  may not be beneficial.
