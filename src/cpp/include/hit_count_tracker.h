//
// Created by Jason on 3/13/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef HIT_COUNT_TRACKER_H
#define HIT_COUNT_TRACKER_H

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <common.h>

/// @brief HitCountTracker maintains per-query hit counts and scanned partition sizes in a sliding window,
/// computes the current scan fraction, and records maintenance history (split and delete events).
class HitCountTracker {
public:
    /// @brief Constructor.
    /// @param window_size Number of queries to maintain in the sliding window.
    /// @param total_vectors Total number of vectors in the index (used to compute scan fraction).
    HitCountTracker(int window_size, int total_vectors);

    /// @brief Reset the tracker.
    void reset();

    /// @brief Set the total number of vectors in the index.
    /// @param total_vectors New total.
    void set_total_vectors(int total_vectors);

    /// @brief Add per-query data.
    /// @param hit_partition_ids The IDs of partitions that were hit during the query.
    /// @param scanned_sizes The sizes of the partitions that were scanned.
    /// Both vectors must have the same length.
    void add_query_data(const vector<int64_t>& hit_partition_ids,
                      const vector<int64_t>& scanned_sizes);

    /// @brief Get the current scan fraction averaged over the sliding window.
    float get_current_scan_fraction() const;

    /// @brief Get the stored per-query hit counts.
    const vector<vector<int64_t>>& get_per_query_hits() const;

    /// @brief Get the stored per-query scanned partition sizes.
    const vector<vector<int64_t>>& get_per_query_scanned_sizes() const;

    /// @brief Record a split event in the history.
    void record_split(int64_t parent_id, int64_t parent_hits,
                     int64_t left_id, int64_t left_hits,
                     int64_t right_id, int64_t right_hits);

    /// @brief Record a delete event in the history.
    void record_delete(int64_t partition_id, int64_t hits);

    /// @brief Get the window size.
    int get_window_size() const;

    /// @brief Get the total number of queries recorded so far.
    int64_t get_num_queries_recorded() const;

private:
    int window_size_;
    int64_t total_vectors_;
    int64_t curr_query_index_;    // Points to the next slot to overwrite in the circular window.
    int64_t num_queries_recorded_; // Total queries recorded so far (up to window_size_)

    vector<vector<int64_t>> per_query_hits_;
    vector<vector<int64_t>> per_query_scanned_sizes_;

    // Running sum of the scan fractions in the current window.
    float running_sum_scan_fraction_;
    float current_scan_fraction_;

    /// @brief Helper function to compute the scan fraction for a query.
    /// @param scanned_sizes Vector of scanned partition sizes for one query.
    /// @return The fraction of total vectors scanned.
    float compute_scan_fraction(const vector<int64_t>& scanned_sizes) const;
};

#endif // HIT_COUNT_TRACKER_H
