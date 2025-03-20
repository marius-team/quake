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


/**
 * @brief HitCountTracker maintains per-query hit counts and scanned partition sizes in a sliding window.
 **/
class HitCountTracker {
public:
    /**
     * @brief Constructs a HitCountTracker.
     *
     * @param window_size Number of queries to maintain in the sliding window.
     * @param total_vectors Total number of vectors in the index (used for computing the scan fraction).
     */
    HitCountTracker(int window_size, int total_vectors);

    /**
     * @brief Resets the tracker by clearing all recorded query data.
     */
    void reset();

    /**
     * @brief Sets the total number of vectors in the index.
     *
     * @param total_vectors The new total vector count.
     */
    void set_total_vectors(int total_vectors);

   /**
    * @brief Adds per-query data.
    *
    * Records the partition IDs that were hit and their corresponding scanned sizes for a query.
    * Both vectors must have the same length.
    *
    * @param hit_partition_ids Vector of partition IDs hit during the query.
    * @param scanned_sizes Vector of scanned sizes corresponding to each partition hit.
    */
    void add_query_data(const vector<int64_t>& hit_partition_ids, const vector<int64_t>& scanned_sizes);

   /**
    * @brief Retrieves the current scan fraction averaged over the sliding window.
    *
    * @return The current scan fraction.
    */
    float get_current_scan_fraction() const;

   /**
    * @brief Retrieves the stored per-query hit counts.
    *
    * @return A constant reference to the vector containing per-query hit counts.
    */
    const vector<vector<int64_t>>& get_per_query_hits() const;

   /**
    * @brief Retrieves the stored per-query scanned partition sizes.
    *
    * @return A constant reference to the vector containing per-query scanned sizes.
    */
    const vector<vector<int64_t>>& get_per_query_scanned_sizes() const;

   /**
    * @brief Returns the sliding window size.
    *
    * @return The window size.
    */
    int get_window_size() const;


   /**
    * @brief Returns the total number of queries recorded so far.
    *
    * @return The number of queries recorded.
    */
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

   /**
    * @brief Computes the scan fraction for a query.
    *
    * The scan fraction is calculated as the sum of the scanned sizes divided by the total number of vectors.
    *
    * @param scanned_sizes Vector of scanned partition sizes for one query.
    * @return The computed scan fraction.
    */
    float compute_scan_fraction(const vector<int64_t>& scanned_sizes) const;
};

#endif // HIT_COUNT_TRACKER_H
