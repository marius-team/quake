//
// Created by Jason on 3/13/25.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef MAINTENANCE_COST_ESTIMATOR_H
#define MAINTENANCE_COST_ESTIMATOR_H

#include <memory>
#include <string>
#include <vector>

using std::vector;
using std::shared_ptr;

/**
 * @brief Estimates the scan latency for a list based on its size and the number of elements to retrieve.
 *
 * The latency function l(n, k) is determined via linear interpolation of measured scan latencies for
 * different list sizes (n) and retrieval counts (k). For points outside the measured grid, the function
 * extrapolates using the slope between the last two points.
 */
class ListScanLatencyEstimator {
public:
    /**
     * @brief Constructor.
     *
     * Attempts to load a latency profile from disk if a file path is provided. If loading fails
     * (e.g. file not found or grid mismatch), the profile_scan_latency() method will be invoked
     * to compute and save the latency profile.
     *
     * @param d Dimension of the vectors.
     * @param n_values Vector of n-values for the grid.
     * @param k_values Vector of k-values for the grid.
     * @param n_trials Number of trials used for profiling (default: 100).
     * @param adaptive_nprobe Flag indicating whether to use adaptive nprobe (default: false).
     * @param profile_filename Optional CSV file name for loading/saving the profile (default: empty string).
     */
    ListScanLatencyEstimator(int d,
                             const std::vector<int> &n_values,
                             const std::vector<int> &k_values,
                             int n_trials = 100,
                             bool adaptive_nprobe = false,
                             const std::string &profile_filename = "");

    /**
     * @brief Profiles the scan latency and populates the latency model.
     *
     * This operation is expensive and should typically be executed only once.
     */
    void profile_scan_latency();

    /**
     * @brief Estimates the scan latency for a given list size and retrieval count.
     *
     * @param n List size.
     * @param k Number of elements to retrieve.
     * @return Estimated latency as a float.
     */
    float estimate_scan_latency(int n, int k) const;


    /**
     * @brief Sets the number of trials to use for latency estimation.
     *
     * @param n_trials New number of trials.
     */
    void set_n_trials(int n_trials) {
        n_trials_ = n_trials;
    }

    /**
     * @brief Saves the internally profiled latency model to a CSV file.
     *
     * @param filename File path for saving the latency profile.
     * @return True if saving is successful; false otherwise.
     */
    bool save_latency_profile(const std::string &filename) const;

    /**
     * @brief Loads an existing latency profile from a CSV file.
     *
     * @param filename File path to load the latency profile from.
     * @return True if loading is successful; false otherwise.
     */
    bool load_latency_profile(const std::string &filename);

    // Public members for convenience/access.
    int d_;
    std::vector<int> n_values_;
    std::vector<int> k_values_;
    std::vector<std::vector<float> > scan_latency_model_;
    int n_trials_;

private:
    /**
     * @brief Helper function for interpolation.
     *
     * This function is provided as an example of how you might implement interpolation logic.
     *
     * @param values The grid values.
     * @param target The target value.
     * @param lower (Output) Lower index.
     * @param upper (Output) Upper index.
     * @param frac (Output) Fractional interpolation value.
     * @return True if successful; false otherwise.
     */
    bool get_interpolation_info(const std::vector<int> &values,
                                int target,
                                int &lower,
                                int &upper,
                                float &frac) const;

    /**
     * @brief Performs linear extrapolation between two float values.
     *
     * @param f1 First value.
     * @param f2 Second value.
     * @param fraction Fractional distance between f1 and f2.
     * @return Extrapolated value.
     */
    inline float linear_extrapolate(float f1, float f2, float fraction) const {
        float slope = f2 - f1;
        return f2 + slope * fraction;
    }

    /// @brief CSV file name for loading/saving the latency profile.
    /// An empty string means no file I/O will be attempted.
    std::string profile_filename_;
};


/**
 * @brief Computes cost deltas for maintenance actions (e.g., splitting or deleting partitions)
 * using a latency estimation model.
 *
 * The MaintenanceCostEstimator uses a ListScanLatencyEstimator along with parameters such as alpha
 * and k to compute the difference in latency (cost) that would result from a maintenance operation.
 */
class MaintenanceCostEstimator {
public:
   /**
    * @brief Constructor.
    *
    * @param d Dimension of the vectors.
    * @param alpha Alpha parameter used to scale the cost for splitting.
    * @param k Parameter used in latency estimation.
    * @throws std::invalid_argument if k is non-positive or alpha is non-positive.
    */
    MaintenanceCostEstimator(int d, float alpha, int k);

   /**
    * @brief Computes the delta cost for splitting a partition.
    *
    * The computed delta represents the difference between the new cost after splitting
    * (assuming an even split) and the original cost, plus the structural overhead of adding one partition.
    *
    * @param partition_size Size of the partition to split.
    * @param hit_rate Hit rate (fraction) for the partition.
    * @param total_partitions Total number of partitions before the split.
    * @return The computed split delta.
    */
    float compute_split_delta(int partition_size, float hit_rate, int total_partitions) const;

   /**
    * @brief Computes the delta cost for deleting a partition.
    *
    * This function estimates the change in latency if a partition were deleted and its
    * vectors redistributed across the remaining partitions.
    *
    * @param partition_size Size of the partition to delete.
    * @param hit_rate Hit rate (fraction) for the partition.
    * @param total_partitions Total number of partitions before deletion.
    * @param avg_partition_hit_rate Average hit rate across all partitions.
    * @param avg_partition_size Average partition size.
    * @return The computed delete delta.
    */
    float compute_delete_delta(int partition_size, float hit_rate, int total_partitions, float avg_partition_hit_rate, float avg_partition_size) const;

   /**
    * @brief Computes the delete delta using reassignment information.
    *
    * This version considers the cost impact of reassigning vectors from the deleted partition
    * to other partitions.
    *
    * @param partition_size Size of the partition to delete.
    * @param hit_rate Hit rate (fraction) for the partition.
    * @param total_partitions Total number of partitions before deletion.
    * @param reassign_counts Vector containing the number of vectors reassigned to each partition.
    * @param reassign_sizes Vector containing the sizes of the partitions to which vectors are reassigned.
    * @param reassign_hit_rates Vector containing the hit rates of the partitions to which vectors are reassigned.
    * @return The computed delete delta.
    */
    float compute_delete_delta_w_reassign(int partition_size, float hit_rate, int total_partitions,  const vector<int64_t> &reassign_counts, const vector<int64_t> &reassign_sizes, const vector<float> &reassign_hit_rates) const;

   /**
    * @brief Returns the latency estimator.
    *
    * @return A shared pointer to the ListScanLatencyEstimator.
    */
    shared_ptr<ListScanLatencyEstimator> get_latency_estimator() const;


   /**
    * @brief Returns the parameter k used in latency estimation.
    *
    * @return The k value.
    */
    int get_k() const;

private:
    float alpha_;
    int k_;
    int d_;
    shared_ptr<ListScanLatencyEstimator> latency_estimator_;
};

#endif // MAINTENANCE_COST_ESTIMATOR_H
