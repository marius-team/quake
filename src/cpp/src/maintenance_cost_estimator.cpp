#include "maintenance_cost_estimator.h"
#include <list_scanning.h>
#include <stdexcept>
#include <iostream>
#include <fstream>

// A simple helper to split a string by delimiter.
// You can replace this with any library function if you wish.
static std::vector<std::string> split_string(const std::string &str,
                                             char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

ListScanLatencyEstimator::ListScanLatencyEstimator(
    int d,
    const std::vector<int> &n_values,
    const std::vector<int> &k_values,
    int n_trials,
    bool /*adaptive_nprobe*/,
    const std::string &profile_filename)
    : d_(d),
      n_values_(n_values),
      k_values_(k_values),
      n_trials_(n_trials),
      profile_filename_(profile_filename) {
    // Initialize the latency model grid
    scan_latency_model_ = std::vector<std::vector<float> >(
        n_values_.size(), std::vector<float>(k_values_.size(), 0.0f));

    // Ensure n_values and k_values are sorted
    if (!std::is_sorted(n_values_.begin(), n_values_.end())) {
        throw std::runtime_error("n_values must be sorted in ascending order.");
    }
    if (!std::is_sorted(k_values_.begin(), k_values_.end())) {
        throw std::runtime_error("k_values must be sorted in ascending order.");
    }

    // Attempt to load from file
    bool loaded = false;
    if (!profile_filename_.empty()) {
        loaded = load_latency_profile(profile_filename_);
    }

    // If not loaded successfully, then do a fresh profile and save
    if (!loaded) {
        profile_scan_latency();
        if (!profile_filename_.empty()) {
            save_latency_profile(profile_filename_);
        }
    }
}

void ListScanLatencyEstimator::profile_scan_latency() {
    // Generate random vectors of size (max_n, d_)
    int max_n = n_values_.back();
    torch::Tensor vectors = torch::rand({max_n, d_});
    torch::Tensor ids = torch::randperm(max_n);
    torch::Tensor query = torch::rand({d_});

    for (size_t i = 0; i < n_values_.size(); ++i) {
        for (size_t j = 0; j < k_values_.size(); ++j) {
            int n = n_values_[i];
            int k = k_values_[j];

            torch::Tensor curr_vectors = vectors.narrow(0, 0, n);
            torch::Tensor curr_ids = ids.narrow(0, 0, n);
            auto topk_buffer = make_shared<TopkBuffer>(k, false);

            const float *query_ptr = query.data_ptr<float>();
            const float *curr_vectors_ptr = curr_vectors.data_ptr<float>();
            const int64_t *curr_ids_ptr = curr_ids.data_ptr<int64_t>();

            uint64_t total_latency_ns = 0;
            for (int m = 0; m < n_trials_; ++m) {
                auto start = std::chrono::high_resolution_clock::now();
                scan_list(query_ptr, curr_vectors_ptr, curr_ids_ptr, n, d_,
                          *topk_buffer);
                auto end = std::chrono::high_resolution_clock::now();

                auto duration =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                total_latency_ns += duration.count();
            }
            double mean_latency_ns = static_cast<double>(total_latency_ns) / n_trials_;
            scan_latency_model_[i][j] = static_cast<float>(mean_latency_ns);
        }
    }
}

bool ListScanLatencyEstimator::get_interpolation_info(
    const std::vector<int> &values, int target, int &lower, int &upper,
    float &frac) const {
    if (target < values.front() || target > values.back()) {
        return false; // Out of bounds
    }

    // Find the first element > target
    auto it = std::upper_bound(values.begin(), values.end(), target);
    if (it == values.end()) {
        // target is exactly the last or beyond
        lower = values.size() - 2;
        upper = values.size() - 1;
    } else {
        upper = static_cast<int>(std::distance(values.begin(), it));
        lower = upper - 1;
    }

    int lower_val = values[lower];
    int upper_val = values[upper];
    if (upper_val == lower_val) {
        frac = 0.0f;
    } else {
        frac = static_cast<float>(target - lower_val) /
               static_cast<float>(upper_val - lower_val);
    }

    return true;
}

float ListScanLatencyEstimator::estimate_scan_latency(int n, int k) const {
    if (n == 0 || k == 0) {
        return 0.0f;
    }
    // Check if n or k is below the minimum values
    if (n < n_values_.front() || k < k_values_.front()) {
        std::cout << "n=" << n << ", k=" << k << std::endl;
        throw std::out_of_range("n or k is below the minimum supported values.");
    }

    // Determine if n and k are within the grid
    bool n_within = (n <= n_values_.back());
    bool k_within = (k <= k_values_.back());

    // Interpolation indices and fractions for n
    size_t i_lower, i_upper;
    float t; // fraction for n
    if (n_within) {
        auto it = std::upper_bound(n_values_.begin(), n_values_.end(), n);
        if (it == n_values_.end()) {
            // n is exactly the last value
            i_lower = n_values_.size() - 2;
            i_upper = n_values_.size() - 1;
            t = 1.0f;
        } else {
            i_upper = static_cast<size_t>(std::distance(n_values_.begin(), it));
            i_lower = i_upper - 1;
            int n1 = n_values_[i_lower];
            int n2 = n_values_[i_upper];
            t = static_cast<float>(n - n1) / static_cast<float>(n2 - n1);
        }
    } else {
        // Extrapolate in n using the last two n_values_
        i_lower = n_values_.size() - 2;
        i_upper = n_values_.size() - 1;
        int n1 = n_values_[i_lower];
        int n2 = n_values_[i_upper];
        t = static_cast<float>(n - n2) / static_cast<float>(n2 - n1); // t > 1
    }

    // Interpolation indices and fractions for k
    size_t j_lower, j_upper;
    float u; // fraction for k
    if (k_within) {
        auto it = std::upper_bound(k_values_.begin(), k_values_.end(), k);
        if (it == k_values_.end()) {
            // k is exactly the last value
            j_lower = k_values_.size() - 2;
            j_upper = k_values_.size() - 1;
            u = 1.0f;
        } else {
            j_upper = static_cast<size_t>(std::distance(k_values_.begin(), it));
            j_lower = j_upper - 1;
            int k1 = k_values_[j_lower];
            int k2 = k_values_[j_upper];
            u = static_cast<float>(k - k1) / static_cast<float>(k2 - k1);
        }
    } else {
        // Extrapolate in k using the last two k_values_
        j_lower = k_values_.size() - 2;
        j_upper = k_values_.size() - 1;
        int k1 = k_values_[j_lower];
        int k2 = k_values_[j_upper];
        u = static_cast<float>(k - k2) / static_cast<float>(k2 - k1); // u > 1
    }

    // Both n and k within the grid => bilinear interpolation
    if (n_within && k_within) {
        float f11 = scan_latency_model_[i_lower][j_lower];
        float f12 = scan_latency_model_[i_lower][j_upper];
        float f21 = scan_latency_model_[i_upper][j_lower];
        float f22 = scan_latency_model_[i_upper][j_upper];

        // Bilinear interpolation
        float interpolated_latency =
                (1 - t) * (1 - u) * f11 + t * (1 - u) * f21 +
                (1 - t) * u * f12 + t * u * f22;
        return interpolated_latency;
    }

    // Extrapolate in n, k within the grid
    if (!n_within && k_within) {
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_upper][j_lower];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, t);

        float f3 = scan_latency_model_[i_lower][j_upper];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, t);

        float interpolated_latency =
                (1 - u) * extrapolated_f_lower + u * extrapolated_f_upper;
        return interpolated_latency;
    }

    // Extrapolate in k, n within the grid
    if (n_within && !k_within) {
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_lower][j_upper];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, u);

        float f3 = scan_latency_model_[i_upper][j_lower];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, u);

        float interpolated_latency =
                (1 - t) * extrapolated_f_lower + t * extrapolated_f_upper;
        return interpolated_latency;
    }

    // Extrapolate in both n and k
    if (!n_within && !k_within) {
        float f1 = scan_latency_model_[i_lower][j_lower];
        float f2 = scan_latency_model_[i_upper][j_lower];
        float extrapolated_f_lower = linear_extrapolate(f1, f2, t);

        float f3 = scan_latency_model_[i_lower][j_upper];
        float f4 = scan_latency_model_[i_upper][j_upper];
        float extrapolated_f_upper = linear_extrapolate(f3, f4, t);

        float extrapolated_latency =
                linear_extrapolate(extrapolated_f_lower, extrapolated_f_upper, u);
        return extrapolated_latency;
    }

    // If none of the above, we can’t estimate
    throw std::runtime_error("Unable to estimate scan latency (unexpected case).");
}

bool ListScanLatencyEstimator::save_latency_profile(
    const std::string &filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error opening file for write: " << filename << std::endl;
        return false;
    }

    // Write a header line to give context
    // Example: "# Latency Profile for dimension=128"
    ofs << "# Latency Profile for dimension=" << d_
            << ", n_values.size=" << n_values_.size()
            << ", k_values.size=" << k_values_.size()
            << ", n_trials=" << n_trials_ << "\n";

    // 1) Write size info
    ofs << n_values_.size() << "," << k_values_.size() << "\n";

    // 2) Write n_values_
    for (size_t i = 0; i < n_values_.size(); i++) {
        ofs << n_values_[i];
        if (i + 1 < n_values_.size()) ofs << ",";
    }
    ofs << "\n";

    // 3) Write k_values_
    for (size_t j = 0; j < k_values_.size(); j++) {
        ofs << k_values_[j];
        if (j + 1 < k_values_.size()) ofs << ",";
    }
    ofs << "\n";

    // 4) Write scan_latency_model_ (N rows, each has K columns)
    for (size_t i = 0; i < n_values_.size(); i++) {
        for (size_t j = 0; j < k_values_.size(); j++) {
            ofs << scan_latency_model_[i][j];
            if (j + 1 < k_values_.size()) ofs << ",";
        }
        ofs << "\n";
    }

    ofs.close();
    return true;
}

bool ListScanLatencyEstimator::load_latency_profile(const std::string &filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        // File not found or no permission => can't load
        return false;
    }

    // Read header line
    {
        std::string header_line;
        if (!std::getline(ifs, header_line)) {
            return false;
        }
    }

    std::string line;
    // 1) Read size info
    if (!std::getline(ifs, line)) return false;
    auto tokens = split_string(line, ',');
    if (tokens.size() != 2) return false;

    int n_size = std::stoi(tokens[0]);
    int k_size = std::stoi(tokens[1]);

    // 2) Read n_values_
    if (!std::getline(ifs, line)) return false;
    tokens = split_string(line, ',');
    if (static_cast<int>(tokens.size()) != n_size) return false;
    std::vector<int> file_n_values(n_size);
    for (int i = 0; i < n_size; i++) {
        file_n_values[i] = std::stoi(tokens[i]);
    }

    // 3) Read k_values_
    if (!std::getline(ifs, line)) return false;
    tokens = split_string(line, ',');
    if (static_cast<int>(tokens.size()) != k_size) return false;
    std::vector<int> file_k_values(k_size);
    for (int j = 0; j < k_size; j++) {
        file_k_values[j] = std::stoi(tokens[j]);
    }

    // Check if they match our current n_values_ and k_values_
    if (file_n_values != n_values_ || file_k_values != k_values_) {
        return false;
    }

    // 4) Read latency matrix
    std::vector<std::vector<float> > file_latency_model(
        n_size, std::vector<float>(k_size, 0.0f));

    for (int i = 0; i < n_size; i++) {
        if (!std::getline(ifs, line)) return false;
        tokens = split_string(line, ',');
        if (static_cast<int>(tokens.size()) != k_size) return false;
        for (int j = 0; j < k_size; j++) {
            file_latency_model[i][j] = std::stof(tokens[j]);
        }
    }

    ifs.close();

    // Assign to our model
    scan_latency_model_ = std::move(file_latency_model);
    return true;
}


MaintenanceCostEstimator::MaintenanceCostEstimator(int d, float alpha, int k)
    : d_(d), alpha_(alpha), k_(k) {
    if (k_ <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (alpha_ <= 0.0f) {
        throw std::invalid_argument("alpha must be positive");
    }

    latency_estimator_ = make_shared<ListScanLatencyEstimator>(
        d_,
        DEFAULT_LATENCY_ESTIMATOR_RANGE_N,
        DEFAULT_LATENCY_ESTIMATOR_RANGE_K,
        DEFAULT_LATENCY_ESTIMATOR_NTRIALS);
}

float MaintenanceCostEstimator::compute_split_delta(int partition_size, float hit_rate, int total_partitions) const {
    // Compute overhead incurred by adding one more partition.
    float delta_overhead = latency_estimator_->estimate_scan_latency(total_partitions + 1, k_) -
                           latency_estimator_->estimate_scan_latency(total_partitions, k_);
    // Cost before splitting.
    float old_cost = latency_estimator_->estimate_scan_latency(partition_size, k_) * hit_rate;
    // Cost after splitting: assume the partition is split in half and cost doubles due to two partitions,
    // scaled by the alpha factor.
    float new_cost = latency_estimator_->estimate_scan_latency(partition_size / 2, k_) * hit_rate * (2.0f * alpha_);
    return delta_overhead + new_cost - old_cost;
}


float MaintenanceCostEstimator::compute_delete_delta(
    int partition_size, // size of the candidate (deleted) partition
    float hit_rate, // scan fraction ("hit rate") for the candidate partition
    int total_partitions,
    float avg_partition_hit_rate, // average scan fraction for each partition
    float avg_partition_size // average size of partitions
) const {
    if (total_partitions <= 1) {
        // Can't delete if there's only 1 partition
        return 0.0f;
    }

    // ----------------------------------------------------
    // 1) Structural overhead difference:
    //    = L(T-1, k) - L(T, k).
    // ----------------------------------------------------
    float latency_T = latency_estimator_->estimate_scan_latency(total_partitions, k_);
    float latency_T_minus_1 = latency_estimator_->estimate_scan_latency(total_partitions - 1, k_);
    float delta_overhead = latency_T_minus_1 - latency_T;

    // ----------------------------------------------------
    // 2) Scanning cost difference:
    //
    // Old scanning cost (approx):
    //    cost_old = (T-1)*(\bar{p}) * L(\bar{n}, k)
    //             + p_d * L(n_d, k)
    //
    // New scanning cost (approx):
    //    each of (T-1) partitions has new size \bar{n}' = \bar{n} + n_d/(T-1)
    //    and new scan fraction \bar{p}' = \bar{p} + p_d/(T-1)
    //    cost_new = (T-1)*(\bar{p}') * L(\bar{n}', k)
    // ----------------------------------------------------
    float cost_old = (total_partitions - 1) * avg_partition_hit_rate
                     * latency_estimator_->estimate_scan_latency(avg_partition_size, k_)
                     + hit_rate
                     * latency_estimator_->estimate_scan_latency(partition_size, k_);

    // Compute the "new" size and scan fraction after merging
    float merged_size = avg_partition_size + static_cast<float>(partition_size) / (total_partitions - 1);
    float merged_hit_rate = avg_partition_hit_rate + hit_rate / static_cast<float>(total_partitions - 1);

    float cost_new;
    if (partition_size < total_partitions) {
        // assume at most partition_size partitions get the extra vectors
        cost_new = partition_size * merged_hit_rate * latency_estimator_->estimate_scan_latency(avg_partition_size + 1, k_)
                   + (total_partitions - partition_size - 1) * merged_hit_rate * latency_estimator_->estimate_scan_latency(avg_partition_size, k_);
    } else {
        cost_new = (total_partitions - 1) * merged_hit_rate * latency_estimator_->estimate_scan_latency(ceil(merged_size), k_);
    }

    float delta_scanning = cost_new - cost_old;

    // ----------------------------------------------------
    // 3) Final delete delta = structural overhead + scanning delta
    // ----------------------------------------------------
    float delta = delta_overhead + delta_scanning;
    return delta;
}

float MaintenanceCostEstimator::compute_delete_delta_w_reassign(int partition_size, float hit_rate, int total_partitions, const vector<int64_t> &reassign_counts, const vector<int64_t> &reassign_sizes, const vector<float> &reassign_hit_rates) const {

    if (total_partitions <= 1) {
        // Can't delete if there's only 1 partition
        return 0.0f;
    }

    int n_reassign = reassign_counts.size();
    assert(reassign_sizes.size() == n_reassign);
    assert(reassign_hit_rates.size() == n_reassign);

    // ----------------------------------------------------
    // 1) Structural overhead difference:
    //    = L(T-1, k) - L(T, k).
    // ----------------------------------------------------
    float latency_T = latency_estimator_->estimate_scan_latency(total_partitions, k_);
    float latency_T_minus_1 = latency_estimator_->estimate_scan_latency(total_partitions - 1, k_);
    float delta_overhead = latency_T_minus_1 - latency_T;

    // ----------------------------------------------------
    // 2) Compute cost delta using reassignments
    //      Delta = Removal of old + increase of existing
    // ----------------------------------------------------
    float removal_delta = hit_rate * latency_estimator_->estimate_scan_latency(partition_size, k_);
    float reassign_delta = 0.0;
    for (int i = 0; i < n_reassign; i++) {
        float old = reassign_hit_rates[i] * latency_estimator_->estimate_scan_latency(reassign_sizes[i], k_);
        float new_size = reassign_sizes[i] + partition_size;
        float new_hit_rate = reassign_hit_rates[i] + hit_rate;
        reassign_delta += new_hit_rate * latency_estimator_->estimate_scan_latency(new_size, k_) - old;
    }

    // ----------------------------------------------------
    // 3) Final delete delta = structural overhead + scanning delta
    // ----------------------------------------------------
    float delta = delta_overhead + removal_delta + reassign_delta;
    return delta;
}


shared_ptr<ListScanLatencyEstimator> MaintenanceCostEstimator::get_latency_estimator() const {
    return latency_estimator_;
}

int MaintenanceCostEstimator::get_k() const {
    return k_;
}
