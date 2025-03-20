#include "hit_count_tracker.h"

HitCountTracker::HitCountTracker(int window_size, int total_vectors)
    : window_size_(window_size),
      total_vectors_(total_vectors),
      curr_query_index_(0),
      num_queries_recorded_(0),
      running_sum_scan_fraction_(0.0f),
      current_scan_fraction_(1.0f) {
    if (window_size_ <= 0) {
        throw std::invalid_argument("Window size must be positive");
    }
    if (total_vectors_ <= 0) {
        throw std::invalid_argument("Total vectors must be positive");
    }
    per_query_hits_.resize(window_size_);
    per_query_scanned_sizes_.resize(window_size_);
}

void HitCountTracker::reset() {
    curr_query_index_ = 0;
    num_queries_recorded_ = 0;
    running_sum_scan_fraction_ = 0.0f;
    current_scan_fraction_ = 1.0f;
    per_query_hits_.clear();
    per_query_hits_.resize(window_size_);
    per_query_scanned_sizes_.clear();
    per_query_scanned_sizes_.resize(window_size_);
}

void HitCountTracker::set_total_vectors(int total_vectors) {
    if (total_vectors <= 0) {
        throw std::invalid_argument("Total vectors must be positive");
    }
    total_vectors_ = total_vectors;
}

float HitCountTracker::compute_scan_fraction(const vector<int64_t>& scanned_sizes) const {
    int sum = std::accumulate(scanned_sizes.begin(), scanned_sizes.end(), 0);
    return static_cast<float>(sum) / static_cast<float>(total_vectors_);
}

void HitCountTracker::add_query_data(const vector<int64_t>& hit_partition_ids,
                                   const vector<int64_t>& scanned_sizes) {
    if (hit_partition_ids.size() != scanned_sizes.size()) {
        throw std::invalid_argument("hit_partition_ids and scanned_sizes must be of equal length");
    }
    float query_fraction = compute_scan_fraction(scanned_sizes);

    // If we haven't filled the window yet, simply add new data.
    if (num_queries_recorded_ < window_size_) {
        per_query_hits_[num_queries_recorded_] = hit_partition_ids;
        per_query_scanned_sizes_[num_queries_recorded_] = scanned_sizes;
        running_sum_scan_fraction_ += query_fraction;
        num_queries_recorded_++;
    } else {
        // Window is full; subtract oldest query data and replace it.
        running_sum_scan_fraction_ -= compute_scan_fraction(per_query_scanned_sizes_[curr_query_index_]);
        per_query_hits_[curr_query_index_] = hit_partition_ids;
        per_query_scanned_sizes_[curr_query_index_] = scanned_sizes;
        running_sum_scan_fraction_ += query_fraction;
        curr_query_index_ = (curr_query_index_ + 1) % window_size_;
    }
    int effective_window = (num_queries_recorded_ < window_size_) ? num_queries_recorded_ : window_size_;
    current_scan_fraction_ = running_sum_scan_fraction_ / static_cast<float>(effective_window);
}

float HitCountTracker::get_current_scan_fraction() const {
    return current_scan_fraction_;
}

const vector<vector<int64_t>>& HitCountTracker::get_per_query_hits() const {
    return per_query_hits_;
}

const vector<vector<int64_t>>& HitCountTracker::get_per_query_scanned_sizes() const {
    return per_query_scanned_sizes_;
}

int HitCountTracker::get_window_size() const {
    return window_size_;
}

int64_t HitCountTracker::get_num_queries_recorded() const {
    return num_queries_recorded_;
}