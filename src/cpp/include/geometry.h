#ifndef CPP_UTILS_GEOMETRY_H
#define CPP_UTILS_GEOMETRY_H

#include <common.h>
#include <faiss/utils/distances.h>

#define NUM_X_VALUES 1001

#define STOP 1.0e-8
#define TINY 1.0e-30
#define CRP_CHECK(cond, msg)  ((void)0)

using torch::Tensor;
using std::vector;

inline double incomplete_beta_table[NUM_X_VALUES];
inline double x_values[NUM_X_VALUES];

inline void subtract_arrays(const float *array_a, const float *array_b, float *result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = array_a[i] - array_b[i];
    }
}

inline void add_arrays(const float *array_a, const float *array_b, float *result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = array_a[i] + array_b[i];
    }
}

inline void multiply_array_by_constant(const float *array, float constant, float *result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = array[i] * constant;
    }
}

inline void divide_array_by_constant(const float *array, float constant, float *result, int dimension) {
    for (int i = 0; i < dimension; i++) {
        result[i] = array[i] / constant;
    }
}

inline float compute_norm(const float *array, int dimension) {
    float sum = 0.0f;
    for (int i = 0; i < dimension; i++) {
        sum += array[i] * array[i];
    }
    return std::sqrt(sum);
}

inline void print_array(const float *array, int dimension) {
    for (int i = 0; i < dimension; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

inline std::vector<float>
compute_boundary_distances(const torch::Tensor& query,
                           std::vector<float*>& centroids,
                           bool euclidean /* kept for API */ )
{
    int dim = query.size(0);
    const float* q = query.data_ptr<float>();

    /* --- ensure centroids[0] is the nearest --------------------- */
    size_t nearest = 0;
    float  best_d2 = std::numeric_limits<float>::max();
    for (size_t i = 0; i < centroids.size(); ++i) {
        float d2 = faiss::fvec_L2sqr(q, centroids[i], dim);
        if (d2 < best_d2) { best_d2 = d2; nearest = i; }
    }
    if (nearest != 0) std::swap(centroids[0], centroids[nearest]);

    const float* c0 = centroids[0];
    std::vector<float> d(centroids.size(), 0.0f);
    std::vector<float> v(dim);

    for (size_t j = 1; j < centroids.size(); ++j)
    {
        const float* cj = centroids[j];

        /* v = cj - c0,  ||v|| */
        faiss::fvec_sub(dim, cj, c0, v.data());
        float v_norm = std::sqrt(
            faiss::fvec_inner_product(v.data(), v.data(), dim));

        /* b = ½ (||cj||² - ||c0||²) */
        float b = 0.5f * (
            faiss::fvec_inner_product(cj, cj, dim) -
            faiss::fvec_inner_product(c0, c0, dim));

        /* signed distance   (q·v − b) / ||v|| */
        float dot_qv = faiss::fvec_inner_product(q, v.data(), dim);
        d[j] = std::fabs(dot_qv - b) / v_norm;          // plane distance
    }
    return d;   // d[0] = 0
}

inline double incomplete_beta(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return 1.0 / 0.0;

    /*The continued fraction converges nicely for x < (a+1)/(a+b+2)*/
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return (1.0 - incomplete_beta(b, a, 1.0 - x)); /*Use the fact that beta is symmetrical.*/
    }

    /*Find the first part before the continued fraction.*/
    const double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);
    const double front = exp(log(x) * a + log(1.0 - x) * b - lbeta_ab) / a;

    /*Use Lentz's algorithm to evaluate the continued fraction.*/
    double f = 1.0, c = 1.0, d = 0.0;

    int i, m;
    for (i = 0; i <= 200; ++i) {
        m = i / 2;

        double numerator;
        if (i == 0) {
            numerator = 1.0; /*First numerator is 1.0.*/
        } else if (i % 2 == 0) {
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m)); /*Even term.*/
        } else {
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1)); /*Odd term.*/
        }

        /*Do an iteration of Lentz's algorithm.*/
        d = 1.0 + numerator * d;
        if (fabs(d) < TINY) d = TINY;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (fabs(c) < TINY) c = TINY;

        const double cd = c * d;
        f *= cd;

        /*Check for stop.*/
        if (fabs(1.0 - cd) < STOP) {
            return front * (f - 1.0);
        }
    }

    return 1.0 / 0.0; /*Needed more loops, did not converge.*/
}

inline void initialize_incomplete_beta_table(int d) {
    // Initialize x_values
    double dx = 1.0 / (NUM_X_VALUES - 1);
    for (int i = 0; i < NUM_X_VALUES; i++) {
        x_values[i] = i * dx;
    }

    // Calculate parameters a and b for the incomplete beta function
    double a = (d + 1.0) / 2.0;
    double b = 0.5;

    // Precompute incomplete_beta_table
    for (int i = 0; i < NUM_X_VALUES; i++) {
        double x = x_values[i];
        incomplete_beta_table[i] = incomplete_beta(a, b, x);
    }
}

inline double incomplete_beta_lookup(double x, int d) {
    static bool incomplete_beta_table_initialized = false;
    if (!incomplete_beta_table_initialized) {
        initialize_incomplete_beta_table(d);
        incomplete_beta_table_initialized = true;
    }

    // Ensure x is within [0, 1]
    x = std::max(0.0, std::min(1.0, x));

    // Calculate index in the lookup table
    double scaled_x = x * (NUM_X_VALUES - 1);
    int x_index = static_cast<int>(scaled_x);

    // Prevent out of bounds
    x_index = std::max(0, std::min(NUM_X_VALUES - 2, x_index));

    // Option 1: Perform linear interpolation between the two nearest points
    double y1 = incomplete_beta_table[x_index];
    double y2 = incomplete_beta_table[x_index + 1];

    double dx = 1.0 / (NUM_X_VALUES - 1);
    double x1 = x_index * dx;

    double y = y1 + (x - x1) * (y2 - y1) / dx;

    return y;

    // Option 2: Perform nearest neighbor interpolation
    // return incomplete_beta_table[x_index];
}

inline double log_hypersphere_volume(double radius, int dimension) {
    double term1 = (dimension / 2.0) * std::log(M_PI);
    double term2 = std::lgamma(dimension / 2.0 + 1.0);
    double term3 = dimension * std::log(radius);
    double log_volume = term1 - term2 + term3;
    return log_volume;
}

inline double hyperspherical_cap_volume(double radius, double boundary_distance, int d, bool use_precomputed = true, bool euclidean = true) {
    if (euclidean) {
        // Ensure boundary distance is non-negative double
        boundary_distance = std::max(0.0, boundary_distance);

        // If boundary is outside or on the query radius, the cap volume is 0
        if (boundary_distance >= radius) return 0.0;

        // Calculate x for incomplete beta function
        double x = sqrt(1.0 - (boundary_distance / radius) * (boundary_distance / radius));
        x = std::clamp(x, 0.0, 1.0); // Clamp x to [0, 1]

        // Incomplete Beta parameters
        double a = 0.5 * (d + 1.0);
        double b = 0.5;
        double I = use_precomputed
            ? incomplete_beta_lookup(x, d)
            : incomplete_beta(a, b, x);

        return std::clamp(0.5 * I, 0.0, 0.5);
    } else {
        // v_i = (1/2) * [ I( sin^2(phi/2); d/2, 1/2 ) - I( sin^2(theta_i/2); d/2, 1/2 ) ]
        double log_inc_beta = std::log(incomplete_beta((d - 1) / 2.0, 0.5, std::sin(radius / 2.0) * std::sin(radius / 2.0)));
        double log_inc_beta_boundary = std::log(incomplete_beta((d - 1) / 2.0, 0.5, std::sin(boundary_distance / 2.0) * std::sin(boundary_distance / 2.0)));
        double log_cap_volume = std::log(0.5) + log_inc_beta - log_inc_beta_boundary;
        return log_cap_volume;
    }
}

inline std::vector<float>
compute_recall_profile(const std::vector<float>& boundary_distances,
                       float query_radius,
                       int   dimension,
                       std::vector<int64_t> partition_sizes = {}, // Unused in these models
                       bool  use_precomputed = true,
                       bool  euclidean       = true) // Unused in these models
{
    const int m = static_cast<int>(boundary_distances.size());
    const float eps = 1e-9f;

    // --- Edge Cases ---
    if (m <= 1) {
        if (m == 1) return {1.0f}; // Only the central partition exists
        return {}; // No partitions defined
    }
    if (query_radius <= eps) {
        std::vector<float> p(m, 0.0f);
        p[0] = 1.0f; // Query point is exactly at origin, must be in partition 0
        return p;
    }

    // Compute raw cap volumes
    std::vector<float> raw_vols(m, 0.0f);
    for (int j = 1; j < m; ++j) {
        raw_vols[j] = hyperspherical_cap_volume(query_radius, boundary_distances[j], dimension, use_precomputed, euclidean);
    }

    float P0 = 0.0f;                       // Root cell probability
    std::vector<float> P_prime(m, 0.0f); // Intermediate neighbor probabilities (k>=1)
    float P_prime_sum = 0.0f;              // Sum of P_prime[k] for k>=1

    std::vector<float> norm_vols = raw_vols; // Copy raw vols
    float S1_for_norm = 0.0f;
    for (int j = 1; j < m; ++j) S1_for_norm += norm_vols[j];

    if (S1_for_norm > eps) {
        for (int j = 1; j < m; ++j) norm_vols[j] /= S1_for_norm;
    } else {
        for (int j = 1; j < m; ++j) norm_vols[j] = 0.0f;
    }

    P0 = 1.0f;
    for (int j = 1; j < m; ++j) P0 *= (1.0f - norm_vols[j]); // Survival probability of the root cell
    P0 = std::clamp(P0, 0.0f, 1.0f);

    for (int k = 1; k < m; ++k) {
        P_prime[k] = norm_vols[k];
        P_prime_sum += P_prime[k];
    }
    // Ensure P_prime is non-negative
    for (int k = 1; k < m; ++k) P_prime[k] = std::max(0.0f, P_prime[k]);

    // // if the cluster_sizes are given, scale P_prime[k] by the size of the cluster. this is a rudimentary density estimation
    // if (partition_sizes.size() > 0) {
    //     for (int k = 1; k < m; ++k) {
    //         if (partition_sizes[k] > 0) {
    //             P_prime[k] *= static_cast<float>(partition_sizes[k]);
    //         }
    //     }
    // }


    // normalize probs
    std::vector<float> probs(m, 0.0f);
    probs[0] = P0;

    float target = 1.0f - P0;
    // Ensure target probability for neighbors is valid [0, 1]
    target = std::clamp(target, 0.0f, 1.0f);

    if (target > eps && P_prime_sum > eps) {
        float scale = target / P_prime_sum;
        for (int k = 1; k < m; ++k) {
            // Ensure final probability is non-negative
            probs[k] = std::max(0.0f, P_prime[k] * scale);
        }


        // Strict renormalization to ensure sum is exactly 1
        float current_sum_k = 0.0f;
        for (int k = 1; k < m; ++k) current_sum_k += probs[k];

        if (current_sum_k > eps) { // Avoid division by zero if sum is negligible
            float final_scale = target / current_sum_k;
            // Check if scale is finite (handles target=0 case correctly)
            if (std::isfinite(final_scale)) {
                for (int k = 1; k < m; ++k) {
                    probs[k] *= final_scale;
                    // Final clamp for safety
                    probs[k] = std::max(0.0f, probs[k]);
                }
            } else if (target <= eps) {
                // If target is zero, all neighbor probs should be zero
                for (int k = 1; k < m; ++k) probs[k] = 0.0f;
            }
        } else if (target <= eps) {
            // If target is zero and calculated sum is zero, ensure all are zero
            for (int k = 1; k < m; ++k) probs[k] = 0.0f;
        }
    }

    return probs;
}

inline std::vector<float>
compute_recall_profile_auncel(
    const std::vector<float>& boundary_distances,
    float                     query_radius,
    int                       K_neighbors,
    float                     a,
    float                     b
) {
    const size_t L = boundary_distances.size();

    // 1) Spherical-cap "angle" terms
    std::vector<float> cap_terms(L, 0.0f);
    if (query_radius > 1e-9f) {
        for (size_t j = 0; j < L; ++j) {
            float r = boundary_distances[j] / query_radius;
            r = std::clamp(r, -1.0f, 1.0f);
            cap_terms[j] = (r >= 1.0f ? 0.0f : std::acos(r));
        }
    }

    // 2) Calculate phi_values for each stage
    std::vector<float> phi_values(L + 1);
    float running_U_sum = std::accumulate(cap_terms.begin(), cap_terms.end(), 0.0f);

    for (size_t i = 0; i <= L; ++i) {
        float denominator = b - a * running_U_sum;
        if (denominator <= 1e-9f) {
            phi_values[i] = std::numeric_limits<float>::max();
        } else {
            phi_values[i] = 1.0f / denominator;
        }
        phi_values[i] = std::max(1.0f, phi_values[i]); // Ensure phi >= 1

        if (i < L) {
            running_U_sum -= cap_terms[i];
        }
    }

    // 3) Calculate cumulative recall based on phi_values
    std::vector<float> cumulative_recalls(L + 1, 0.0f);
    for (size_t i = 0; i <= L; ++i) {
        float current_phi = phi_values[i];
        float j_star = 0.0f;

        if (K_neighbors > 0) {
            if (current_phi == std::numeric_limits<float>::max()) {
                j_star = 0.0f;
            } else if (current_phi > 1e-9f) {
                j_star = std::floor((float)K_neighbors / current_phi);
                j_star = std::min(std::max(0.0f, j_star), (float)K_neighbors);
            }
        }

        float error_i = 1.0f; // Default error if K_neighbors is 0 or no items found
        if (K_neighbors > 0) {
            error_i = 1.0f - j_star / (float)K_neighbors;
        }
        cumulative_recalls[i] = std::clamp(1.0f - error_i, 0.0f, 1.0f);
    }

    // 4) Calculate per-stage incremental recall
    std::vector<float> recall_profile(L + 1, 0.0f);
    if (!cumulative_recalls.empty()) {
        recall_profile[0] = cumulative_recalls[0];
        for (size_t i = 1; i <= L; ++i) {
            recall_profile[i] = cumulative_recalls[i] - cumulative_recalls[i-1];
            recall_profile[i] = std::max(0.0f, recall_profile[i]);
        }
    }

    // 5) Normalize the profile to sum to 1
    float S = std::accumulate(recall_profile.begin(), recall_profile.end(), 0.0f);
    if (S > 1e-9f) {
        for (auto &v : recall_profile) {
            v /= S;
        }
    }
    // If S is 0 (e.g., K_neighbors=0 or zero predicted recall), profile remains all zeros.

    return recall_profile;
}

#endif // CPP_UTILS_GEOMETRY_H