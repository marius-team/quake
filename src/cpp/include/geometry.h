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

inline double hypersphere_surface_area(double radius, int d) {
    return 2 * std::pow(M_PI, d / 2.0) * std::pow(radius, d - 1) / std::tgamma(d / 2.0);
}

inline double compute_partial_cap_volume(double radius, double theta_min, double theta_max, int d) {
    // Compute the volume of a hyperspherical cap between angles theta_min and theta_max
    if (theta_min < 0 || theta_max > M_PI) {
        throw std::invalid_argument("Theta values out of bounds.");
    }

    double cap_volume = 0.0;

    // Using the regularized incomplete beta function
    double sin2_theta_min = std::sin(theta_min) * std::sin(theta_min);
    double sin2_theta_max = std::sin(theta_max) * std::sin(theta_max);

    double I_min = incomplete_beta((d - 1.0) / 2.0, 0.5, sin2_theta_min);
    double I_max = incomplete_beta((d - 1.0) / 2.0, 0.5, sin2_theta_max);

    double surface_area = hypersphere_surface_area(radius, d);
    cap_volume = surface_area * (I_max - I_min);

    return cap_volume;
}


inline double log_hyperspherical_cap_volume(double radius, double boundary_distance, int d, bool ratio = true,
                                            bool use_precomputed = true, bool euclidean = true) {
    double h = radius - boundary_distance;

    // Ensure h is within valid range
    h = std::max(0.0, std::min(2 * radius, h));

    if (euclidean) {
        double x = pow((2 * radius * h - h * h) / (radius * radius), 1.0);
        // use precomputed incomplete beta function
        double inc_beta;
        if (use_precomputed) {
            inc_beta = incomplete_beta_lookup(x, d);
        } else {
            inc_beta = incomplete_beta((d + 1.0) / 2.0, 0.5, x);
        }

        if (inc_beta <= 0.0 || std::isnan(inc_beta) || std::isinf(inc_beta)) {
            std::cerr << "Invalid incomplete beta value: " << inc_beta << std::endl;
            return -std::numeric_limits<double>::infinity();
        }

        double log_inc_beta = std::log(inc_beta);
        double log_cap_volume;
        if (!ratio) {
            double log_sphere_volume = log_hypersphere_volume(radius, d);
            log_cap_volume = std::log(0.5) + log_inc_beta + log_sphere_volume;
        } else {
            log_cap_volume = std::log(0.5) + log_inc_beta;
        }

        return log_cap_volume;
    } else {
        // use intersection of spherical caps
        if (ratio != true) {
            throw std::invalid_argument("Ratio must be true for dot product distance");
        }

        // v_i = (1/2) * [ I( sin^2(phi/2); d/2, 1/2 ) - I( sin^2(theta_i/2); d/2, 1/2 ) ]

        double log_inc_beta = std::log(incomplete_beta((d - 1) / 2.0, 0.5, std::sin(radius / 2.0) * std::sin(radius / 2.0)));
        double log_inc_beta_boundary = std::log(incomplete_beta((d - 1) / 2.0, 0.5, std::sin(boundary_distance / 2.0) * std::sin(boundary_distance / 2.0)));
        double log_cap_volume = std::log(0.5) + log_inc_beta - log_inc_beta_boundary;
        return log_cap_volume;
        // compute volume of the intersection of two spherical caps (from the paper: concise formulas for the volume of hyperspherical caps)


    }
}

inline vector<float> compute_intersection_volume(const Tensor &boundary_distances, float query_radius, int dimension,
                                                 bool use_precomputed = true) {
    auto boundary_distances_ptr = boundary_distances.data_ptr<float>();
    int num_partitions = boundary_distances.size(0);
    std::vector<float> partition_volumes(num_partitions, 0.0f);

    for (int j = 0; j < num_partitions; j++) {
        float boundary_distance = boundary_distances_ptr[j];

        if (boundary_distance >= query_radius) {
            partition_volumes[j] = -1e8;
            continue;
        }

        double volume_ratio = log_hyperspherical_cap_volume(query_radius, boundary_distance, dimension, false,
                                                            use_precomputed);

        partition_volumes[j] = volume_ratio;
    }

    return partition_volumes;
}

inline Tensor compute_variance_in_direction_of_query(Tensor query, Tensor centroids, Tensor variance) {
    int dimension = query.size(0);
    int num_partitions = centroids.size(0);
    auto query_ptr = query.data_ptr<float>();
    auto centroids_ptr = centroids.data_ptr<float>();
    auto variance_ptr = variance.data_ptr<float>();

    std::vector<float> variances(num_partitions, 0.0f);

    for (int j = 0; j < num_partitions; j++) {
        float *centroid = centroids_ptr + j * dimension;
        float *variance = variance_ptr + j * dimension;

        float *query_minus_centroid = new float[dimension];
        subtract_arrays(query_ptr, centroid, query_minus_centroid, dimension);

        float dot_product = faiss::fvec_inner_product(query_minus_centroid, query_minus_centroid, dimension);

        variances[j] = dot_product;
        delete[] query_minus_centroid;
    }

    return torch::tensor(variances).clone();
}

inline void check_boundary_distances(const float*          q,
                                     const std::vector<float*>& C,
                                     const std::vector<float>&  d,
                                     int dim)
{
    const float* c0 = C[0];
    std::vector<float> v(dim);

    for (size_t j = 1; j < C.size(); ++j) {
        const float* cj = C[j];

        /* signed distance via the explicit formula again */
        faiss::fvec_sub(dim, cj, c0, v.data());
        float vnorm = std::sqrt(faiss::fvec_inner_product(v.data(), v.data(), dim));
        float b     = 0.5f*(faiss::fvec_inner_product(cj,cj,dim)
                            -faiss::fvec_inner_product(c0,c0,dim));
        float dj    = std::fabs(faiss::fvec_inner_product(q, v.data(), dim) - b) / vnorm;

        if (std::fabs(dj - d[j]) > 0.01f * dj + 1e-6f)
            throw std::runtime_error(
                "boundary_distances[" + std::to_string(j) + "] inaccurate");
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
    // --- Configuration Flag ---
    // 0: Original Scaled-IE (reconstructed, uses norm_vols internally)
    // 1: Direct Proportional Allocation (P'_k = raw_vols[k])
    // 2: Independent Entry Allocation (P'_k = raw_vols[k]*P0 / (1-raw_vols[k]))
    // 3: Concentration Penalized Allocation (P'_k = raw_vols[k] * (1-S2_norm))
    // 4: Radius-Difference Weighting (P'_k = raw_vols[k] * (R-d_k))
    // 5: Simplified IE Approx Weighting (P'_k = raw_vols[k] * (1 - 0.5*S1 + 0.5*v_k))
    const int method_flag = 0; // <<< CHANGE THIS FLAG FOR ABLATION STUDY >>>
    // ---

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

    // --- Step 1: Compute Raw Cap Volumes (Common to all methods) ---
    std::vector<float> raw_vols(m, 0.0f);
    for (int j = 1; j < m; ++j) {
        // Ensure boundary distance is non-negative double
        double d = std::max(0.0, static_cast<double>(boundary_distances[j]));
        // If boundary is outside or on the query radius, the cap volume is 0
        if (d >= query_radius) continue;

        // Calculate x for incomplete beta function
        double x = 1.0 - (d / query_radius) * (d / query_radius);
        x = std::clamp(x, 0.0, 1.0); // Clamp x to [0, 1]

        // Incomplete Beta parameters
        double a = 0.5 * (dimension + 1.0);
        double b = 0.5;
        double I = use_precomputed
            ? incomplete_beta_lookup(x, dimension) // Needs definition
            : incomplete_beta(a, b, x);            // Needs definition

        // Raw volume is half the normalized surface area ratio, clamped to [0, 0.5]
        raw_vols[j] = static_cast<float>(std::clamp(0.5 * I, 0.0, 0.5));
    }

    // --- Method Selection ---
    float P0 = 0.0f;                       // Root cell probability
    std::vector<float> P_prime(m, 0.0f); // Intermediate neighbor probabilities (k>=1)
    float P_prime_sum = 0.0f;            // Sum of P_prime[k] for k>=1

    switch (method_flag) {
        case 0: { // M0: Original Scaled-IE (reconstructed)
            //std::cout << "Using Method 0: Original Scaled-IE (Cumulative)" << std::endl;
            // Uses intermediate normalization and P0 based on normalized vols
            std::vector<float> norm_vols = raw_vols; // Copy raw vols
            float S1_for_norm = 0.0f;
            for (int j = 1; j < m; ++j) S1_for_norm += norm_vols[j];

            if (S1_for_norm > eps) {
                for (int j = 1; j < m; ++j) norm_vols[j] /= S1_for_norm;
            } else {
                for (int j = 1; j < m; ++j) norm_vols[j] = 0.0f;
            }

            // Calculate P0 using normalized vols (as per original logic)
            P0 = 1.0f;
            for (int j = 1; j < m; ++j) {
                P0 *= (1.0f - norm_vols[j]);
            }
            P0 = std::clamp(P0, 0.0f, 1.0f);

            // Calculate P_prime using cumulative S1, S2 on normalized vols
            float S1_cum = 0.0f, S2_cum = 0.0f;
            // Use lambda = m as per original code's default, or allow tuning
            float lambda = static_cast<float>(m); // Could be tuned here if needed

            for (int k = 1; k < m; ++k) {
                S1_cum += norm_vols[k]; // Update cumulative sums *before* using them for k
                S2_cum += norm_vols[k] * norm_vols[k];

                // Calculate R factor using *cumulative* S1, S2 up to k
                // Note: Original code used S1*S1 - S2. Ensure S1_cum is used correctly.
                float R = 1.0f - S1_cum + 0.5f * lambda * (S1_cum * S1_cum - S2_cum);

                P_prime[k] = norm_vols[k] * std::max(0.0f, R);
                P_prime_sum += P_prime[k]; // Accumulate sum for final normalization
            }
            // Note: This calculation is order-dependent based on how boundaries are indexed.
            break;
        }

        case 1: { // M1: Direct Proportional Allocation
             //std::cout << "Using Method 1: Direct Proportional" << std::endl;
            // P0 based on raw volumes

            std::vector<float> norm_vols = raw_vols; // Copy raw vols
            float S1_for_norm = 0.0f;
            for (int j = 1; j < m; ++j) S1_for_norm += norm_vols[j];

            if (S1_for_norm > eps) {
                for (int j = 1; j < m; ++j) norm_vols[j] /= S1_for_norm;
            } else {
                for (int j = 1; j < m; ++j) norm_vols[j] = 0.0f;
            }

            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - norm_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            // P_prime is just raw_vols
            for (int k = 1; k < m; ++k) {
                P_prime[k] = norm_vols[k];
                P_prime_sum += P_prime[k];
            }
            // Ensure P_prime is non-negative
            for (int k = 1; k < m; ++k) P_prime[k] = std::max(0.0f, P_prime[k]);
            break;
        }

        case 2: { // M2: Independent Entry Allocation
             //std::cout << "Using Method 2: Independent Entry" << std::endl;
            // P0 based on raw volumes
            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - raw_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            if (P0 > eps) { // Avoid issues if P0 is near zero
                 for (int k = 1; k < m; ++k) {
                     float v_k = raw_vols[k];
                     float one_minus_v_k = 1.0f - v_k;
                     // Check for valid v_k and avoid division by zero/small number
                     if (v_k > eps && one_minus_v_k > eps) {
                          P_prime[k] = v_k * P0 / one_minus_v_k;
                          // Clamp potentially large values if v_k is very close to 1?
                          // P_prime[k] = std::min(P_prime[k], 1.0f); // Optional clamping
                          P_prime_sum += P_prime[k];
                     }
                      P_prime[k] = std::max(0.0f, P_prime[k]); // Ensure non-negative
                 }
            } // else P_prime remains 0
            break;
        }

        case 3: { // M3: Concentration Penalized Allocation
             //std::cout << "Using Method 3: Concentration Penalized" << std::endl;
            // P0 based on raw volumes
            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - raw_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            float S1_raw = 0.0f;
            float S2_raw = 0.0f;
            for (int j = 1; j < m; ++j) {
                 S1_raw += raw_vols[j];
                 S2_raw += raw_vols[j] * raw_vols[j];
            }

            float penalty_factor = 0.0f;
            if (S1_raw > eps) {
                 float S1_raw_sq = S1_raw * S1_raw;
                 // Avoid division by zero if S1_raw_sq is somehow zero or negative
                 if (S1_raw_sq > eps) {
                     float S2_norm = S2_raw / S1_raw_sq;
                     penalty_factor = std::max(0.0f, 1.0f - S2_norm);
                 }
                 // Optional: include lambda=m scaling -> penalty_factor *= 0.5f * m;
            }

            for (int k = 1; k < m; ++k) {
                 P_prime[k] = raw_vols[k] * penalty_factor; // Apply uniform penalty
                 P_prime_sum += P_prime[k];
            }
            break;
        }

        case 4: { // M4: Radius-Difference Weighting
             //std::cout << "Using Method 4: Radius-Difference Weighting" << std::endl;
            // P0 based on raw volumes
            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - raw_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            for (int k = 1; k < m; ++k) {
                float v_k = raw_vols[k];
                if (v_k > eps) {
                    // Use boundary_distances directly, ensure non-negative
                    float d_k = std::max(0.0f, boundary_distances[k]);
                    // Calculate non-uniform weight based on distance from radius
                    float mu_k_tilde = std::max(0.0f, query_radius - d_k);
                    P_prime[k] = v_k * mu_k_tilde;
                    P_prime_sum += P_prime[k];
                }
            }
            break;
        }

        case 5: { // M5: Simplified IE Approx Weighting
             //std::cout << "Using Method 5: Simplified IE Approx" << std::endl;
            // P0 based on raw volumes
            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - raw_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            float S1_raw = 0.0f;
            for(int j=1; j<m; ++j) S1_raw += raw_vols[j];

            for (int k = 1; k < m; ++k) {
                 float v_k = raw_vols[k];
                 if (v_k > eps) {
                     // Calculate non-uniform scaling factor
                     float mu_k_tilde = std::max(0.0f, 1.0f - 0.5f * S1_raw + 0.5f * v_k);
                     P_prime[k] = v_k * mu_k_tilde;
                     P_prime_sum += P_prime[k];
                 }
            }
            break;
         }

        case 6: {
            std::vector<float> norm_vols = raw_vols; // Copy raw vols
            float S1_for_norm = 0.0f;
            for (int j = 1; j < m; ++j) S1_for_norm += norm_vols[j];

            if (S1_for_norm > eps) {
                for (int j = 1; j < m; ++j) norm_vols[j] /= S1_for_norm;
            } else {
                for (int j = 1; j < m; ++j) norm_vols[j] = 0.0f;
            }

            P0 = 1.0f;
            for (int j = 1; j < m; ++j) P0 *= (1.0f - norm_vols[j]);
            P0 = std::clamp(P0, 0.0f, 1.0f);

            float S2_raw = 0.0f;
            for (int j = 1; j < m; ++j) {
                S2_raw += norm_vols[j];
            }

            if (S2_raw > eps) {
                for (int k = 1; k < m; ++k) {
                    float v_k = norm_vols[k];
                    S2_raw -= v_k; // Update S2_raw for next iteration
                    float rel_L2_sq = (.5 * v_k * S2_raw);
                    float mu_k_tilde = 1 / (1.0f + rel_L2_sq); // Non-uniform scaling
                    P_prime[k] = v_k * std::max(mu_k_tilde, .001f);
                    P_prime_sum += P_prime[k];
                }
            } else { // If S2 is zero, all raw_vols are zero, P_prime remains zero
                for (int k = 1; k < m; ++k) {
                    P_prime[k] = norm_vols[k]; // Should be 0
                    P_prime_sum += P_prime[k]; // Should be 0
                }
            }
            // Ensure P_prime is non-negative
            for (int k = 1; k < m; ++k) P_prime[k] = std::max(0.0f, P_prime[k]);

            break;
        }

        default:
            // Handle unknown method flag, e.g., throw error or default to Method 1
            //std::cerr << "Error: Unknown method_flag: " << method_flag << std::endl;
            throw std::runtime_error("Unknown method_flag in compute_recall_profile");
            // Or fallback: goto method1_label; (using labels/goto is generally discouraged)
    }

    // if the cluster_sizes are given, we can use them to scale the probabilities
    if (partition_sizes.size() > 0) {
        for (int k = 1; k < m; ++k) {
            if (partition_sizes[k] > 0) {
                P_prime[k] *= static_cast<float>(partition_sizes[k]);
            }
        }
    }

    // --- Final Normalization ---
    std::vector<float> probs(m, 0.0f);
    probs[0] = P0; // Assign calculated P0 for the chosen method

    float target = 1.0f - P0;
    // Ensure target probability for neighbors is valid [0, 1]
    target = std::clamp(target, 0.0f, 1.0f);

    if (target > eps && P_prime_sum > eps) {
        float scale = target / P_prime_sum;
        for (int k = 1; k < m; ++k) {
            // Ensure final probability is non-negative
            probs[k] = std::max(0.0f, P_prime[k] * scale);
        }

        // --- Strict Renormalization ---
        // This step ensures probs[1..m-1] sum *exactly* to target,
        // correcting minor float inaccuracies or effects of max(0,...)
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
        // --- End Optional Strict Renormalization ---

    }

    return probs;
}

// inline std::vector<float>
// compute_recall_profile(const std::vector<float>& boundary_distances,
//                        float query_radius,
//                        int   dimension,
//                        std::vector<int64_t> partition_sizes = {},
//                        bool  use_precomputed = true,
//                        bool  euclidean       = true)
// {
//     const int m = static_cast<int>(boundary_distances.size());
//     const float eps = 1e-9f;
//
//     if (m <= 1) {
//         if (m == 1) return {1.0f};
//         return {};
//     }
//     if (query_radius <= eps) {
//         std::vector<float> p(m, 0.0f);
//         p[0] = 1.0f;
//         return p;
//     }
//
//     // 1) compute raw cap‐volumes vols[1..m-1]
//     std::vector<float> vols(m, 0.0f);
//     for (int j = 1; j < m; ++j) {
//         double d = std::max(0.0, static_cast<double>(boundary_distances[j]));
//         if (d >= query_radius) continue;
//         double x = 1.0 - (d/query_radius)*(d/query_radius);
//         x = std::clamp(x, 0.0, 1.0);
//
//         double a = 0.5*(dimension + 1.0), b = 0.5, I = 0.0;
//         try {
//             I = use_precomputed
//                 ? incomplete_beta_lookup(x, dimension)
//                 : incomplete_beta(a, b, x);
//         } catch(...) {
//             I = 0.0;
//         }
//         vols[j] = std::clamp(0.5 * I, 0.0, 0.5);
//     }
//
//     // // 2) normalize vols[1..] to sum = 1
//     // float sumv = std::accumulate(vols.begin()+1, vols.end(), 0.0f);
//     // if (sumv > eps) {
//     //     for (int j = 1; j < m; ++j) vols[j] /= sumv;
//     // } else {
//     //     std::fill(vols.begin()+1, vols.end(), 0.0f);
//     // }
//
//     // 3) compute reference P0 = ∏_{j=1..m-1}(1 - vols[j])
//     float P0 = 1.0f;
//     for (int j = 1; j < m; ++j) {
//         P0 *= (1.0f - vols[j]);
//     }
//
//     // 4) compute raw Pk for k>=1 under chosen mode
//     std::vector<float> raw(m, 0.0f);
//
//     // 4) if scaled‐IE, compute λ = 1 / ∑v_j^2
//     float lambda = (float) boundary_distances.size();
//     // float lambda =
//     // if (use_scaled_ie) {
//     //     float sum2 = 0.0f;
//     //     for (int j=1;j<m;++j) sum2 += vols[j]*vols[j];
//     //     lambda = (sum2>eps ? 1.0f/sum2 : 1.0f);
//     // }
//
//     std::cout << "lambda: " << lambda << std::endl;
//
//     // -- Scaled‐IE: λ·(S1²−S2) in pairwise term --
//     float S1 = 0.0f, S2 = 0.0f;
//     for (int k = 1; k < m; ++k) {
//         S1 += vols[k];
//         S2 += vols[k]*vols[k];
//         float R = 1.0f
//                   - S1
//                   + 0.5f*lambda*(S1*S1 - S2);
//         raw[k] = vols[k] * std::max(0.0f, R);
//     }
//     std::cout << "S1: " << S1 << " S2: " << S2 << std::endl;
//
//     // float scale_term = 1.0f;
//     // for (int k = 1; k < m; ++k) {
//         // raw[k] = vols[k] * scale_term;
//         // scale_term *= (1.0f - vols[k]);
//     // }
//
//     // for (int k = 1; k < m; ++k) {
//     //     float scale_term = P0 / (1 - vols[k]);
//     //     raw[k] = vols[k] * scale_term;
//     // }
//
//     // 5) scale k>=1 so they sum to (1 - P0), leave P0 unchanged
//     float sum_raw = std::accumulate(raw.begin()+1, raw.end(), 0.0f);
//     std::vector<float> probs(m, 0.0f);
//     probs[0] = P0;
//
//     float target = 1.0f - P0;
//     if (sum_raw > eps) {
//         float scale = target / sum_raw;
//         for (int k = 1; k < m; ++k) {
//             probs[k] = std::max(0.0f, raw[k] * scale);
//         }
//     }
//
//     return probs;
// }

// inline std::vector<float>
// compute_recall_profile(const std::vector<float>& boundary_distances,
//                        float query_radius,
//                        int   dimension,
//                        std::vector<int64_t> partition_sizes = {},
//                        bool  use_precomputed = true,
//                        bool  euclidean       = true)
// {
//     const int m = static_cast<int>(boundary_distances.size());
//     const float float_epsilon = 1e-9f; // Small value for float comparisons
//
//     // Handle edge cases: no cells or only the home cell
//     if (m == 0) {
//         return {};
//     }
//     if (m == 1) {
//         return {1.0f}; // Only home cell exists, probability is 1
//     }
//      if (query_radius <= float_epsilon) {
//         // If query radius is zero or negligible, k-NN must be in home cell
//         std::vector<float> probs(m, 0.0f);
//         probs[0] = 1.0f;
//         return probs;
//     }
//
//
//     std::vector<float> vols(m, 0.0f); // Raw cap volume ratios (v_j for j>=1)
//
//     // --- 1. Calculate raw cap ratios v_j for neighbors j = 1 to m-1 ---
//     for (int j = 1; j < m; ++j) {
//         float d_edge = boundary_distances[j]; // Distance to bisector plane j
//
//         // Clamp distance to be non-negative (numerical safety)
//         d_edge = std::max(0.0f, d_edge);
//
//         // If plane is beyond radius, cap volume is zero
//         if (d_edge >= query_radius) {
//              vols[j] = 0.0f;
//              continue;
//         }
//
//         // Calculate x = 1 - (d_edge/query_radius)^2 for betainc arg I_x(a, b)
//         // Use doubles for intermediate calculation for precision
//         double r_double = static_cast<double>(query_radius);
//         double d_double = static_cast<double>(d_edge);
//         double ratio_d_r = d_double / r_double;
//         double x = 1.0 - ratio_d_r * ratio_d_r;
//
//         // Clamp x to [0, 1] due to potential floating point inaccuracies
//         x = std::clamp(x, 0.0, 1.0);
//
//         // Parameters for incomplete beta I_x(a, b) for hyperspherical cap volume ratio
//         double a_param = 0.5 * (static_cast<double>(dimension) + 1.0);
//         double b_param = 0.5;
//
//         double beta_inc_value = 0.0;
//         try {
//              beta_inc_value = use_precomputed
//                                ? incomplete_beta_lookup(x, dimension)
//                                : incomplete_beta(a_param, b_param, x);
//         } catch (const std::exception& e) {
//             // Warn if beta calculation fails, default volume to 0
//             // Consider more robust error handling if needed
//             #if DEBUG // Optional: Only print in debug builds
//             fprintf(stderr,
//                     "Warning: Incomplete Beta calculation failed for x=%.6f, dim=%d. Defaulting cap vol to 0. Error: %s\n",
//                     x, dimension, e.what());
//             #endif
//              beta_inc_value = 0.0;
//         }
//
//         // Cap volume ratio = 0.5 * I_x(a, b)
//         vols[j] = 0.5f * static_cast<float>(beta_inc_value);
//         // Clamp result to ensure it's within valid range [0, 0.5]
//         vols[j] = std::clamp(vols[j], 0.0f, 0.5f);
//     }
//
//     // --- 2. Calculate mu = sum(v_j) for j>=1 ---
//     float mu_total = std::accumulate(vols.begin() + 1, vols.end(), 0.0f);
//
//     // normalize volumes to sum to 1.0
//     if (mu_total > float_epsilon) {
//         for (int j = 1; j < m; ++j) {
//             vols[j] /= mu_total;
//         }
//     } else {
//         // If mu is near zero, set all volumes to zero
//         std::fill(vols.begin() + 1, vols.end(), 0.0f);
//     }
//     // --- 4. Calculate Improved P0 = max(exp(-mu), 1 - v1) ---
//
//     vector<float> survival_function(m, 1.0f);
//     int k_neighbors = 100;
//     for (int j = m-1; j > 0; --j) {
//         survival_function[j-1] = survival_function[j] * (1.0f - (vols[j]));
//     }
//
//     // --- 5. Calculate neighbor probabilities P_j = (1-P0)*v_j/mu ---
//     std::vector<float> probs(m, 0.0f);
//     probs[0] = survival_function[0];
//
//     // for (int j = 1; j < m; j++) {
//     //     probs[j] = vols[j] * (survival_function[0] / (1 - vols[j]));
//     // }
//
//     // for (int j = 1; j < m; ++j) {
//     //     probs[j] = survival_function[j] - survival_function[j - 1];
//     // }
//
//     float remaining_mass = 1.0f - probs[0];
//     float cum_v = 0.0f;                     // Σ_{i<j} v_i
//
//     for (int j = 1; j < m; ++j) {
//         cum_v += vols[j];                  // add v_j for denom of next step
//         float denom = 1.0f - (cum_v - vols[j]);   // 1 - Σ_{i<j} v_i
//         if (denom < 1e-8f) denom = 1e-8f;         // numeric guard
//
//         // exclusive volume for cap j
//         float e_j = remaining_mass * (vols[j] / denom);
//         probs[j]   = e_j;
//         remaining_mass -= e_j;
//     }
//
//     // // // Distribute remaining probability only if there's probability to distribute
//     // // // AND the sum of volumes 'mu' is non-zero (to avoid division by zero).
//     // float accum = probs[0];
//     // if (mu_total > float_epsilon) {
//     //     for (int j = 1; j < m; ++j) {
//     //         probs[j] = (accum / (1 - vols[j])) - accum; // reverse the survival function
//     //         probs[j] = std::clamp(probs[j], 0.0f, 1.0f);
//     //         accum += probs[j];
//     //     }
//     // }
//     // If P_neighbors_total or mu is near zero, probs[1...m-1] remain 0.0f.
//
//     // --- 6. Final Renormalization (Safety Net for Floating Point Drift) ---
//     float sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
//
//     if (sum_probs > float_epsilon) {
//         float inv_sum = 1.0f / sum_probs;
//         for (int j = 0; j < m; ++j) {
//             // Ensure probabilities are non-negative after division
//             probs[j] = std::max(0.0f, probs[j] * inv_sum);
//         }
//         // Ensure sum is exactly 1.0, assign difference to largest element (usually P0)
//         float final_sum_check = std::accumulate(probs.begin(), probs.end(), 0.0f);
//         probs[0] += (1.0f - final_sum_check); // Add residual to P0
//
//
//     } else {
//         // If sum is zero (should be rare unless m=0), reset to P0=1.
//          if (m > 0) {
//              std::fill(probs.begin() + 1, probs.end(), 0.0f);
//              probs[0] = 1.0f;
//          }
//     }
//
//     return probs;
// }

inline float compute_intersection_volume_one(float boundary_distance, float query_radius, int dimension) {
    if (boundary_distance >= query_radius) {
        return -1e8;
    }

    double volume_ratio = log_hyperspherical_cap_volume(query_radius, boundary_distance, dimension, true);

    return volume_ratio;
}

inline Tensor estimate_overlap(const Tensor &new_centroid, const Tensor &old_centroid, const Tensor &nbr_centroids) {
    Tensor residual = new_centroid - old_centroid;
    int dimension = new_centroid.size(0);

    vector<float> old_boundary_distance(nbr_centroids.size(0), -1.0f);
    vector<float> new_boundary_distance(nbr_centroids.size(0), -1.0f);

    const float *residual_ptr = residual.data_ptr<float>();
    const float *new_centroid_ptr = new_centroid.data_ptr<float>();
    const float *old_centroid_ptr = old_centroid.data_ptr<float>();
    const float *nbr_centroids_ptr = nbr_centroids.data_ptr<float>();

    std::vector<float> line_vector(dimension);
    std::vector<float> midpoint(dimension);
    std::vector<float> projection(dimension);

    // compute distance to old boundary
    for (int j = 0; j < nbr_centroids.size(0); j++) {
        subtract_arrays(nbr_centroids_ptr + (dimension * j), old_centroid_ptr, line_vector.data(), dimension);
        divide_array_by_constant(line_vector.data(), 2.0f, midpoint.data(), dimension);
        float norm = faiss::fvec_inner_product(midpoint.data(), midpoint.data(), dimension);
        norm = std::sqrt(norm);
        old_boundary_distance[j] = norm;
    }

    // compute distance to new boundary
    for (int j = 0; j < nbr_centroids.size(0); j++) {
        subtract_arrays(nbr_centroids_ptr + (dimension * j), new_centroid_ptr, line_vector.data(), dimension);
        divide_array_by_constant(line_vector.data(), 2.0f, midpoint.data(), dimension);
        float norm = faiss::fvec_inner_product(midpoint.data(), midpoint.data(), dimension);
        norm = std::sqrt(norm);
        new_boundary_distance[j] = norm;
    }

    Tensor overlap_ratio = torch::empty({nbr_centroids.size(0)}, torch::kFloat32);

    // for each neighbor, compute the hyperspherical cap volume, where the radius of the sphere is the distance to new boundary
    // and the old boundary distance gives the height of the cap
    float mean_new_boundary_distance = 0.0f;
    float mean_old_boundary_distance = 0.0f;
    for (int j = 0; j < nbr_centroids.size(0); j++) {
        mean_new_boundary_distance += new_boundary_distance[j];
        mean_old_boundary_distance += old_boundary_distance[j];
    }
    mean_new_boundary_distance /= nbr_centroids.size(0);
    mean_old_boundary_distance /= nbr_centroids.size(0);

    for (int j = 0; j < nbr_centroids.size(0); j++) {
        overlap_ratio[j] = abs(new_boundary_distance[j] - old_boundary_distance[j]) / mean_old_boundary_distance;
    }

    return overlap_ratio;
}

#endif // CPP_UTILS_GEOMETRY_H
