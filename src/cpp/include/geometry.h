#ifndef CPP_UTILS_GEOMETRY_H
#define CPP_UTILS_GEOMETRY_H

#include <common.h>
#include <faiss/utils/distances.h>

#define NUM_X_VALUES 1001

#define STOP 1.0e-8
#define TINY 1.0e-30

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

inline vector<float> compute_boundary_distances(const Tensor &query, vector<float *> centroids, bool euclidean = true) {

    auto start = std::chrono::high_resolution_clock::now();
    int dimension = query.size(0);

    std::vector<float> boundary_distances(centroids.size(), -1.0f);

    const float *query_ptr = query.data_ptr<float>();
    const float *nearest_centroid_ptr = centroids[0];

    vector<float> line_vector(dimension);
    vector<float> midpoint(dimension);
    vector<float> residual(dimension);

    auto end = std::chrono::high_resolution_clock::now();

    // used for euclidean distance
    if (euclidean) {
        // Compute residual: r = q - c0.
        faiss::fvec_sub(dimension, query_ptr, nearest_centroid_ptr, residual.data());

        // For each centroid j (starting at index 1).
        for (int j = 1; j < centroids.size(); j++) {
            // Compute v = c_j - c0.
            const float* c_j = centroids[j];
            faiss::fvec_sub(dimension, c_j, nearest_centroid_ptr, line_vector.data());

            // Compute squared norm: A2 = ||v||^2.
            float A2 = faiss::fvec_inner_product(line_vector.data(), line_vector.data(), dimension);
            float A = std::sqrt(A2);  // Guaranteed nonzero.

            // Compute dot product: dot = <r, v>.
            float dot_val = faiss::fvec_inner_product(residual.data(), line_vector.data(), dimension);

            // Instead of computing dot_val/A and 0.5*A separately,
            // we compute: d = |dot_val - 0.5 * A2| / A.
            float d = std::fabs(dot_val - 0.5f * A2) / A;
            boundary_distances[j] = d;
        }
    } else {
        // for dot product distance
        float residual_angle = faiss::fvec_inner_product(query_ptr, nearest_centroid_ptr, dimension);
        for (int j = 1; j < centroids.size(); j++) {
            // get angle of the bisector using dot product
            subtract_arrays(centroids[j], nearest_centroid_ptr, line_vector.data(), dimension);
            divide_array_by_constant(line_vector.data(), 2.0f, midpoint.data(), dimension);
            add_arrays(nearest_centroid_ptr, midpoint.data(), midpoint.data(), dimension);
            float norm = faiss::fvec_inner_product(midpoint.data(), midpoint.data(), dimension);
            norm = std::sqrt(norm);
            divide_array_by_constant(midpoint.data(), norm, midpoint.data(), dimension);
            float boundary_angle = faiss::fvec_inner_product(query_ptr, midpoint.data(), dimension);
            boundary_distances[j] = std::acos(boundary_angle);
        }
    }

    return boundary_distances;
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
        double x = std::sqrt((2 * radius * h - h * h) / (radius * radius));
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

inline vector<float> compute_recall_profile(vector<float> boundary_distances, float query_radius, int dimension,
                                     vector<int64_t> partition_sizes = {}, bool use_precomputed = true,
                                     bool euclidean = true) {

    // boundary_distances shape is (num_partitions,) and num_partitions must be greater than 1
    if (boundary_distances.size() < 2) {
        throw std::runtime_error("Boundary distances must have at least 2 partitions to create an estimate.");
    }

    int num_partitions = boundary_distances.size();
    vector<float> partition_probabilities(num_partitions, 0.0f);

    double total_volume = 0.0;
    bool weigh_using_partition_sizes = partition_sizes.size() == num_partitions;

    for (int j = 1; j < num_partitions; j++) {
        float boundary_distance = boundary_distances[j];

        if (boundary_distance >= query_radius) {
            partition_probabilities[j] = 0.0;
            continue;
        }

        double volume_ratio = std::exp(
            log_hyperspherical_cap_volume(query_radius,
                boundary_distance,
                dimension,
                true,
                use_precomputed,
                euclidean));
        partition_probabilities[j] = (volume_ratio > 0.0) ? volume_ratio : 0.0;
    }

    // TODO: Implement a better way to compute the probabilities for the first partition. This heuristic works well on tested datasets.
    partition_probabilities[0] = 2.0 * partition_probabilities[1];
    // partition_probabilities[0] = 1 - partition_probabilities[1];

    // if (weigh_using_partition_sizes) {
    //     for (int j = 0; j < num_partitions; j++) {
    //         partition_probabilities[j] *= partition_sizes[j];
    //     }
    // }

    // Ensure the probabilities sum to 1
    double sum_probabilities = 0.0;
    for (int j = 0; j < num_partitions; j++) {
        sum_probabilities += partition_probabilities[j];
    }
    if (sum_probabilities > 0.0f) {
        for (int j = 0; j < num_partitions; j++) {
            partition_probabilities[j] /= sum_probabilities;
        }
    } else {
        for (int j = 0; j < num_partitions; j++) {
            partition_probabilities[j] = 1.0 / num_partitions;
        }
    }

    // Compute the recall profile
    // Tensor recall_profile = torch::cumsum(probabilities_tensor, 0);

    return partition_probabilities;
}

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
