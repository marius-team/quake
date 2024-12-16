//
// Created by Jason on 12/16/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef PARAMS_H
#define PARAMS_H

#include <common.h>

struct MaintenancePolicyParams {
    std::string maintenance_policy = "query_cost";
    int window_size = 1000;
    int refinement_radius = 100;
    int refinement_iterations = 3;
    int min_partition_size = 32;
    float alpha = .9;
    bool enable_split_rejection = true;
    bool enable_delete_rejection = true;

    float delete_threshold_ns = 20.0;
    float split_threshold_ns = 20.0;

    // de-drift parameters
    int k_large = 50;
    int k_small = 50;
    bool modify_centroids = true;

    // lire parameters
    int target_partition_size = 1000;
    float max_partition_ratio = 2.0;

    MaintenancePolicyParams() = default;
};

#endif //PARAMS_H
