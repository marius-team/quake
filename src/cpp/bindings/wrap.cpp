//
// Created by Jason on 7/25/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUAKE_WRAP_H
#define QUAKE_WRAP_H

#include "common.h"
#include <quake_index.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include "torch/extension.h"
#include <pybind11/stl/filesystem.h>

// Helper macros for converting constants to strings
#define TOSTRING(x) + std::to_string(x)

namespace py = pybind11;

using py::arg;
using py::class_;
using py::enum_;
using py::init;
using py::module;

using std::shared_ptr;

using faiss::idx_t;

/**
 * @brief Pybind11 module definition for the Quake bindings.
 *
 * This module exposes the following classes:
 *  - QuakeIndex: The central class for building, searching, and updating the index.
 *  - MaintenanceTimingInfo: Contains timing details for maintenance operations.
 *  - BuildTimingInfo: Contains timing information for the build phase.
 *  - ModifyTimingInfo: Contains timing info for add/remove operations.
 *  - SearchTimingInfo: Contains detailed timing statistics for search.
 *  - MaintenancePolicyParams: Parameters to control the maintenance policy.
 *  - IndexBuildParams: Parameters used during the index build.
 *  - SearchParams: Parameters used during index search.
 *  - SearchResult: The result structure returned from a search.
 */
PYBIND11_MODULE(_bindings, m) {
    m.doc() = R"pbdoc(
        Quake Python Bindings
        ----------------------
        This module provides Python access to the Quake index. Use it to build, search, update,
        and maintain your index.
    )pbdoc";

    /*********** QuakeIndex Binding ***********/
    class_<QuakeIndex, shared_ptr<QuakeIndex>>(m, "QuakeIndex")
        .def(init<int>(), arg("current_level") = 0,
             "Create a new QuakeIndex (default current_level = 0).")
        .def("build", &QuakeIndex::build,
             ([]() -> const char* {
                 static const std::string doc = std::string("Build the index from a tensor of vectors and a corresponding tensor of IDs.\n\n"
                     "Args:\n"
                     "    x (Tensor): Tensor of shape [num_vectors, dimension].\n"
                     "    ids (Tensor): Tensor of shape [num_vectors].\n"
                     "    build_params (IndexBuildParams): Parameters for building the index. Defaults include:\n"
                     "         - niter: default = ") + std::to_string(DEFAULT_NITER) + "\n"
                     "         - metric: default = " + std::string(DEFAULT_METRIC) + "\n"
                     "         - num_workers: default = " + std::to_string(DEFAULT_NUM_WORKERS);
                 return doc.c_str();
             })())
        .def("search", &QuakeIndex::search,
             ([]() -> const char* {
                 static const std::string doc = std::string("Search the index for nearest neighbors.\n\n"
                     "Args:\n"
                     "    x (Tensor): Query tensor of shape [num_queries, dimension].\n"
                     "    search_params (SearchParams): Parameters for the search operation. Defaults include:\n"
                     "         - k: default = ") + std::to_string(DEFAULT_K) + "\n"
                     "         - nprobe: default = " + std::to_string(DEFAULT_NPROBE) + "\n"
                     "         - recall_target: default = " + std::to_string(DEFAULT_RECALL_TARGET);
                 return doc.c_str();
             })())
        .def("get", &QuakeIndex::get,
             "Retrieve vectors from the index by ID.\n\n"
             "Args:\n"
             "    ids (Tensor): Tensor of IDs to retrieve.")
        .def("get_ids", &QuakeIndex::get_ids, "Return all vector IDs stored in the index.")
        .def("add", &QuakeIndex::add,
             "Add new vectors to the index.\n\n"
             "Args:\n"
             "    x (Tensor): Tensor of vectors to add.\n"
             "    ids (Tensor): Tensor of corresponding IDs.")
        .def("remove", &QuakeIndex::remove,
             "Remove vectors from the index.\n\n"
             "Args:\n"
             "    ids (Tensor): Tensor of IDs to remove.")
        .def("maintenance", &QuakeIndex::maintenance,
             "Perform maintenance operations on the index (e.g., splits and merges).\n"
             "Returns timing information for the maintenance operation.")
        .def("initialize_maintenance_policy", &QuakeIndex::initialize_maintenance_policy,
             "Initialize the maintenance policy for the index.\n\n"
             "Args:\n"
             "    maintenance_policy_params (MaintenancePolicyParams): Parameters for the maintenance policy.")
        .def("save", &QuakeIndex::save,
             "Save the index to a specified path.\n\n"
             "Args:\n"
             "    path (str): The path to save the index.")
        .def("load", &QuakeIndex::load,
             "Load an index from a specified path.\n\n"
             "Args:\n"
             "    path (str): The path from which to load the index.\n"
             "    n_workers (int, optional): Number of workers for query processing (default = 0).")
        .def("ntotal", &QuakeIndex::ntotal,
             "Return the total number of vectors stored in the index.")
        .def("nlist", &QuakeIndex::nlist,
             "Return the number of partitions (lists) in the index.")
        .def_readonly("parent", &QuakeIndex::parent_,
            "Return the parent index over the centroids.")
        .def_readonly("current_level", &QuakeIndex::current_level_,
             "The current level of the index.");

    /*********** IndexBuildParams Binding ***********/
    class_<IndexBuildParams, shared_ptr<IndexBuildParams>>(m, "IndexBuildParams")
        .def(init<>())
        .def_readwrite("nlist", &IndexBuildParams::nlist,
             (std::string("Number of clusters (lists). default = ") + std::to_string(DEFAULT_NLIST)).c_str())
        .def_readwrite("niter", &IndexBuildParams::niter,
             (std::string("Number of k-means iterations. default = ") + std::to_string(DEFAULT_NITER)).c_str())
        .def_readwrite("metric", &IndexBuildParams::metric,
             (std::string("Distance metric. default = ") + DEFAULT_METRIC).c_str())
        .def_readwrite("num_workers", &IndexBuildParams::num_workers,
             (std::string("Number of workers. default = ") + std::to_string(DEFAULT_NUM_WORKERS)).c_str());

    /*********** SearchParams Binding ***********/
    class_<SearchParams, shared_ptr<SearchParams>>(m, "SearchParams")
        .def(init<>())
        .def_readwrite("k", &SearchParams::k,
             (std::string("Number of neighbors to return. default = ") + std::to_string(DEFAULT_K)).c_str())
        .def_readwrite("nprobe", &SearchParams::nprobe,
             (std::string("Number of partitions to probe. default = ") + std::to_string(DEFAULT_NPROBE)).c_str())
        .def_readwrite("recall_target", &SearchParams::recall_target,
             (std::string("Recall target. default = ") + std::to_string(DEFAULT_RECALL_TARGET)).c_str())
        .def_readwrite("batched_scan", &SearchParams::batched_scan,
             (std::string("Flag for batched scanning. default = ") + std::to_string(DEFAULT_BATCHED_SCAN)).c_str())
        .def_readwrite("use_precomputed", &SearchParams::use_precomputed,
             (std::string("Flag to use precomputed inc beta fn for APS. default = ") + std::to_string(DEFAULT_PRECOMPUTED)).c_str())
        .def_readwrite("initial_search_fraction", &SearchParams::initial_search_fraction,
             (std::string("Initial fraction of partitions to search. default = ") + std::to_string(DEFAULT_INITIAL_SEARCH_FRACTION)).c_str())
        .def_readwrite("recompute_threshold", &SearchParams::recompute_threshold,
             (std::string("Threshold to trigger recomputation of APS. default = ") + std::to_string(DEFAULT_RECOMPUTE_THRESHOLD)).c_str())
        .def_readwrite("aps_flush_period_us", &SearchParams::aps_flush_period_us,
             (std::string("APS flush period in microseconds. default = ") + std::to_string(DEFAULT_APS_FLUSH_PERIOD_US)).c_str());

    /*********** MaintenancePolicyParams Binding ***********/
    class_<MaintenancePolicyParams, shared_ptr<MaintenancePolicyParams>>(m, "MaintenancePolicyParams")
        .def(init<>())
        .def_readwrite("maintenance_policy", &MaintenancePolicyParams::maintenance_policy,
             (std::string("Maintenance policy type. default = ") + DEFAULT_MAINTENANCE_POLICY).c_str())
        .def_readwrite("window_size", &MaintenancePolicyParams::window_size,
             (std::string("Window size for measuring hit rates. default = ") + std::to_string(DEFAULT_WINDOW_SIZE)).c_str())
        .def_readwrite("refinement_radius", &MaintenancePolicyParams::refinement_radius,
             (std::string("Radius for local partition refinement. default = ") + std::to_string(DEFAULT_REFINEMENT_RADIUS)).c_str())
        .def_readwrite("refinement_iterations", &MaintenancePolicyParams::refinement_iterations,
             (std::string("Number of refinement iterations. default = ") + std::to_string(DEFAULT_REFINEMENT_ITERATIONS)).c_str())
        .def_readwrite("min_partition_size", &MaintenancePolicyParams::min_partition_size,
             (std::string("Minimum allowed partition size. default = ") + std::to_string(DEFAULT_MIN_PARTITION_SIZE)).c_str())
        .def_readwrite("alpha", &MaintenancePolicyParams::alpha,
             (std::string("Alpha parameter. default = ") + std::to_string(DEFAULT_ALPHA)).c_str())
        .def_readwrite("enable_split_rejection", &MaintenancePolicyParams::enable_split_rejection,
             (std::string("Enable split rejection. default = ") + std::to_string(DEFAULT_ENABLE_SPLIT_REJECTION)).c_str())
        .def_readwrite("enable_delete_rejection", &MaintenancePolicyParams::enable_delete_rejection,
             (std::string("Enable delete rejection. default = ") + std::to_string(DEFAULT_ENABLE_DELETE_REJECTION)).c_str())
        .def_readwrite("delete_threshold_ns", &MaintenancePolicyParams::delete_threshold_ns,
             (std::string("Delete threshold (ns). default = ") + std::to_string(DEFAULT_DELETE_THRESHOLD_NS)).c_str())
        .def_readwrite("split_threshold_ns", &MaintenancePolicyParams::split_threshold_ns,
             (std::string("Split threshold (ns). default = ") + std::to_string(DEFAULT_SPLIT_THRESHOLD_NS)).c_str())
        .def_readwrite("k_large", &MaintenancePolicyParams::k_large,
             (std::string("Large k value for maintenance. default = ") + std::to_string(DEFAULT_K_LARGE)).c_str())
        .def_readwrite("k_small", &MaintenancePolicyParams::k_small,
             (std::string("Small k value for maintenance. default = ") + std::to_string(DEFAULT_K_SMALL)).c_str())
        .def_readwrite("modify_centroids", &MaintenancePolicyParams::modify_centroids,
             (std::string("Flag to modify centroids during maintenance. default = ") + std::to_string(DEFAULT_MODIFY_CENTROIDS)).c_str())
        .def_readwrite("target_partition_size", &MaintenancePolicyParams::target_partition_size,
             (std::string("Target partition size. default = ") + std::to_string(DEFAULT_TARGET_PARTITION_SIZE)).c_str())
        .def_readwrite("max_partition_ratio", &MaintenancePolicyParams::max_partition_ratio,
             (std::string("Maximum allowed partition ratio. default = ") + std::to_string(DEFAULT_MAX_PARTITION_RATIO)).c_str());

     /*********** MaintenanceTimingInfo Binding ***********/
     class_<MaintenanceTimingInfo, shared_ptr<MaintenanceTimingInfo>>(m, "MaintenanceTimingInfo")
         .def_readonly("total_time_us", &MaintenanceTimingInfo::total_time_us,
             "Total time taken for maintenance in microseconds.")
     .def_readonly("split_time_us", &MaintenanceTimingInfo::split_time_us,
         "Time taken for split operations in microseconds.")
     .def_readonly("delete_time_us", &MaintenanceTimingInfo::delete_time_us,
         "Time taken for delete operations in microseconds.")
     .def_readonly("split_refine_time_us", &MaintenanceTimingInfo::split_refine_time_us,
         "Time taken for refinement of split operations in microseconds.")
     .def_readonly("delete_refine_time_us", &MaintenanceTimingInfo::delete_refine_time_us,
         "Time taken for refinement of delete operations in microseconds.")
     .def_readonly("n_splits", &MaintenanceTimingInfo::n_splits,
         "Number of partition split operations performed.")
     .def_readonly("n_deletes", &MaintenanceTimingInfo::n_deletes,
         "Number of partition delete operations performed.");

     /*********** ModifyTimingInfo Binding ***********/
     class_<ModifyTimingInfo, shared_ptr<ModifyTimingInfo>>(m, "ModifyTimingInfo")
         .def_readonly("modify_time_us", &ModifyTimingInfo::modify_time_us,
             "Total time taken for the modify operation in microseconds.")
         .def_readonly("modify_count", &ModifyTimingInfo::n_vectors)
     .def_readonly("find_partition_time_us", &ModifyTimingInfo::find_partition_time_us,
         "Time taken to find the partition for the modify operation in microseconds.");

     /*********** SearchTimingInfo Binding ***********/
     class_<SearchTimingInfo, shared_ptr<SearchTimingInfo>>(m, "SearchTimingInfo")
     .def_readonly("total_time_ns", &SearchTimingInfo::total_time_ns,
         "Total time taken for the search operation in nanoseconds.")
     .def_readonly("n_queries", &SearchTimingInfo::n_queries,
         "Number of queries performed.")
     .def_readonly("n_clusters", &SearchTimingInfo::n_clusters,
         "Number of clusters searched.")
     .def_readonly("partitions_scanned", &SearchTimingInfo::partitions_scanned,
         "Number of partitions scanned.")
     .def_readonly("search_params", &SearchTimingInfo::search_params,
         "Parameters used for the search operation.")
     .def_readonly("parent_info", &SearchTimingInfo::parent_info,
         "Search info for the parent index.");

    /**************** BuildTimingInfo Binding ***********/
    class_<BuildTimingInfo, shared_ptr<BuildTimingInfo>>(m, "BuildTimingInfo")
        .def_readonly("total_time_us", &BuildTimingInfo::total_time_us,
            "Total time taken for the build operation in microseconds.")
        .def_readonly("assign_time_us", &BuildTimingInfo::assign_time_us,
            "Time taken for assignment in microseconds.")
        .def_readonly("train_time_us", &BuildTimingInfo::train_time_us,
            "Time taken for training in microseconds.")
        .def_readonly("d", &BuildTimingInfo::d,
            "Dimension of the vectors.")
        .def_readonly("code_size", &BuildTimingInfo::code_size,
            "Size of PQ codes in bytes.")
        .def_readonly("n_codebooks", &BuildTimingInfo::num_codebooks,
            "Number of codebooks in the index.")
        .def_readonly("n_vectors", &BuildTimingInfo::n_vectors,
            "Number of vectors in the index.");

     /************* SearchResult Binding ***********/
     class_<SearchResult, shared_ptr<SearchResult>>(m, "SearchResult")
         .def_readonly("distances", &SearchResult::distances,
             "Distances to the nearest neighbors.")
         .def_readonly("ids", &SearchResult::ids,
             "Indices of the nearest neighbors.")
         .def_readonly("timing_info", &SearchResult::timing_info,
             "Timing information for the search operation.");
}

#endif //QUAKE_WRAP_H
