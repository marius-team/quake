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
#include <sstream>  // For JSON-style string formatting

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
             "The current level of the index.")
        // __repr__ produces a JSON-style string summary.
        .def("__repr__", [](const QuakeIndex &q) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"current_level\": " << q.current_level_ << ", ";
            oss << "}";
            return oss.str();
        });

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
             (std::string("Number of workers. default = ") + std::to_string(DEFAULT_NUM_WORKERS)).c_str())
        .def("__repr__", [](const IndexBuildParams &p) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"nlist\": " << p.nlist << ", ";
            oss << "\"niter\": " << p.niter << ", ";
            oss << "\"metric\": \"" << p.metric << "\", ";
            oss << "\"num_workers\": " << p.num_workers;
            oss << "}";
            return oss.str();
        });

    /*********** SearchParams Binding ***********/
    class_<SearchParams, shared_ptr<SearchParams>>(m, "SearchParams")
        .def(init<>())
        .def_readwrite("k", &SearchParams::k,
             (std::string("Number of neighbors to return. default = ") + std::to_string(DEFAULT_K)).c_str())
        .def_readwrite("nprobe", &SearchParams::nprobe,
             (std::string("Number of partitions to probe. default = ") + std::to_string(DEFAULT_NPROBE)).c_str())
        .def_readwrite("recall_target", &SearchParams::recall_target,
             (std::string("Recall target. default = ") + std::to_string(DEFAULT_RECALL_TARGET)).c_str())
        .def_readwrite("num_threads", &SearchParams::num_threads,
             "Number of threads to use for search within a single worker.")
        .def_readwrite("batched_scan", &SearchParams::batched_scan,
             (std::string("Flag for batched scanning. default = ") + std::to_string(DEFAULT_BATCHED_SCAN)).c_str())
        .def_readwrite("use_precomputed", &SearchParams::use_precomputed,
             (std::string("Flag to use precomputed inc beta fn for APS. default = ") + std::to_string(DEFAULT_PRECOMPUTED)).c_str())
        .def_readwrite("initial_search_fraction", &SearchParams::initial_search_fraction,
             (std::string("Initial fraction of partitions to search. default = ") + std::to_string(DEFAULT_INITIAL_SEARCH_FRACTION)).c_str())
        .def_readwrite("recompute_threshold", &SearchParams::recompute_threshold,
             (std::string("Threshold to trigger recomputation of APS. default = ") + std::to_string(DEFAULT_RECOMPUTE_THRESHOLD)).c_str())
        .def_readwrite("aps_flush_period_us", &SearchParams::aps_flush_period_us,
             (std::string("APS flush period in microseconds. default = ") + std::to_string(DEFAULT_APS_FLUSH_PERIOD_US)).c_str())
        .def("__repr__", [](const SearchParams &s) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"k\": " << s.k << ", ";
            oss << "\"nprobe\": " << s.nprobe << ", ";
            oss << "\"recall_target\": " << s.recall_target << ", ";
            oss << "\"batched_scan\": " << (s.batched_scan ? "true" : "false") << ", ";
            oss << "\"use_precomputed\": " << (s.use_precomputed ? "true" : "false") << ", ";
            oss << "\"initial_search_fraction\": " << s.initial_search_fraction << ", ";
            oss << "\"recompute_threshold\": " << s.recompute_threshold << ", ";
            oss << "\"aps_flush_period_us\": " << s.aps_flush_period_us;
            oss << "}";
            return oss.str();
        });

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
        .def("__repr__", [](const MaintenancePolicyParams &m) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"maintenance_policy\": \"" << m.maintenance_policy << "\", ";
            oss << "\"window_size\": " << m.window_size << ", ";
            oss << "\"refinement_radius\": " << m.refinement_radius << ", ";
            oss << "\"refinement_iterations\": " << m.refinement_iterations << ", ";
            oss << "\"min_partition_size\": " << m.min_partition_size << ", ";
            oss << "\"alpha\": " << m.alpha << ", ";
            oss << "\"enable_split_rejection\": " << (m.enable_split_rejection ? "true" : "false") << ", ";
            oss << "\"enable_delete_rejection\": " << (m.enable_delete_rejection ? "true" : "false") << ", ";
            oss << "\"delete_threshold_ns\": " << m.delete_threshold_ns << ", ";
            oss << "\"split_threshold_ns\": " << m.split_threshold_ns << ", ";
            oss << "}";
            return oss.str();
        });

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
             "Number of partition delete operations performed.")
         .def("__repr__", [](const MaintenanceTimingInfo &t) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_us\": " << t.total_time_us << ", ";
             oss << "\"split_time_us\": " << t.split_time_us << ", ";
             oss << "\"delete_time_us\": " << t.delete_time_us << ", ";
             oss << "\"split_refine_time_us\": " << t.split_refine_time_us << ", ";
             oss << "\"delete_refine_time_us\": " << t.delete_refine_time_us << ", ";
             oss << "\"n_splits\": " << t.n_splits << ", ";
             oss << "\"n_deletes\": " << t.n_deletes;
             oss << "}";
             return oss.str();
         });

    /*********** ModifyTimingInfo Binding ***********/
    class_<ModifyTimingInfo, shared_ptr<ModifyTimingInfo>>(m, "ModifyTimingInfo")
         .def_readonly("modify_time_us", &ModifyTimingInfo::modify_time_us,
             "Total time taken for the modify operation in microseconds.")
        .def_readonly("input_validation_time_us", &ModifyTimingInfo::input_validation_time_us,
             "Time taken for validation in microseconds.")
         .def_readonly("modify_count", &ModifyTimingInfo::n_vectors)
         .def_readonly("find_partition_time_us", &ModifyTimingInfo::find_partition_time_us,
             "Time taken to find the partition for the modify operation in microseconds.")
         .def("__repr__", [](const ModifyTimingInfo &m) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"modify_count\": " << m.n_vectors << ", ";
             oss << "\"input_validation_time_us\": " << m.input_validation_time_us << ", ";
             oss << "\"modify_time_us\": " << m.modify_time_us << ", ";
             oss << "\"find_partition_time_us\": " << m.find_partition_time_us;
             oss << "}";
             return oss.str();
         });

    /*********** SearchTimingInfo Binding ***********/
    class_<SearchTimingInfo, shared_ptr<SearchTimingInfo>>(m, "SearchTimingInfo")
         .def(init<>())
         .def_readwrite("total_time_ns", &SearchTimingInfo::total_time_ns,
             "Total time taken for the search operation in nanoseconds.")
        .def_readwrite("buffer_init_time_ns", &SearchTimingInfo::buffer_init_time_ns,
             "Time spent on initializing buffers in nanoseconds.")
        .def_readwrite("job_enqueue_time_ns", &SearchTimingInfo::job_enqueue_time_ns,
             "Time spent on creating jobs in nanoseconds.")
        .def_readwrite("boundary_distance_time_ns", &SearchTimingInfo::boundary_distance_time_ns,
             "Time spent on computing boundary distances in nanoseconds.")
        .def_readwrite("job_wait_time_ns", &SearchTimingInfo::job_wait_time_ns,
             "Time spent waiting for jobs to complete in nanoseconds.")
        .def_readwrite("result_aggregate_time_ns", &SearchTimingInfo::result_aggregate_time_ns,
             "Time spent on aggregating results in nanoseconds.")
         .def_readwrite("n_queries", &SearchTimingInfo::n_queries,
             "Number of queries performed.")
         .def_readwrite("n_clusters", &SearchTimingInfo::n_clusters,
             "Number of clusters searched.")
         .def_readwrite("partitions_scanned", &SearchTimingInfo::partitions_scanned,
             "Number of partitions scanned.")
         .def_readwrite("search_params", &SearchTimingInfo::search_params,
             "Parameters used for the search operation.")
         .def_readwrite("parent_info", &SearchTimingInfo::parent_info,
             "Search info for the parent index.")
         .def("__repr__", [](const SearchTimingInfo &s) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_ns\": " << s.total_time_ns << ", ";
             oss << "\"buffer_init_time_ns\": " << s.buffer_init_time_ns << ", ";
             oss << "\"job_enqueue_time_ns\": " << s.job_enqueue_time_ns << ", ";
             oss << "\"boundary_distance_time_ns\": " << s.boundary_distance_time_ns << ", ";
             oss << "\"job_wait_time_ns\": " << s.job_wait_time_ns << ", ";
             oss << "\"result_aggregate_time_ns\": " << s.result_aggregate_time_ns << ", ";
             if (s.parent_info != nullptr) {
                 oss << "\"parent_scan_time_ns\": " << s.parent_info->total_time_ns << ", ";
             }
             oss << "\"n_queries\": " << s.n_queries << ", ";
             oss << "\"n_clusters\": " << s.n_clusters << ", ";
             oss << "\"partitions_scanned\": " << s.partitions_scanned;
             oss << "}";
             return oss.str();
         });

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
            "Number of vectors in the index.")
         .def("__repr__", [](const BuildTimingInfo &b) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"total_time_us\": " << b.total_time_us << ", ";
             oss << "\"assign_time_us\": " << b.assign_time_us << ", ";
             oss << "\"train_time_us\": " << b.train_time_us << ", ";
             oss << "\"d\": " << b.d << ", ";
             oss << "\"code_size\": " << b.code_size << ", ";
             oss << "\"n_codebooks\": " << b.num_codebooks << ", ";
             oss << "\"n_vectors\": " << b.n_vectors;
             oss << "}";
             return oss.str();
         });

    /************* SearchResult Binding ***********/
    class_<SearchResult, shared_ptr<SearchResult>>(m, "SearchResult")
         .def(init<>())
         .def_readwrite("distances", &SearchResult::distances,
             "Distances to the nearest neighbors.")
         .def_readwrite("ids", &SearchResult::ids,
             "Indices of the nearest neighbors.")
         .def_readwrite("timing_info", &SearchResult::timing_info,
             "Timing information for the search operation.")
         .def("__repr__", [](const SearchResult &r) {
             std::ostringstream oss;
             oss << "{";
             oss << "\"num_ids\": " << r.ids.numel() << ", ";
             oss << "\"num_distances\": " << r.distances.numel();
             oss << "}";
             return oss.str();
         });
}

#endif //QUAKE_WRAP_H