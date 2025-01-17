//
// Created by Jason on 7/25/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <quake_index.h>
#ifndef COMPASS_WRAP_H
#define COMPASS_WRAP_H

#include "common.h"
#include "pybind11/embed.h"
#include "torch/extension.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

using py::arg;
using py::class_;
using py::enum_;
using py::init;
using py::module;

using std::shared_ptr;

using faiss::idx_t;

PYBIND11_MODULE(_bindings, m) {
    // wrap Quake Index
    class_<QuakeIndex, shared_ptr<QuakeIndex> >(m, "QuakeIndex")
            .def(init<int>(), py::arg("current_level") = 0)
            .def("build", &QuakeIndex::build)
            .def("search", &QuakeIndex::search)
            .def("get", &QuakeIndex::get)
            .def("add", &QuakeIndex::add)
            .def("remove", &QuakeIndex::remove)
            .def("save", &QuakeIndex::save)
            .def("load", &QuakeIndex::load)
            .def("ntotal", &QuakeIndex::ntotal)
            .def("nlist", &QuakeIndex::nlist);

    class_<MaintenanceTimingInfo, shared_ptr<MaintenanceTimingInfo> >(m, "MaintenanceTimingInfo")
            .def_readonly("n_splits", &MaintenanceTimingInfo::n_splits)
            .def_readonly("n_deletes", &MaintenanceTimingInfo::n_deletes)
            .def_readonly("delete_time_us", &MaintenanceTimingInfo::delete_time_us)
            .def_readonly("delete_refine_time_us", &MaintenanceTimingInfo::delete_refine_time_us)
            .def_readonly("split_time_us", &MaintenanceTimingInfo::split_time_us)
            .def_readonly("split_refine_time_us", &MaintenanceTimingInfo::split_refine_time_us)
            .def_readonly("total_time_us", &MaintenanceTimingInfo::total_time_us);

    class_<BuildTimingInfo, shared_ptr<BuildTimingInfo> >(m, "BuildTimingInfo")
            .def_readonly("n_vectors", &BuildTimingInfo::n_vectors)
            .def_readonly("n_clusters", &BuildTimingInfo::n_clusters)
            .def_readonly("d", &BuildTimingInfo::d)
            .def_readonly("num_codebooks", &BuildTimingInfo::num_codebooks)
            .def_readonly("code_size", &BuildTimingInfo::code_size)
            .def_readonly("train_time_us", &BuildTimingInfo::train_time_us)
            .def_readonly("assign_time_us", &BuildTimingInfo::assign_time_us)
            .def_readonly("total_time_us", &BuildTimingInfo::total_time_us);

    class_<ModifyTimingInfo, shared_ptr<ModifyTimingInfo> >(m, "ModifyTimingInfo")
            .def_readonly("n_vectors", &ModifyTimingInfo::n_vectors)
            .def_readonly("find_partition_time_us", &ModifyTimingInfo::find_partition_time_us)
            .def_readonly("modify_time_us", &ModifyTimingInfo::modify_time_us)
            .def_readonly("maintenance_time_us", &ModifyTimingInfo::maintenance_time_us);

    class_<SearchTimingInfo, shared_ptr<SearchTimingInfo> >(m, "SearchTimingInfo")
            .def(init<>())
            .def_readwrite("n_queries", &SearchTimingInfo::n_queries)
            .def_readonly("n_clusters", &SearchTimingInfo::n_clusters)
            .def_readonly("partitions_scanned", &SearchTimingInfo::partitions_scanned)
            .def_readonly("search_params", &SearchTimingInfo::search_params)
            .def_readonly("parent_info", &SearchTimingInfo::parent_info)
            .def_readonly("total_time_ns", &SearchTimingInfo::total_time_ns)
            .def_readonly("buffer_init_time_ns", &SearchTimingInfo::buffer_init_time_ns)
            .def_readonly("job_enqueue_time_ns", &SearchTimingInfo::job_enqueue_time_ns)
            .def_readonly("boundary_distance_time_ns", &SearchTimingInfo::boundary_distance_time_ns)
            .def_readonly("job_wait_time_ns", &SearchTimingInfo::job_wait_time_ns)
            .def_readonly("result_aggregate_time_ns", &SearchTimingInfo::result_aggregate_time_ns);

    class_<MaintenancePolicyParams, shared_ptr<MaintenancePolicyParams> >(m, "MaintenancePolicyParams")
            .def(init<>())
            .def_readwrite("maintenance_policy", &MaintenancePolicyParams::maintenance_policy)
            .def_readwrite("window_size", &MaintenancePolicyParams::window_size)
            .def_readwrite("refinement_radius", &MaintenancePolicyParams::refinement_radius)
            .def_readwrite("refinement_iterations", &MaintenancePolicyParams::refinement_iterations)
            .def_readwrite("min_partition_size", &MaintenancePolicyParams::min_partition_size)
            .def_readwrite("alpha", &MaintenancePolicyParams::alpha)
            .def_readwrite("enable_split_rejection", &MaintenancePolicyParams::enable_split_rejection)
            .def_readwrite("enable_delete_rejection", &MaintenancePolicyParams::enable_delete_rejection)
            .def_readwrite("delete_threshold_ns", &MaintenancePolicyParams::delete_threshold_ns)
            .def_readwrite("split_threshold_ns", &MaintenancePolicyParams::split_threshold_ns)
            .def_readwrite("k_large", &MaintenancePolicyParams::k_large)
            .def_readwrite("k_small", &MaintenancePolicyParams::k_small)
            .def_readwrite("modify_centroids", &MaintenancePolicyParams::modify_centroids)
            .def_readwrite("target_partition_size", &MaintenancePolicyParams::target_partition_size)
            .def_readwrite("max_partition_ratio", &MaintenancePolicyParams::max_partition_ratio);

    class_<IndexBuildParams, shared_ptr<IndexBuildParams> >(m, "IndexBuildParams")
            .def(init<>())
            .def_readwrite("nlist", &IndexBuildParams::nlist)
            .def_readwrite("metric", &IndexBuildParams::metric)
            .def_readwrite("niter", &IndexBuildParams::niter)
            .def_readwrite("num_codebooks", &IndexBuildParams::num_codebooks)
            .def_readwrite("code_size", &IndexBuildParams::code_size)
            .def_readwrite("num_workers", &IndexBuildParams::num_workers)
            .def_readwrite("use_numa", &IndexBuildParams::use_numa)
            .def_readwrite("verbose", &IndexBuildParams::verbose)
            .def_readwrite("verify_numa", &IndexBuildParams::verify_numa)
            .def_readwrite("same_core", &IndexBuildParams::same_core)
            .def_readwrite("use_adaptive_nprobe", &IndexBuildParams::use_adaptive_nprobe);

    class_<SearchParams, shared_ptr<SearchParams> >(m, "SearchParams")
            .def(init<>())
            .def_readwrite("k", &SearchParams::k)
            .def_readwrite("nprobe", &SearchParams::nprobe)
            .def_readwrite("recall_target", &SearchParams::recall_target)
            .def_readwrite("batched_scan", &SearchParams::batched_scan)
            .def_readwrite("initial_search_fraction", &SearchParams::initial_search_fraction)
            .def_readwrite("recompute_threshold", &SearchParams::recompute_threshold)
            .def_readwrite("use_precomputed", &SearchParams::use_precomputed);

    class_<SearchResult, shared_ptr<SearchResult> >(m, "SearchResult")
            .def_readonly("ids", &SearchResult::ids)
            .def_readonly("distances", &SearchResult::distances)
            .def_readonly("timing_info", &SearchResult::timing_info);
}


#endif //COMPASS_WRAP_H
