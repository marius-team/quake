//
// Created by Jason on 7/25/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef COMPASS_WRAP_H
#define COMPASS_WRAP_H

#include "pybind11/embed.h"
#include "torch/extension.h"
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <dynamic_ivf.h>

namespace py = pybind11;

using py::arg;
using py::class_;
using py::enum_;
using py::init;
using py::module;

using std::shared_ptr;

using faiss::idx_t;


// struct SearchTimingInfo {
//     int64_t n_queries;
//     int64_t n_vectors;
//     int64_t n_clusters;
//     int d;
//     int num_codebooks;
//     int code_size;
//     int k;
//     int nprobe;
//     float k_factor;
//     float recall_target;
//
//     int quantizer_search_time_us;
//     int scan_pq_time_us;
//     int refine_time_us;
//     int partition_scan_time_us;
//     int total_time_us;
//
//     shared_ptr<SearchTimingInfo> parent_info = nullptr;
// }

PYBIND11_MODULE(_bindings, m) {
    m.def("merge_faiss_ivf", &merge_faiss_ivf, "Merge two faiss IndexIVF objects");
    // m.def("measure_list_scan_cost", &measure_list_scan_cost, "Measure the cost of scanning a list");

    // wrap faiss IVF index
    class_<DynamicIVF_C, shared_ptr<DynamicIVF_C> >(m, "DynamicIVF_C")
            .def(init<int, int, int, int, int, int, bool, bool, bool, bool, bool, bool>(), arg("d"), arg("nlist"),
                 arg("metric"), arg("num_workers") = 1, arg("m") = -1, arg("code_size") = -1, arg("use_numa") = false,
                 arg("verbose") = false, arg("verify_numa") = false, arg("same_core") = true, arg("use_centroid_workers") = true,
                 arg("use_adaptive_nprobe") = false)
            .def("nlist", &DynamicIVF_C::nlist)
            .def("ntotal", &DynamicIVF_C::ntotal)
            .def_readonly("d", &DynamicIVF_C::d_)
            .def("centroids", &DynamicIVF_C::centroids)
            .def_readonly("parent", &DynamicIVF_C::parent_)
            .def("index_ready", &DynamicIVF_C::index_ready)
            .def("reset_workers", &DynamicIVF_C::reset_workers)
            .def("set_timeout_values", &DynamicIVF_C::set_timeout_values)
            .def("get_partition_ids", &DynamicIVF_C::get_partition_ids)
            .def("get_scan_fraction", &DynamicIVF_C::get_scan_fraction)
            .def("build_given_centroids", &DynamicIVF_C::build_given_centroids)
            .def("add_level", &DynamicIVF_C::add_level)
            .def("remove_level", &DynamicIVF_C::remove_level)
            .def("build", &DynamicIVF_C::build)
            .def("rebuild", &DynamicIVF_C::rebuild)
            .def("save", &DynamicIVF_C::save)
            .def("load", &DynamicIVF_C::load)
            .def("add", &DynamicIVF_C::add)
            .def("add_centroids_and_reassign_existing", &DynamicIVF_C::add_centroids_and_reassign_existing)
            .def("remove", &DynamicIVF_C::remove)
            .def("modify", &DynamicIVF_C::modify)
            .def("maintenance", &DynamicIVF_C::maintenance)
            .def("set_maintenance_policy_params", &DynamicIVF_C::set_maintenance_policy_params)
            .def("search", &DynamicIVF_C::search)
            .def("search_one", &DynamicIVF_C::search_one)
            .def("get_cluster_sizes", &DynamicIVF_C::get_cluster_sizes)
            .def("get_cluster_ids", &DynamicIVF_C::get_cluster_ids)
            .def("recompute_centroids", &DynamicIVF_C::recompute_centroids)
            .def("select_clusters", &DynamicIVF_C::select_clusters)
            .def("select_vectors", &DynamicIVF_C::select_vectors)
            .def("search_quantizer", &DynamicIVF_C::search_quantizer)
            .def("refine_clusters", &DynamicIVF_C::refine_clusters)
            .def("compute_partition_boundary_distances", &DynamicIVF_C::compute_partition_boundary_distances)
            .def("compute_kth_nearest_neighbor_distance", &DynamicIVF_C::compute_kth_nearest_neighbor_distance)
            .def("compute_partition_probabilities", &DynamicIVF_C::compute_partition_probabilities)
            .def("compute_partition_intersection_volumes", &DynamicIVF_C::compute_partition_intersection_volumes)
            .def("compute_partition_distances", &DynamicIVF_C::compute_partition_distances)
            .def("compute_partition_density", &DynamicIVF_C::compute_partition_density)
            .def("compute_partition_volume", &DynamicIVF_C::compute_partition_volume)
            .def("compute_partition_variances", &DynamicIVF_C::compute_partition_variances)
            .def("get_partition_sizes", &DynamicIVF_C::get_partition_sizes)
            .def("compute_quantization_error", &DynamicIVF_C::compute_quantization_error)
            .def("compute_partition_covariances", &DynamicIVF_C::compute_partition_covariances)
            .def("selective_merge", &DynamicIVF_C::selective_merge)
            .def("get_split_history", &DynamicIVF_C::get_split_history);

    class_<MaintenanceTimingInfo, shared_ptr<MaintenanceTimingInfo> >(m, "MaintenanceTimingInfo")
            .def_readonly("n_splits", &MaintenanceTimingInfo::n_splits)
            .def_readonly("n_deletes", &MaintenanceTimingInfo::n_deletes)
            .def_readonly("delete_time_us", &MaintenanceTimingInfo::delete_time_us)
            .def_readonly("delete_refine_time_us", &MaintenanceTimingInfo::delete_refine_time_us)
            .def_readonly("split_time_us", &MaintenanceTimingInfo::split_time_us)
            .def_readonly("split_refine_time_us", &MaintenanceTimingInfo::split_refine_time_us)
            .def_readonly("total_time_us", &MaintenanceTimingInfo::total_time_us)
            .def("print", &MaintenanceTimingInfo::print);

    class_<BuildTimingInfo, shared_ptr<BuildTimingInfo> >(m, "BuildTimingInfo")
            .def_readonly("n_vectors", &BuildTimingInfo::n_vectors)
            .def_readonly("n_clusters", &BuildTimingInfo::n_clusters)
            .def_readonly("d", &BuildTimingInfo::d)
            .def_readonly("num_codebooks", &BuildTimingInfo::num_codebooks)
            .def_readonly("code_size", &BuildTimingInfo::code_size)
            .def_readonly("train_time_us", &BuildTimingInfo::train_time_us)
            .def_readonly("assign_time_us", &BuildTimingInfo::assign_time_us)
            .def_readonly("total_time_us", &BuildTimingInfo::total_time_us)
            .def("print", &BuildTimingInfo::print);

    class_<ModifyTimingInfo, shared_ptr<ModifyTimingInfo> >(m, "ModifyTimingInfo")
            .def_readonly("n_vectors", &ModifyTimingInfo::n_vectors)
            .def_readonly("find_partition_time_us", &ModifyTimingInfo::find_partition_time_us)
            .def_readonly("modify_time_us", &ModifyTimingInfo::modify_time_us)
            .def_readonly("maintenance_time_us", &ModifyTimingInfo::maintenance_time_us)
            .def("print", &ModifyTimingInfo::print);

    class_<SearchTimingInfo, shared_ptr<SearchTimingInfo> >(m, "SearchTimingInfo")
            .def(init<>())
            .def_readwrite("n_queries", &SearchTimingInfo::n_queries)
            .def_readonly("n_vectors", &SearchTimingInfo::n_vectors)
            .def_readonly("n_clusters", &SearchTimingInfo::n_clusters)
            .def_readonly("d", &SearchTimingInfo::d)
            .def_readonly("num_codebooks", &SearchTimingInfo::num_codebooks)
            .def_readonly("code_size", &SearchTimingInfo::code_size)
            .def_readonly("k", &SearchTimingInfo::k)
            .def_readwrite("nprobe", &SearchTimingInfo::nprobe)
            .def_readonly("k_factor", &SearchTimingInfo::k_factor)
            .def_readonly("recall_target", &SearchTimingInfo::recall_target)
            .def_readwrite("quantizer_search_time_us", &SearchTimingInfo::quantizer_search_time_us)
            .def_readwrite("scan_pq_time_us", &SearchTimingInfo::scan_pq_time_us)
            .def_readwrite("refine_time_us", &SearchTimingInfo::refine_time_us)
            .def_readwrite("metadata_update_time_us", &SearchTimingInfo::metadata_update_time_us)
            .def_readwrite("partition_scan_time_us", &SearchTimingInfo::partition_scan_time_us)
            .def_readwrite("total_time_us", &SearchTimingInfo::total_time_us)
            .def_readwrite("partition_scan_setup_time_us", &SearchTimingInfo::partition_scan_setup_time_us)
            .def_readwrite("partition_scan_search_time_us", &SearchTimingInfo::partition_scan_search_time_us)
            .def_readwrite("partition_scan_post_process_time_us",
                           &SearchTimingInfo::partition_scan_post_process_time_us)
            .def_readwrite("average_worker_job_time_us", &SearchTimingInfo::average_worker_job_time_us)
            .def_readwrite("average_worker_scan_time_us", &SearchTimingInfo::average_worker_scan_time_us)
            .def_readwrite("target_vectors_scanned", &SearchTimingInfo::target_vectors_scanned)
            .def_readwrite("total_vectors_scanned", &SearchTimingInfo::total_vectors_scanned)
            .def_readwrite("average_worker_throughput", &SearchTimingInfo::average_worker_throughput)
            .def_readwrite("parent_info", &SearchTimingInfo::parent_info)
            .def_readwrite("total_numa_preprocessing_time_us", &SearchTimingInfo::total_numa_preprocessing_time_us)
            .def_readwrite("total_job_distribute_time_us", &SearchTimingInfo::total_job_distribute_time_us)
            .def_readwrite("total_result_wait_time_us", &SearchTimingInfo::total_result_wait_time_us)
            .def_readwrite("total_numa_postprocessing_time_us", &SearchTimingInfo::total_numa_postprocessing_time_us)
            .def_readwrite("using_faiss_index", &SearchTimingInfo::using_faiss_index)
            .def_readwrite("using_numa", &SearchTimingInfo::using_numa)
            .def_readwrite("recall_profile_us", &SearchTimingInfo::recall_profile_us)
            .def_readwrite("boundary_time_us", &SearchTimingInfo::boundary_time_us)
            .def("print", &SearchTimingInfo::print, arg("indent") = 0)
            .def("get_scan_secs", &SearchTimingInfo::get_scan_secs)
            .def("get_scan_bytes", &SearchTimingInfo::get_scan_bytes)
            .def("get_overall_bytes", &SearchTimingInfo::get_overall_bytes)
            .def("get_scan_throughput", &SearchTimingInfo::get_scan_throughput);

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
}


#endif //COMPASS_WRAP_H
