//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef QUAKE_INDEX_H
#define QUAKE_INDEX_H

#include <maintenance_policies.h>
#include <dynamic_inverted_list.h>
#include <partition_manager.h>
#include <query_coordinator.h>


class QuakeIndex {
public:
    shared_ptr<QuakeIndex> parent_;
    shared_ptr<PartitionManager> partition_manager_;
    shared_ptr<QueryCoordinator> query_coordinator_;
    shared_ptr<MaintenancePolicy> maintenance_policy_;

    MetricType metric_;
    shared_ptr<IndexBuildParams> build_params_;
    shared_ptr<MaintenancePolicyParams> maintenance_policy_params_;
    int current_level_ = 0;

    QuakeIndex(int current_level = 0);

    ~QuakeIndex();

    shared_ptr<BuildTimingInfo> build(Tensor x, Tensor ids, shared_ptr<IndexBuildParams> build_params);

    shared_ptr<SearchResult> search(Tensor x, shared_ptr<SearchParams> search_params);

    Tensor get(Tensor ids);

    shared_ptr<ModifyTimingInfo> add(Tensor x, Tensor ids);

    shared_ptr<ModifyTimingInfo> remove(Tensor ids);

    void initialize_maintenance_policy(shared_ptr<MaintenancePolicyParams> maintenance_policy_params);

    shared_ptr<MaintenanceTimingInfo> maintenance();

    void save(const std::string &path);

    void load(const std::string &path);

    int64_t ntotal();

    int64_t nlist();

    int d();
};

#endif //QUAKE_INDEX_H
