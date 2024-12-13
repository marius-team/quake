//
// Created by Jason on 9/25/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <dynamic_ivf.h>
#include <list_scanning.h>

std::tuple<int, float> get_static_nprobe_to_meet_recall_target(DynamicIVF_C& index, Tensor x, Tensor gt_ids, float recall_target) {
    // Ensure input tensor is contiguous
    x = x.contiguous();
    int num_queries = x.size(0);
    int k = gt_ids.size(1);

    // Initialize binary search for nprobe
    int min_nprobe = 1;
    int max_nprobe = index.nlist();

    float best_recall = 0;

    int best_nprobe = max_nprobe;
    while (min_nprobe <= max_nprobe) {
        int nprobe = (min_nprobe + max_nprobe) / 2;
        Tensor distances;
        Tensor labels;
        shared_ptr<SearchTimingInfo> timing_info;

        std::tie(labels, distances, timing_info) = index.search(x, nprobe, k);

        Tensor recalls = calculate_recall(labels, gt_ids);
        float recall = recalls.mean().item<float>();

        if (recall >= recall_target) {
            // Achieved desired recall; try to find smaller nprobe
            best_nprobe = nprobe;
            best_recall = recall;
            max_nprobe = nprobe - 1;
        } else {
            // Not enough recall; increase nprobe
            min_nprobe = nprobe + 1;
        }
    }
    // std::cout << "Best nprobe: " << best_nprobe << ", best recall: " << best_recall << std::endl;
    // Return the minimal nprobe that achieves the recall target
    return std::make_tuple(best_nprobe, best_recall);
}

std::tuple<Tensor, Tensor> get_per_query_nprobes_to_meet_recall_target(DynamicIVF_C& index, Tensor x, Tensor gt_ids, float recall_target) {
    // Ensure input tensor is contiguous
    x = x.contiguous();
    int num_queries = x.size(0);

    vector<int> nprobes_list;
    vector<float> recalls_list;
    for (int i = 0; i < num_queries; i++) {
        Tensor query = x[i].unsqueeze(0);
        Tensor gt_ids_i = gt_ids[i].unsqueeze(0);
        int best_nprobe;
        float best_recall;
        std::tie(best_nprobe, best_recall) = get_static_nprobe_to_meet_recall_target(index, query, gt_ids_i, recall_target);
        nprobes_list.push_back(best_nprobe);
        recalls_list.push_back(best_recall);
    }

    return std::make_tuple(torch::tensor(nprobes_list, torch::kInt32).clone().to(torch::kFloat32), torch::tensor(recalls_list).clone());
}


void measure_recall_estimation_overhead() {
    int n_vectors = 100000;
    int n_list = 1000;
    int n_queries = 100;
    int k = 10;
    int d = 32;
    float recall_target = 0.9;
    Tensor vectors = torch::rand({n_vectors, d});
    Tensor ids = torch::arange(n_vectors, torch::kInt64);
    Tensor queries = torch::rand({n_queries, d});

    // Create flat index
    shared_ptr<DynamicIVF_C> flat_index = std::make_shared<DynamicIVF_C>(d, 1, 1);
    flat_index->build(vectors, ids);

    // Compute ground truth
    Tensor gt_ids;
    Tensor gt_distances;
    shared_ptr<SearchTimingInfo> timing_info;
    std::tie(gt_ids, gt_distances, timing_info) = flat_index->search(queries, 1, k);

    std::cout << "Computing ground truth took " << timing_info->total_time_us << " us" << std::endl;

    // Create IVF index
    shared_ptr<DynamicIVF_C> ivf_index = std::make_shared<DynamicIVF_C>(d, n_list, 1);
    ivf_index->build(vectors, ids);

    // Get nprobe to meet recall target for whole dataset
    int best_static_nprobe;
    int best_static_recall;
    std::tie(best_static_nprobe, best_static_recall) = get_static_nprobe_to_meet_recall_target(*ivf_index.get(), queries, gt_ids, recall_target);
    // Run each query
    vector<Tensor> baseline_ids_list;
    vector<Tensor> baseline_distances_list;
    vector<shared_ptr<SearchTimingInfo>> baseline_timing_info_list;

    // run queries one-by-one
    for (int i = 0; i < n_queries; i++) {
        Tensor query = queries[i].unsqueeze(0);
        Tensor curr_ids;
        Tensor curr_distances;
        shared_ptr<SearchTimingInfo> curr_timing_info;

        std::tie(curr_ids, curr_distances, curr_timing_info) = ivf_index->search(query, best_static_nprobe, k);

        baseline_ids_list.push_back(curr_ids);
        baseline_distances_list.push_back(curr_distances);
        baseline_timing_info_list.push_back(curr_timing_info);
    }
    Tensor baseline_ids = torch::cat(baseline_ids_list);
    Tensor baseline_distances = torch::cat(baseline_distances_list);

    // compute recall of the baseline
    Tensor recalls = calculate_recall(baseline_ids, gt_ids);
    std::cout << "Recall of the baseline: " << recalls.mean().item<float>() << std::endl;
    std::cout << "Average nprobe: " << best_static_nprobe << std::endl;

    // run queries all at once
    std::tie(baseline_ids, baseline_distances, timing_info) = ivf_index->search(queries, best_static_nprobe, k);

    Tensor baseline_recalls = calculate_recall(baseline_ids, gt_ids);
    std::cout << "Recall of the baseline (all queries at once): " << baseline_recalls.mean().item<float>() << std::endl;

    // Get per-query nprobe to meet recall target
    Tensor per_query_nprobes;
    Tensor per_query_recalls;
    std::tie(per_query_nprobes, per_query_recalls) = get_per_query_nprobes_to_meet_recall_target(*ivf_index.get(), queries, gt_ids, recall_target);
    std::cout << "Average recall of the per-query nprobe: " << per_query_recalls.mean().item<float>() << std::endl;
    std::cout << "Average nprobe: " << per_query_nprobes.mean().item<float>() << std::endl;

    // Run queries using search_one with a recall target
    vector<Tensor> search_one_ids_list;
    vector<Tensor> search_one_distances_list;
    vector<shared_ptr<SearchTimingInfo>> search_one_timing_info_list;
    for (int i = 0; i < n_queries; i++) {
        Tensor query = queries[i].unsqueeze(0);
        Tensor curr_ids;
        Tensor curr_distances;
        shared_ptr<SearchTimingInfo> curr_timing_info;

        std::tie(curr_ids, curr_distances, curr_timing_info) = ivf_index->search_one(query, k, recall_target);

        search_one_ids_list.push_back(curr_ids.unsqueeze(0));
        search_one_distances_list.push_back(curr_distances.unsqueeze(0));
        search_one_timing_info_list.push_back(curr_timing_info);
    }

    Tensor search_one_ids = torch::cat(search_one_ids_list);
    Tensor search_one_distances = torch::cat(search_one_distances_list);
    Tensor search_one_recalls = calculate_recall(search_one_ids, gt_ids);
    std::cout << "Recall of search_one: " << search_one_recalls.mean().item<float>() << std::endl;
}

int main() {
    torch::manual_seed(0);
    measure_recall_estimation_overhead();
    return 0;
}