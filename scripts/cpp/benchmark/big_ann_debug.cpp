#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <string>
#include <string_view>
#include <vector>
#include <sstream> 
#include <stdexcept> 
#include <map>     
#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm> 
#include <numeric>
#include <iterator>
#include <utility>
#include <random>
#include <chrono>

#include <torch/script.h>    // For c10::IValue
#include <torch/serialize.h> // For torch::load

#include "quake_index.h"  // Quake API header
#include "partition_manager.h"
#include "dynamic_inverted_list.h"
#include "index_partition.h"
#include "parallel.h"

using torch::Tensor;
using std::vector;
using std::shared_ptr;

#define DATASET_PATH "/home/devesh/big-ann-benchmark/big-ann-benchmarks/data/MSTuring-30M-clustered/"
#define DIRECTORY_PATH DATASET_PATH "index_arguments/"
#define GROUND_TRUTH_PATH DATASET_PATH "29998994/final_runbook.yaml/"

constexpr float DELETE_THRESHOLD = 0.7;
constexpr float CAPACITY_THRESHOLD = 1.15;

constexpr float MS_TO_US = 1000.0;
constexpr float MS_TO_NS = 1000.0 * 1000.0;

struct Step {
    uint32_t step_number;
    std::string type;
    std::string vectors_path;
    std::string ids_path;

    // Default constructor
    Step() : step_number(0), type("UNKNOWN") {}
};

bool parse_filename(const std::string& filename,
                    uint32_t& out_step_num,
                    std::string& out_type, // Changed to std::string&
                    std::string& out_target)
{
    // 1. Check prefix and suffix
    if (filename.find("step_") != 0 || 
        filename.rfind(".pth") != (filename.length() - 4)) {
        return false;
    }

    // 2. Extract the core "1_insert_ids"
    std::string core = filename.substr(5, filename.length() - 5 - 4);

    // 3. Split the core by underscores
    std::vector<std::string> parts;
    std::stringstream ss(core);
    std::string part;
    while (std::getline(ss, part, '_')) {
        parts.push_back(part);
    }

    if (parts.size() != 3) {
        return false; // Malformed name
    }

    // 4. Extract and convert parts
    try {
        out_step_num = static_cast<uint32_t>(std::stoul(parts[0]));
        out_type = parts[1]; // Directly assign the string
        out_target = parts[2];
        
        // Validate the extracted parts
        bool valid_type = (out_type == "insert" || out_type == "delete" || out_type == "search");
        bool valid_target = (out_target == "ids" || out_target == "vectors");

        if (!valid_type || !valid_target) {
            return false;
        }

    } catch (const std::invalid_argument&) {
        return false; // Step number wasn't a number
    } catch (const std::out_of_range&) {
        return false; // Step number was too large
    }

    return true;
}

std::vector<Step> extract_step_details() { 
    // Use a map to group files by step number.
    // Key: step_number, Value: Step struct
    std::map<uint32_t, Step> step_map;

    // --- POSIX Directory Reading ---
    DIR* dir;
    struct dirent* ent;

    if ((dir = opendir(DIRECTORY_PATH)) != NULL) {
        // Read all files and directories within directory
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;

            // Skip "." and ".."
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }

            // Skip non .pth files
            if (filename.length() < 3 || 
                filename.rfind(".pth") != (filename.length() - 4)) {
                continue;
            }

            uint32_t step_num;
            std::string type; // Now a string
            std::string target;

            if (parse_filename(filename, step_num, type, target)) {
                // Get or create the Step struct for this step number
                Step& current_step = step_map[step_num];

                // Populate its data
                current_step.step_number = step_num;
                current_step.type = type; // Assign the string directly
                
                std::string full_path = std::string(DIRECTORY_PATH) + filename;

                if (target == "ids") {
                    current_step.ids_path = full_path;
                } else if (target == "vectors") {
                    current_step.vectors_path = full_path;
                }
            } else { 
                std::cerr << "Failed to parse file name " << filename << std::endl;
                exit(1);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open directory " << DIRECTORY_PATH << std::endl;
        exit(1);
    }
    // --- End of Directory Reading ---


    // --- Convert map to final vector ---
    std::vector<Step> steps;
    for (std::map<uint32_t, Step>::const_iterator it = step_map.begin(); it != step_map.end(); ++it) {
        steps.push_back(it->second);
    }
    return steps;
}

Tensor load_tensor(std::string file_path) { 
    torch::jit::Module loaded_module = torch::jit::load(file_path);
    return loaded_module.attr("tensor").toTensor();;
}

constexpr bool RUN_MAINTEANCE = true;
void perform_mainteance(std::shared_ptr<QuakeIndex> index, std::ofstream& result_writer) { 
    if(RUN_MAINTEANCE) { 
        std::shared_ptr<MaintenanceTimingInfo> result = index->maintenance();
        result_writer << result->total_time_us/MS_TO_US << ",";
        
        std::cout << "Mainteance Metrics: Total Time ms - " << result->total_time_us/MS_TO_US << ", Num Splits - " << result->n_splits << ", Num Deletes - " << result->n_deletes;
        std::cout << ", Num Reclusters - " << result->n_recluster << ", Delete Time ms - " << result->delete_time_us/MS_TO_US << ", Split Time ms - " << result->split_time_us/MS_TO_US;
        std::cout << ", Recluster Time ms - " <<  result->recluster_time_us/MS_TO_US << ", Refinement Time ms - " << result->refinement_time_us/MS_TO_US << std::endl;
    }
}

void print_search_metrics(std::shared_ptr<SearchTimingInfo> timing_info, int level) { 
    std::cout << "Search Level " << level << " Timing Breakdown: " << std::endl;
    std::cout << "\t[Main] Total Time ms - " << timing_info->total_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Buffer Init Time ms - " << timing_info->buffer_init_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Copy Query Time ms - " << timing_info->copy_query_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Job Enqueue Time ms - " << timing_info->job_enqueue_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Boundary Distance ms - " << timing_info->boundary_distance_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] APS Time ms - " << timing_info->aps_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Scan Time ms - " << timing_info->scan_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Job Wait Time ms - " << timing_info->job_wait_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Main] Result Aggregate Time ms - " << timing_info->result_aggregate_time_ns/MS_TO_NS << std::endl << std::endl;

    std::cout << "\t[Worker] Total Worker Jobs Executed - " << timing_info->total_worker_jobs << std::endl;
    std::cout << "\t[Worker] Job Time ms - " << timing_info->worker_job_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Wait Time ms - " << timing_info->worker_wait_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Process Preamable Time ms - " << timing_info->worker_process_preamble_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Total Scan Time ms - " << timing_info->worker_scan_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Partition Size - " << timing_info->worker_partition_size << std::endl;
    std::cout << "\t[Worker] Partition Scan Bytes - " << timing_info->worker_partition_size_bytes << std::endl;
    std::cout << "\t[Worker] Global Partition Scan Through (GB/s) - " << timing_info->worker_scan_throughput << std::endl;
    std::cout << "\t[Worker] Local Partition Scan Through (GB/s) - " << timing_info->local_scan_throughput << std::endl;
    std::cout << "\t[Worker] Result Enque Time ms - " << timing_info->worker_enqueue_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Process Time ms - " << timing_info->worker_process_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Worker Batch Scan IPC - " << timing_info->worker_batch_scan_ipc << std::endl;
    std::cout << "\t[Worker] Worker Batch Scan Miss Rate - " << timing_info->worker_batch_scan_miss_rate << std::endl;
    std::cout << "\t[Worker] Single Scan Time us - " << timing_info->single_scan_job_time_ns/MS_TO_US << std::endl;
    std::cout << "\t[Worker] Worker Batch Scan Norms X us - " << timing_info->faiss_norms_x_time_ns/MS_TO_US << std::endl;
    std::cout << "\t[Worker] Worker Batch Scan Norms Y us - " << timing_info->faiss_norms_y_time_ns/MS_TO_US << std::endl;
    std::cout << "\t[Worker] Sgemm Matrix Multiply us - " << timing_info->sgemm_time_ns/MS_TO_US << std::endl;
    std::cout << "\t[Worker] IP TO L2 us - " << timing_info->ip_to_l2_time_ns/MS_TO_US << std::endl;
    std::cout << "\t[Worker] Top K Buffer Add us - " << timing_info->top_k_buffer_add_ns/MS_TO_US << std::endl;
}

std::pair<Tensor, Tensor> load_ground_truth_data(uint32_t step_num) { 
    // Open the input file
    std::string gt_path = std::string(GROUND_TRUTH_PATH) + "step" + std::to_string(step_num) + ".gt100";
    std::ifstream input_file(gt_path, std::ios::binary | std::ios::in);
    if (!input_file) {
        throw std::runtime_error("Failed to open file: " + gt_path);
    }

    // Read header (two uint32_t values: n and d)
    uint32_t n, d;
    input_file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    input_file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));

    // Read the gt ids
    std::vector<int32_t> ids_data(n * d);
    input_file.read(reinterpret_cast<char*>(ids_data.data()), n * d * sizeof(int32_t));
    Tensor gt_ids = torch::from_blob(ids_data.data(), {static_cast<long>(n), static_cast<long>(d)}, torch::kInt32).clone();

    // Read the gt distances 
    std::vector<float> distances_data(n * d);
    input_file.read(reinterpret_cast<char*>(distances_data.data()), n * d * sizeof(float));
    Tensor gt_distances = torch::from_blob(distances_data.data(), {static_cast<long>(n), static_cast<long>(d)}, torch::kFloat32).clone();

    return std::make_pair(gt_ids, gt_distances); // Return empty tensor for distances
}

float calculateStandardDeviation(const std::vector<float>& data, float mean) {
    if (data.empty()) {
        return 0.0; // Or handle error appropriately
    }

    // Calculate the variance
    float sum_sq_diff = 0.0;
    for (double value : data) {
        sum_sq_diff += std::pow(value - mean, 2);
    }

    // Use that to calculate the std dev
    return std::sqrt(sum_sq_diff / (data.size() - 1));
}

constexpr bool LOG_RECALL_IDS = false;
constexpr bool PRINT_GT_DISTANCES = false;
std::pair<float, float> calculate_recall(std::shared_ptr<QuakeIndex> index, Tensor queries, std::shared_ptr<SearchResult> search_result, uint32_t step_num) { 
    // Load the tensors
    std::pair<torch::Tensor, torch::Tensor> gt_data = load_ground_truth_data(step_num);
    Tensor gt_ids = gt_data.first.to(torch::kInt64).contiguous();
    Tensor gt_distances = gt_data.second.to(torch::kFloat32).contiguous();

    Tensor predicted_ids = search_result->ids.to(torch::kInt64).contiguous();
    Tensor predicted_distances = search_result->distances.to(torch::kFloat32).contiguous();

    assert(gt_ids.sizes()[0] == predicted_ids.sizes()[0]);

    // Calculate the recall for each query
    int num_queries = predicted_ids.sizes()[0];
    int k = predicted_ids.sizes()[1];
    Tensor gt_top_k = gt_ids.slice(1, 0, k).contiguous();
    
    // Calculate the recall for each query
    std::vector<float> per_query_recall;
    int64_t* gt_ids_ptr = gt_top_k.data_ptr<int64_t>();
    float* gt_distances_ptr = gt_distances.data_ptr<float>();

    int64_t* predicted_ids_ptr = predicted_ids.data_ptr<int64_t>();
    float* predicted_distances_ptr = predicted_distances.data_ptr<float>();
    for(int i = 0; i < num_queries; i++) {
        // Get the ground truth and predicted sets
        int64_t* query_gt = gt_ids_ptr + i * k; float* query_gt_distances = gt_distances_ptr + i * k;
        int64_t* query_predicted = predicted_ids_ptr + i * k; float* query_predicted_distances = predicted_distances_ptr + i * k;
        std::set<int64_t> gt_ids; std::set<int64_t> predicted_ids; 
        if constexpr(LOG_RECALL_IDS) std::cout << "Query " << i << " Result: " << std::endl;
        for(int j = 0; j < k; j++) { 
            gt_ids.insert(query_gt[j]);
            predicted_ids.insert(query_predicted[j]);
        }

        // Get their set intersection
        std::set<int64_t> intersection_set;
        std::set_intersection(gt_ids.begin(), gt_ids.end(),
                            predicted_ids.begin(), predicted_ids.end(),
                            std::inserter(intersection_set, intersection_set.begin()));

        // Use that to predict the recall
        float query_recall = (1.0 * intersection_set.size())/k;
        per_query_recall.push_back(query_recall);
    }

    std::shared_ptr<faiss::DynamicInvertedLists> partition_store = index->partition_manager_->partition_store_;
    queries = queries.contiguous();
    float* all_queries_vectors_ptr = queries.data_ptr<float>();
    int d = partition_store->d_;

    if constexpr(PRINT_GT_DISTANCES) { 
        for(int i = 0; i < num_queries; i++) {
            std::cout << "-------------" << std::endl;
            std::cout << "Query " << i << " Ground Truth Details: " << std::endl;
            int64_t* query_gt = gt_ids_ptr + i * k; float* query_gt_distances = gt_distances_ptr + i * k;
            int64_t* query_predicted = predicted_ids_ptr + i * k; float* query_predicted_distances = predicted_distances_ptr + i * k;
            float* query_vector_ptr = all_queries_vectors_ptr + i * d;

            // First log GT vector results
            for(int j = 0; j < k; j++) { 
                int64_t curr_gt_id = query_gt[j];
                std::cout << "\tGT Result " << j << " - ID: " << curr_gt_id << ", GT Distance: " << query_gt_distances[j];

                // Also calculate the distance of the gt vector to the query vector
                std::pair<IndexPartition*, int64_t> vector_location = partition_store->id_to_location_[curr_gt_id];
                float* gt_vector_ptr = reinterpret_cast<float*>(vector_location.first->codes_) + vector_location.second * d;
                float distance = std::sqrt(faiss::fvec_L2sqr(query_vector_ptr, reinterpret_cast<float*>(gt_vector_ptr), d));

                std::cout << ", GT Vector Faiss L2 Distance: " << distance << std::endl;
            }

            std::cout << std::endl;

            // Now log the predicted results
            std::cout << "Query " << i << " Search Details: " << std::endl;
            for(int j = 0; j < k; j++) {
                float curr_distance = query_predicted_distances[j];
                float l2_value = std::pow(curr_distance, 2);
                std::cout << "\tSearch Result " << j << " - ID: " << query_predicted[j] << ", L2 Distance: " << l2_value << std::endl;
            }

            std::cout << "-------------" << std::endl;
        }
    }
    
    // Return the average recall across all the queries
    float mean = std::accumulate(per_query_recall.begin(), per_query_recall.end(), 0.0) / per_query_recall.size();
    float standard_dev = calculateStandardDeviation(per_query_recall, mean);
    return std::make_pair(mean, standard_dev);
}

std::shared_ptr<QuakeIndex> build_index(Step& build_step, int num_search_workers) { 
    // Verify step
    if(build_step.step_number != 1 || build_step.type != "insert") { 
        std::cerr << "Build Index called with step num " << build_step.step_number << " and type " << build_step.type << std::endl;
        exit(1);
    }

    // Create and build the index
    std::shared_ptr<QuakeIndex> index = std::make_shared<QuakeIndex>();

    std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
    build_params->dimension = 100;
    build_params->nlist = 1000;
    build_params->num_workers = num_search_workers;
    build_params->metric = "l2";
    build_params->niter = 25;
    build_params->use_numa = false;

    build_params->parent_params = std::make_shared<IndexBuildParams>();
    build_params->parent_params->dimension = 100;
    build_params->parent_params->nlist = 1;
    build_params->parent_params->metric = "l2";
    build_params->parent_params->num_workers = 0;
    build_params->parent_params->use_numa = false;

    Tensor build_vectors = load_tensor(build_step.vectors_path).to(torch::kFloat32);
    Tensor build_ids = load_tensor(build_step.ids_path).to(torch::kInt64);
    index->build(build_vectors, build_ids, build_params);

    // Also initialize with the default mainteance policy
    std::shared_ptr<MaintenancePolicyParams> mainteance_policy = std::make_shared<MaintenancePolicyParams>();
    mainteance_policy->window_size = 5000;
    mainteance_policy->split_threshold_ns = 3250;
    mainteance_policy->split_knn_iterations = 6;
    mainteance_policy->delete_threshold_ns = 2000;
    mainteance_policy->partition_reduction_threshold = 0.22;
    mainteance_policy->refinement_radius = 0;
    mainteance_policy->refinement_iterations = 5;
    mainteance_policy->min_partition_size = 1500;
    mainteance_policy->enable_split_rejection = true;
    mainteance_policy->enable_delete_rejection = true;
    index->initialize_maintenance_policy(mainteance_policy);

    return index;
}

constexpr bool CALCULATE_RECALL_GIVEN_GT_PARITITONS = false;
std::pair<float, float> check_gt_partitions_scanned(std::shared_ptr<QuakeIndex> index, Step& search_step, float partition_search_fraction) { 
    // Load the search query
    Tensor search_queries = load_tensor(search_step.vectors_path).to(torch::kFloat32);

    // Get the ground ids that were scanned
    std::pair<torch::Tensor, torch::Tensor> gt_data = load_ground_truth_data(search_step.step_number);
    Tensor gt_ids = gt_data.first.to(torch::kInt64);
    int k = 10;
    Tensor gt_top_k = gt_ids.slice(1, 0, k).contiguous();

    // Get the partitions that the ground ids belong to
    int64_t* gt_ids_ptr = gt_top_k.data_ptr<int64_t>();
    int num_queries = gt_ids.sizes()[0];
    std::shared_ptr<faiss::DynamicInvertedLists> partition_store = index->partition_manager_->partition_store_;
    if (partition_store->id_to_location_.empty()) partition_store->build_map();

    std::vector<std::vector<int64_t>> gt_partitions(num_queries);
    for(int i = 0; i < num_queries; i++) {
        int64_t* query_gt = gt_ids_ptr + i * k;
        for(int j = 0; j < k; j++) { 
            // Get the partition id for this gt id
            int64_t vector_id = query_gt[j];
            if(partition_store->id_to_location_.find(vector_id) == partition_store->id_to_location_.end()) { 
                std::string err_message = std::string("[ERROR] Failed to find id ") + std::to_string(vector_id) + " in partition store";
                throw std::runtime_error(err_message);
            }

            IndexPartition* partition = partition_store->id_to_location_[vector_id].first;
            if(partition->partition_id_ == -1) { 
                std::string err_message = std::string("[ERROR] Vector id ") + std::to_string(vector_id) + " maps to partition with invalid partition id -1";
                throw std::runtime_error(err_message);
            }

            gt_partitions[i].push_back(partition->partition_id_);
        }
    }

    if constexpr(CALCULATE_RECALL_GIVEN_GT_PARITITONS) { 
        // Initialize the tensor of the gt partitions
        Tensor partitions_to_scan = torch::full({num_queries, k}, -1, torch::kInt64);
        auto search_partition_id_accessor = partitions_to_scan.accessor<int64_t, 2>();
        for(int i = 0; i < num_queries; i++) {
            std::set<int64_t> unique_partitions;
            for(int j = 0; j < k; j++) { 
                int64_t curr_partition_id = static_cast<int64_t>(gt_partitions[i][j]);
                if(unique_partitions.find(curr_partition_id) != unique_partitions.end()) {
                    curr_partition_id = -1;
                } else { 
                    unique_partitions.insert(curr_partition_id);
                }
                search_partition_id_accessor[i][j] = curr_partition_id;
            }  
        }

        // Create the search params for the partition scan
        std::shared_ptr<SearchParams> search_params = std::make_shared<SearchParams>();
        search_params->nprobe = k;
        search_params->k = k;
        search_params->recall_target = -1.0;
        search_params->batched_scan = true;
        search_params->batch_size = 500;
        search_params->track_hits = false;

        // Get the resulting vectors and return the recall/mean
        std::shared_ptr<SearchResult> gt_search_result = index->query_coordinator_->scan_partitions(search_queries, partitions_to_scan, search_params);

        std::pair<float, float> recall_result = calculate_recall(index, search_queries, gt_search_result, search_step.step_number);
        std::cout << "Scanning index with GT partitions Recall: Mean - " << recall_result.first << ", Std Dev - " << recall_result.second << std::endl;
    }

    // Now search the parent for the query vectors to get the partitions that were scanned
    std::shared_ptr<SearchParams> search_params = std::make_shared<SearchParams>();
    int num_partitions_to_scan = static_cast<int>(index->nlist());
    search_params->k = num_partitions_to_scan;
    search_params->batched_scan = true;
    search_params->track_hits = false;
    search_params->batch_size = 500;
    std::shared_ptr<QuakeIndex> parent_index = index->parent_;
    std::shared_ptr<SearchResult> search_result = parent_index->search(search_queries, search_params);

    // Verify that the ground truth partitions were scanned
    Tensor searched_ids = search_result->ids.to(torch::kInt64).contiguous();
    int64_t* searched_ids_ptr = searched_ids.data_ptr<int64_t>();
    Tensor search_dists = search_result->distances.to(torch::kFloat32).contiguous();
    float* searched_dists_ptr = search_dists.data_ptr<float>();

    std::shared_ptr<faiss::DynamicInvertedLists> parent_partition_store = parent_index->partition_manager_->partition_store_;
    int d = parent_partition_store->d_;

    search_queries = search_queries.contiguous();
    float* all_queries_vectors_ptr = search_queries.data_ptr<float>();
    std::vector<float> per_query_scan_fraction(num_queries);
    for(int i = 0; i < num_queries; i++) {
        // Get the ranking of the partitions for this query
        float* query_vec_ptr = all_queries_vectors_ptr + i * d;
        int64_t* query_searched = searched_ids_ptr + i * num_partitions_to_scan;
        float* query_searched_dists = searched_dists_ptr + i * num_partitions_to_scan;
        std::unordered_map<int64_t, std::pair<int, float>> partition_loc_map; 

        for(int j = 0; j < num_partitions_to_scan; j++) { 
            int64_t partition_id = query_searched[j];
            if(partition_id != -1) partition_loc_map[partition_id] = std::make_pair(j, query_searched_dists[j]);
        }

        // Now determine the max rank of the gt partitions
        bool log_query = false;
        if(log_query) std::cout << "Query " << i << " GT Partition Ranks: " << std::endl;
        int max_rank = -1;
        for(int64_t gt_partition_id : gt_partitions[i]) { 
            if(partition_loc_map.find(gt_partition_id) == partition_loc_map.end()) { 
                std::cerr << "[ERROR] Query " << i << " didn't get rank for gt partition id " << gt_partition_id << std::endl;
                std::cout << "Result for " << num_partitions_to_scan << " partitions: ";
                for(int j = 0; j < num_partitions_to_scan; j++) { 
                    std::cout << "(" << query_searched[j] << "," << query_searched_dists[j] << std::endl;
                }
                std::cout << std::endl;
                exit(1);
            }
            std::pair<int, float> partition_details = partition_loc_map[gt_partition_id];

            max_rank = std::max(max_rank, partition_details.first);
            if(log_query) { 
                std::cout << "\t GT Partition ID: " << gt_partition_id << ", Rank: " << partition_details.first << "/" << num_partitions_to_scan << ", Quake Distance: " << partition_details.second;
                
                // Get the centroid with the ground truth partition id 
                if(parent_partition_store->id_to_location_.empty()) parent_partition_store->build_map();
                if(parent_partition_store->id_to_location_.find(gt_partition_id) == parent_partition_store->id_to_location_.end()) { 
                    std::string err_message = std::string("[ERROR] Failed to find partition id ") + std::to_string(gt_partition_id) + " in parent partition store";
                    throw std::runtime_error(err_message);
                }

                // Calculate the distance to it from the query vector
                std::pair<IndexPartition*, int64_t> vector_location = parent_partition_store->id_to_location_[gt_partition_id];
                float* centorid_vec_ptr = reinterpret_cast<float*>(vector_location.first->codes_) + vector_location.second * d;
                float faiss_distance = std::sqrt(faiss::fvec_L2sqr(query_vec_ptr, centorid_vec_ptr, d));
                std::cout << ", Faiss Distance: " << faiss_distance << std::endl;
            } 
        }
        if(log_query) std::cout << std::endl;

        // Now calculate the percentage of centroids we would need to scan to get all the gt partitions
        per_query_scan_fraction[i] = (100.0 * max_rank)/num_partitions_to_scan;
    }

    // Calculate the mean and std dev of scan fraction
    float mean = std::accumulate(per_query_scan_fraction.begin(), per_query_scan_fraction.end(), 0.0) / per_query_scan_fraction.size();
    float standard_dev = calculateStandardDeviation(per_query_scan_fraction, mean);
    return std::make_pair(mean, standard_dev);
}

static int CURR_BATCH_SIZE = 256;
constexpr bool CHECK_QUERY_RECALL = true;
constexpr bool CHECK_GT_PARITIONS_SCANNED = false;
constexpr float RECALL_TARGET = -1.0; // Use this to enable/disable APS
void perform_search(std::shared_ptr<QuakeIndex> index, Step& search_step, std::ofstream& result_writer, float partition_search_fraction) { 
    // Load the search query
    Tensor search_queries = load_tensor(search_step.vectors_path).to(torch::kFloat32);

    // Create the search parameters
    std::shared_ptr<SearchParams> search_params = std::make_shared<SearchParams>();
    search_params->k = 10;
    search_params->nprobe = static_cast<int>(partition_search_fraction * index->nlist());
    search_params->recall_target = RECALL_TARGET;
    search_params->initial_search_fraction = partition_search_fraction;
    search_params->batched_scan = true;
    search_params->batch_size = CURR_BATCH_SIZE;
    search_params->track_hits = true;

    // Run the search
    std::shared_ptr<SearchResult> search_result = index->search(search_queries, search_params);
    print_search_metrics(search_result->timing_info->parent_info, 0);
    std::cout << std::endl;
    print_search_metrics(search_result->timing_info, 1);
    std::cout << std::endl;
    
    // Calculate the recall
    auto timing_info = search_result->timing_info;
    result_writer << search_step.step_number << ",search," << timing_info->total_time_ns/MS_TO_NS << ","; 
    result_writer << timing_info->worker_partition_size << "," << timing_info->worker_scan_time_ns/MS_TO_NS << ",";
    result_writer << timing_info->local_scan_throughput << "," << timing_info->worker_enqueue_time_ns/MS_TO_NS << ",";
    result_writer << timing_info->worker_batch_scan_ipc << "," << timing_info->worker_batch_scan_miss_rate << ",";
    if constexpr(CHECK_QUERY_RECALL) { 
        std::pair<float, float> recall_result = calculate_recall(index, search_queries, search_result, search_step.step_number);
        std::cout << "\nSearch Recall: Mean - " << recall_result.first << ", Std Dev - " << recall_result.second << std::endl;  
        result_writer << recall_result.first << "," << recall_result.second << ",";
    } else { 
        result_writer << "-1.0,-1.0,";   
    }    

    // Optionally perform gt checks
    if constexpr(CHECK_GT_PARITIONS_SCANNED) { 
        std::pair<float, float> gt_partitions_result = check_gt_partitions_scanned(index, search_step, partition_search_fraction);
        std::cout << "\nGT Percent of Partitions to Scan: Mean - " << gt_partitions_result.first << ", Std Dev - " << gt_partitions_result.second << std::endl;
        result_writer << gt_partitions_result.first << "," << gt_partitions_result.second << ",";
    } else { 
        result_writer << "-1.0,-1.0,";
    }
}

constexpr size_t INSERT_CHUNK_SIZE = 10000;
void perform_insert(std::shared_ptr<QuakeIndex> index, Step& insert_step, std::ofstream& result_writer) { 
    // Load the arguments
    Tensor insert_vectors = load_tensor(insert_step.vectors_path).to(torch::kFloat32);
    Tensor insert_ids = load_tensor(insert_step.ids_path).to(torch::kInt64);

    // Insert in the vectors in chunk
    int total_time_us = 0;
    size_t num_vectors = insert_vectors.size(0);
    size_t num_chunks = (num_vectors + INSERT_CHUNK_SIZE - 1)/INSERT_CHUNK_SIZE;
    for(size_t i = 0; i < num_chunks; i++) { 
        size_t start_idx = i * INSERT_CHUNK_SIZE;
        size_t end_idx = std::min(start_idx + INSERT_CHUNK_SIZE, num_vectors);

        auto result = index->add(insert_vectors.slice(0, start_idx, end_idx), insert_ids.slice(0, start_idx, end_idx));
        total_time_us += result->modify_time_us;
    }
    std::cout << "Finished insertion in " << num_chunks << " chunks in " << total_time_us << " us" << std::endl;

    result_writer << insert_step.step_number << ",insert," << total_time_us/MS_TO_US << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0,-1.0,";
    result_writer << "-1.0,-1.0,";
}

constexpr size_t DELETE_CHUNK_SIZE = 10000;
void perform_delete(std::shared_ptr<QuakeIndex> index, Step& delete_step, std::ofstream& result_writer) { 
    Tensor delete_ids = load_tensor(delete_step.ids_path).to(torch::kInt64);

    // Insert in the vectors in chunk
    int total_time_us = 0;
    size_t num_vectors = delete_ids.size(0);
    size_t num_chunks = (num_vectors + DELETE_CHUNK_SIZE - 1)/DELETE_CHUNK_SIZE;
    for(size_t i = 0; i < num_chunks; i++) { 
        size_t start_idx = i * DELETE_CHUNK_SIZE;
        size_t end_idx = std::min(start_idx + DELETE_CHUNK_SIZE, num_vectors);

        auto result = index->remove(delete_ids.slice(0, start_idx, end_idx));
        total_time_us += result->modify_time_us;
    }
    std::cout << "Finished delete in " << num_chunks << " chunks in " << total_time_us << " us" << std::endl;

    result_writer << delete_step.step_number << ",delete," << total_time_us/MS_TO_US << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0" << "," << "-1.0" << ",";
    result_writer << "-1.0,-1.0,";
    result_writer << "-1.0,-1.0,";
}

constexpr float BYTES_TO_GB = 1000.0 * 1000.0 * 1000.0;

float log_memory_stats(std::shared_ptr<QuakeIndex> index, int level, std::ofstream& result_writer) { 
    auto partition_map = index->partition_manager_->partition_store_->partitions_;
    int64_t total_vectors = 0;
    int64_t buffer_capacity = 0;
    int64_t total_memory = 0;

    for(const auto& pair : partition_map) {
        total_vectors += pair.second->num_vectors_;
        buffer_capacity += pair.second->buffer_size_;
        total_memory += pair.second->buffer_size_ * (pair.second->code_size_ + sizeof(idx_t));
    }

    // Only log the memory for the first level
    float total_memory_gb = total_memory/BYTES_TO_GB;

    // std::cout << "Level " << level << " Memory Consumption: Vectors - " << total_vectors << ", Buffer Capacity - " << buffer_capacity << ", Memory (GB) - " << total_memory_gb << std::endl;

    if(index->parent_ != nullptr) { 
        total_memory_gb += log_memory_stats(index->parent_, level + 1, result_writer);
    }

    return total_memory_gb;
}

// Code to calculate the Sillehoute Score for the matrix
struct PartitionView {
    int64_t id;
    std::vector<const float*> vectors;
};

std::pair<float, float> calculate_silhouette_scores(shared_ptr<QuakeIndex> index, int clustering_num_samples = 0, int num_score_calculate_workers = 4) {
    if (!index || !index->partition_manager_ || !index->partition_manager_->partition_store_) {
        throw std::runtime_error("Invalid QuakeIndex: PartitionManager or Store is null.");
    }

    auto pm = index->partition_manager_;
    auto store = pm->partition_store_;
    int d = pm->d();

    // 1. Organize vectors by partition for efficient access
    std::vector<PartitionView> partitions;
    int64_t total_vectors = 0;

    for (const auto& entry : store->partitions_) {
        int64_t pid = entry.first;
        shared_ptr<IndexPartition> part = entry.second;
        if (part->num_vectors_ == 0) continue;

        PartitionView view;
        view.id = pid;
        view.vectors.reserve(part->num_vectors_);

        const uint8_t* codes = part->codes_;
        size_t code_size = part->code_size_;

        for (int64_t i = 0; i < part->num_vectors_; ++i) {
            view.vectors.push_back(reinterpret_cast<const float*>(codes + i * code_size));
        }
        partitions.push_back(view);
        total_vectors += part->num_vectors_;
    }

    // 2. Flatten for parallel iteration (tuples of: partition_idx, vector_idx_in_partition)
    struct Task {
        size_t p_idx; // Index in `partitions` vector
        size_t v_idx; // Index inside that partition
    };

    std::vector<Task> tasks;
    tasks.reserve(total_vectors);

    for (size_t p = 0; p < partitions.size(); ++p) {
        for (size_t v = 0; v < partitions[p].vectors.size(); ++v) {
            tasks.push_back({p, v});
        }
    }

    // Apply sampling if requested
    if (clustering_num_samples > 0) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::shuffle(tasks.begin(), tasks.end(), generator);

        tasks.resize(clustering_num_samples);
    }

    std::vector<float> scores(tasks.size());

    // 3. Compute scores in parallel
    parallel_for(size_t(0), tasks.size(), [&](size_t t) {
        const auto& task = tasks[t];
        const auto& current_partition = partitions[task.p_idx];
        const float* current_vec = current_partition.vectors[task.v_idx];

        // --- Calculate a(i): Mean dist to same cluster ---
        double sum_dist_a = 0.0;
        size_t count_a = 0;

        for (size_t other_v = 0; other_v < current_partition.vectors.size(); ++other_v) {
            if (task.v_idx == other_v) continue; 
            
            // Use Faiss optimized L2 squared distance
            float dist_sq = faiss::fvec_L2sqr(current_vec, current_partition.vectors[other_v], d);
            sum_dist_a += std::sqrt(dist_sq); 
            count_a++;
        }

        float a_i = (count_a > 0) ? (sum_dist_a / count_a) : 0.0f;

        // Silhouette is 0 for singleton clusters
        if (current_partition.vectors.size() <= 1) {
            scores[t] = 0.0f;
            return; 
        }

        // --- Calculate b(i): Min mean dist to other clusters ---
        float b_i = std::numeric_limits<float>::max();

        for (size_t p = 0; p < partitions.size(); ++p) {
            if (p == task.p_idx) continue; 

            const auto& other_partition = partitions[p];
            if (other_partition.vectors.empty()) continue;

            double sum_dist_b = 0.0;
            
            for (const float* other_vec : other_partition.vectors) {
                float dist_sq = faiss::fvec_L2sqr(current_vec, other_vec, d);
                sum_dist_b += std::sqrt(dist_sq);
            }

            float mean_dist_b = sum_dist_b / other_partition.vectors.size();
            if (mean_dist_b < b_i) {
                b_i = mean_dist_b;
            }
        }

        // Use a_i and b_i to figure out the score
        if (b_i == std::numeric_limits<float>::max()) {
            scores[t] = 0.0f;
        } else {
            scores[t] = (b_i - a_i) / std::max(a_i, b_i);
        }

    }, num_score_calculate_workers); // End parallel_for

    // Calculate the mean and std dev of scan fraction
    float mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    float standard_dev = calculateStandardDeviation(scores, mean);
    return std::make_pair(mean, standard_dev);
}

constexpr float SCAN_PERCENTAGE_RANGE[2] = {0.13, 0.13};
constexpr size_t MIN_OPERATIONS_BEFORE_MAINTEANCE = 0;
constexpr size_t NUM_OPERATIONS_BETWEEN_MAINTEANCE = 1;
constexpr size_t NUM_TEST_OPERATIONS = 0;

#define RESULT_WRITE_PATH "../scripts/big_ann_perf_numbers/perf_debug_scan_0.13_worker_batch_clustering_score.csv"

int main() { 
    // Configure global params
    IndexPartition::delete_resize_threshold_ = DELETE_THRESHOLD;
    IndexPartition::capacity_resize_threshold_ = CAPACITY_THRESHOLD;

    // Extract the steps
    std::vector<Step> steps_arr = extract_step_details();
    std::cout << "Loaded " << steps_arr.size() << " steps" << std::endl;
    assert(!steps_arr.empty());

    // Create the output csv file
    std::ofstream result_writer(RESULT_WRITE_PATH);
    result_writer << "step_num,step_type,latency_ms,worker_partition_size,worker_scan_time_ms,worker_scan_throughput,worker_result_time_ms,measured_ipc,cache_miss_rate,recall_mean,recall_std_dev,gt_scan_mean,gt_scan_dev,mainteance_ms,index_mem_gb,num_partitions,num_vectors,silleheoute_mean,silleheoute_std_dev" << std::endl;

    int num_default_workers = 8;
    float clustering_num_samples = 8192;
    int num_score_calculate_workers = 32;

    auto build_start_time = std::chrono::high_resolution_clock::now();
    std::shared_ptr<QuakeIndex> index = build_index(steps_arr[0], num_default_workers); 
    auto build_end_time = std::chrono::high_resolution_clock::now();
    int64_t build_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(build_end_time - build_start_time).count();
    std::cout << "Initialized Quake Index with batch size in " << build_time_ns/MS_TO_NS << " ms" << std::endl;

    // Now perform the streaming operations against the index
    size_t num_test_operations = NUM_TEST_OPERATIONS; 
    if(num_test_operations == 0) { 
        num_test_operations = steps_arr.size();
    } 

    float curr_scan_percentage = SCAN_PERCENTAGE_RANGE[1];
    float scan_percentage_step = (SCAN_PERCENTAGE_RANGE[1] - SCAN_PERCENTAGE_RANGE[0])/num_test_operations;

    for(size_t i = 1; i <= num_test_operations; i++) { 
        // Run the step
        Step& curr_step = steps_arr[i];
        std::cout << "\n------ START: Step " << curr_step.step_number << " of type " << curr_step.type << " -----" << std::endl;
        if(curr_step.type == "search") { 
            perform_search(index, curr_step, result_writer, curr_scan_percentage);
        } else if(curr_step.type == "insert") { 
            perform_insert(index, curr_step, result_writer);
        } else if(curr_step.type == "delete") { 
            perform_delete(index, curr_step, result_writer);
        } else { 
            std::cerr << "Unsupport step type of " << curr_step.type << std::endl;
            exit(1);
        }

        // Run the mainteance
        if(curr_step.step_number > MIN_OPERATIONS_BEFORE_MAINTEANCE && curr_step.step_number % NUM_OPERATIONS_BETWEEN_MAINTEANCE == 0) {
            std::cout << std::endl;
            perform_mainteance(index, result_writer);
        } else { 
            result_writer << "0.0,";
        }

        std::cout << std::endl;
        float index_memory_gb = log_memory_stats(index, 0, result_writer);
        result_writer << index_memory_gb << "," << index->nlist() << "," << index->ntotal() << ",";

        // Get the silleheoute score
        std::pair<float, float> silleheouet_score = calculate_silhouette_scores(index, clustering_num_samples, num_score_calculate_workers);
        std::cout << "Silleheoute Score: Mean - " << silleheouet_score.first << ", Std Dev - " << silleheouet_score.second << std::endl;
        result_writer << silleheouet_score.first << "," << silleheouet_score.second << "," << std::endl;
        
        std::cout << "------ FINISH: Step " << curr_step.step_number << " of type " << curr_step.type << " ------\n" << std::endl;

        curr_scan_percentage -= scan_percentage_step;
    }

    result_writer.close();
    return 0;
}