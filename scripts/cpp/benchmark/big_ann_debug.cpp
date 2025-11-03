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

#include <torch/script.h>    // For c10::IValue
#include <torch/serialize.h> // For torch::load
#include "quake_index.h"  // Quake API header

using torch::Tensor;

#define DATASET_PATH "/home/devesh/big-ann-benchmark/big-ann-benchmarks/data/MSTuring-30M-clustered/"
#define DIRECTORY_PATH DATASET_PATH "index_arguments/"
#define GROUND_TRUTH_PATH DATASET_PATH "29998994/final_runbook.yaml/"

constexpr bool RUN_MAINTEANCE = true;
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

void perform_mainteance(std::shared_ptr<QuakeIndex> index, std::ofstream& result_writer) { 
    if(RUN_MAINTEANCE) { 
        std::shared_ptr<MaintenanceTimingInfo> result = index->maintenance();
        result_writer << result->total_time_us/MS_TO_US << ",";
        
        std::cout << "Mainteance Metrics: Total Time ms - " << result->total_time_us/MS_TO_US << ", Num Splits - " << result->n_splits << ", Num Deletes - " << result->n_deletes << ", Delete Time ms - " << result->delete_time_us/MS_TO_US;
        std::cout << ", Split Time ms - " << result->split_time_us/MS_TO_US << ", Refinement Time ms - " << result->refinement_time_us/MS_TO_US << std::endl;
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

    std::cout << "\t[Worker] Job Time ms - " << timing_info->worker_job_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Wait Time ms - " << timing_info->worker_wait_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Process Preamable Time ms - " << timing_info->worker_process_preamble_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Scan Time ms - " << timing_info->worker_scan_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Result Enque Time ms - " << timing_info->worker_enqueue_time_ns/MS_TO_NS << std::endl;
    std::cout << "\t[Worker] Process Time ms - " << timing_info->worker_process_time_ns/MS_TO_NS << std::endl;
}

Tensor load_ground_truth_ids(uint32_t step_num) { 
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
    std::vector<int32_t> I_data(n * d);
    input_file.read(reinterpret_cast<char*>(I_data.data()), n * d * sizeof(int32_t));

    return torch::from_blob(I_data.data(), {static_cast<long>(n), static_cast<long>(d)}, torch::kInt32).clone();
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

std::pair<float, float> calculate_recall(std::shared_ptr<SearchResult> search_result, uint32_t step_num) { 
    // Load the tensors
    Tensor gt_ids = load_ground_truth_ids(step_num).to(torch::kInt64);
    Tensor predicted_ids = search_result->ids.to(torch::kInt64).contiguous();
    assert(gt_ids.sizes()[0] == predicted_ids.sizes()[0]);

    // Calculate the recall for each query
    int num_queries = predicted_ids.sizes()[0];
    int k = predicted_ids.sizes()[1];
    Tensor gt_top_k = gt_ids.slice(1, 0, k).contiguous();
    
    // Calculate the recall for each query
    std::vector<float> per_query_recall;
    int64_t* gt_ids_ptr = gt_top_k.data_ptr<int64_t>();
    int64_t* predicted_ids_ptr = predicted_ids.data_ptr<int64_t>();
    for(int i = 0; i < num_queries; i++) {
        // Get the ground truth and predicted sets
        int64_t* query_gt = gt_ids_ptr + i * k;
        int64_t* query_predicted = predicted_ids_ptr + i * k;
        std::set<int64_t> gt_ids; std::set<int64_t> predicted_ids; 
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
    
    // Return the average recall across all the queries
    float mean = std::accumulate(per_query_recall.begin(), per_query_recall.end(), 0.0) / per_query_recall.size();
    float standard_dev = calculateStandardDeviation(per_query_recall, mean);
    return std::make_pair(mean, standard_dev);
}

std::shared_ptr<QuakeIndex> build_index(Step& build_step) { 
    // Verify step
    if(build_step.step_number != 1 || build_step.type != "insert") { 
        std::cerr << "Build Index called with step num " << build_step.step_number << " and type " << build_step.type << std::endl;
        exit(1);
    }

    // Allocate some memory but don't free it (to trigger LSAN)
    char* allocated_memory = reinterpret_cast<char*>(std::malloc(100 * sizeof(char)));

    // Create and build the index
    std::shared_ptr<QuakeIndex> index = std::make_shared<QuakeIndex>();

    std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 512;
    build_params->num_workers = 8;
    build_params->metric = "l2";

    build_params->parent_params = std::make_shared<IndexBuildParams>();
    build_params->parent_params->metric = "l2";
    build_params->parent_params->num_workers = 0;

    Tensor build_vectors = load_tensor(build_step.vectors_path).to(torch::kFloat32);
    Tensor build_ids = load_tensor(build_step.ids_path).to(torch::kInt64);
    index->build(build_vectors, build_ids, build_params);

    // Also initialize with the default mainteance policy
    std::shared_ptr<MaintenancePolicyParams> mainteance_policy = std::make_shared<MaintenancePolicyParams>();
    mainteance_policy->window_size = 2500;
    mainteance_policy->split_threshold_ns = 10000;
    mainteance_policy->delete_threshold_ns = 75000;
    mainteance_policy->refinement_radius = 10;
    mainteance_policy->refinement_iterations = 1;
    mainteance_policy->min_partition_size = 8192;
    mainteance_policy->enable_split_rejection = true;
    mainteance_policy->enable_delete_rejection = true;
    index->initialize_maintenance_policy(mainteance_policy);

    return index;
}

constexpr float RECALL_TARGET = 0.9;
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
    search_params->batch_size = 500;

    // Run the search
    std::shared_ptr<SearchResult> search_result = index->search(search_queries, search_params);
    print_search_metrics(search_result->timing_info->parent_info, 0);
    std::cout << std::endl;
    print_search_metrics(search_result->timing_info, 1);
    
    // Log the results
    std::pair<float, float> recall_result = calculate_recall(search_result, search_step.step_number);
    std::cout << "\nSearch Recall: Mean - " << recall_result.first << ", Std Dev - " << recall_result.second << std::endl;

    result_writer << search_step.step_number << ",search," << search_result->timing_info->total_time_ns/MS_TO_NS << ","; 
    result_writer << recall_result.first << "," << recall_result.second << ",";
}

constexpr size_t INSERT_CHUNK_SIZE = 5000;
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

    result_writer << insert_step.step_number << ",insert," << total_time_us/MS_TO_US << ",-1.0,-1.0,";
}

constexpr size_t DELETE_CHUNK_SIZE = 5000;
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

    result_writer << delete_step.step_number << ",delete," << total_time_us/MS_TO_US << ",-1.0,-1.0,";
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

    std::cout << "Level " << level << " Memory Consumption: Vectors - " << total_vectors << ", Buffer Capacity - " << buffer_capacity << ", Memory (GB) - " << total_memory_gb << std::endl;

    if(index->parent_ != nullptr) { 
        total_memory_gb += log_memory_stats(index->parent_, level + 1, result_writer);
    }

    return total_memory_gb;
}

constexpr float SCAN_PERCENTAGE_RANGE[2] = {0.1, 0.2};
constexpr size_t NUM_OPERATIONS_BETWEEN_MAINTEANCE = 1;
constexpr size_t NUM_TEST_OPERATIONS = 0;

#define RESULT_WRITE_PATH "../scripts/big_ann_perf_numbers/scan_using_batching_250_aps_recall_0.9_search_0.2_0.1.csv"

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
    result_writer << "step_num,step_type,latency_ms,recall_mean,recall_std_dev,mainteance_ms,index_mem_gb,num_partitions" << std::endl;

    std::shared_ptr<QuakeIndex> index = build_index(steps_arr[0]); 
    std::cout << "Initialized Quake Index " << std::endl;

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
        if(curr_step.step_number % NUM_OPERATIONS_BETWEEN_MAINTEANCE == 0) {
            std::cout << std::endl;
            perform_mainteance(index, result_writer);
        } else { 
            result_writer << "0.0,";
        }

        std::cout << std::endl;
        float index_memory_gb = log_memory_stats(index, 0, result_writer);
        result_writer << index_memory_gb << "," << index->nlist() << std::endl;
        std::cout << "------ FINISH: Step " << curr_step.step_number << " of type " << curr_step.type << " ------\n" << std::endl;

        curr_scan_percentage -= scan_percentage_step;
    }
    

    result_writer.close();

    return 0;
}