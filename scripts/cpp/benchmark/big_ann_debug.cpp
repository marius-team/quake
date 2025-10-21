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

#include <torch/script.h>    // For c10::IValue
#include <torch/serialize.h> // For torch::load
#include "quake_index.h"  // Quake API header

using torch::Tensor;

constexpr const char* DIRECTORY_PATH = "/home/devesh/big-ann-benchmark/big-ann-benchmarks/data/MSTuring-30M-clustered/index_arguments/";
constexpr const bool RUN_MAINTEANCE = true;
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

std::shared_ptr<QuakeIndex> build_index(Step& build_step) { 
    // Verify step
    if(build_step.step_number != 1 || build_step.type != "insert") { 
        std::cerr << "Build Index called with step num " << build_step.step_number << " and type " << build_step.type << std::endl;
        exit(1);
    }

    // Create and build the index
    std::shared_ptr<QuakeIndex> index = std::make_shared<QuakeIndex>();

    std::shared_ptr<IndexBuildParams> build_params = std::make_shared<IndexBuildParams>();
    build_params->nlist = 8192;
    build_params->num_workers = 32;
    build_params->metric = "l2";
    build_params->use_numa = true;
    build_params->num_merge_workers = 1;

    /*
    build_params->parent_params = std::make_shared<IndexBuildParams>();
    build_params->parent_params->metric = "l2";
    build_params->parent_params->nlist = 256;
    build_params->parent_params->num_workers = 4;
    build_params->parent_params->use_numa = true;
    */

    Tensor build_vectors = load_tensor(build_step.vectors_path).to(torch::kFloat32);
    Tensor build_ids = load_tensor(build_step.ids_path).to(torch::kInt64);
    index->build(build_vectors, build_ids, build_params);

    // Also initialize with the default mainteance policy
    std::shared_ptr<MaintenancePolicyParams> mainteance_policy = std::make_shared<MaintenancePolicyParams>();
    mainteance_policy->window_size = 5000;
    mainteance_policy->split_threshold_ns = 100;
    mainteance_policy->delete_threshold_ns = 5000;
    mainteance_policy->refinement_radius = 500;
    mainteance_policy->refinement_iterations = 3;
    mainteance_policy->min_partition_size = 64;
    mainteance_policy->enable_split_rejection = true;
    mainteance_policy->enable_delete_rejection = true;
    index->initialize_maintenance_policy(mainteance_policy);

    return index;
}

void perform_mainteance(std::shared_ptr<QuakeIndex> index) { 
    if(RUN_MAINTEANCE) { 
        std::shared_ptr<MaintenanceTimingInfo> result = index->maintenance();
        std::cout << "Mainteance Metrics: Num Splits - " << result->n_splits << ", Num Deletes - " << result->n_deletes << ", Delete Time ms - " << result->delete_time_us/MS_TO_US;
        std::cout << ", Split Time ms - " << result->split_time_us/MS_TO_US << ", Refinement Time - " << result->refinement_time_us/MS_TO_US << std::endl;
    }
}

void print_search_metrics(std::shared_ptr<SearchTimingInfo> timing_info, int level) { 
    std::cout << "Search Level " << level << " Breakdown: Total Time ms - " << timing_info->total_time_ns/MS_TO_NS << ", Main Scan Time ms - " << timing_info->scan_time_ns/MS_TO_NS;
    std::cout << ", Worker Scan Time ms - " << timing_info->worker_scan_time_ns/MS_TO_NS << ", Nlist - " << timing_info->n_clusters;
    std::cout << ", Partitions Scanned - " << timing_info->partitions_scanned << ", N Queries - " << timing_info->n_queries << std::endl;
}

void perform_search(std::shared_ptr<QuakeIndex> index, Step& search_step) { 
    // Load the search query
    Tensor search_queries = load_tensor(search_step.vectors_path).to(torch::kFloat32);

    // Create the search parameters
    std::shared_ptr<SearchParams> search_params = std::make_shared<SearchParams>();
    search_params->k = 10;
    search_params->nprobe = 32;
    search_params->recall_target = 0.9;
    search_params->batched_scan = true;
    search_params->batch_size = 1000;

    /*
    search_params->parent_params = std::make_shared<SearchParams>();
    search_params->parent_params->nprobe = 8;
    search_params->parent_params->recall_target = -1.0;
    search_params->parent_params->batched_scan = true;
    */

    // Run the search
    std::shared_ptr<SearchResult> search_result = index->search(search_queries, search_params);
    print_search_metrics(search_result->timing_info, 0);
    // print_search_metrics(search_result->timing_info->parent_info, 1);
}

void perform_insert(std::shared_ptr<QuakeIndex> index, Step& insert_step) { 
    // Load the arguments
    Tensor insert_vectors = load_tensor(insert_step.vectors_path).to(torch::kFloat32);
    Tensor insert_ids = load_tensor(insert_step.ids_path).to(torch::kInt64);
    index->add(insert_vectors, insert_ids);
}

void perform_delete(std::shared_ptr<QuakeIndex> index, Step& delete_step) { 
    Tensor delete_ids = load_tensor(delete_step.ids_path).to(torch::kInt64);
    index->remove(delete_ids);
}

int main() { 
    // Extract the steps
    std::vector<Step> steps_arr = extract_step_details();
    std::cout << "Loaded " << steps_arr.size() << " steps" << std::endl;
    assert(!steps_arr.empty());

    // Build the initial index
    std::shared_ptr<QuakeIndex> index = build_index(steps_arr[0]); 
    std::cout << "Initialized index" << std::endl;

    // Now perform the streaming operations against the index
    size_t num_test_operations = 20; // steps_arr.size();
    for(size_t i = 1; i <= num_test_operations; i++) { 
        // Run the step
        Step& curr_step = steps_arr[i];
        std::cout << "\n------ START: Step " << curr_step.step_number << " of type " << curr_step.type << " -----" << std::endl;
        if(curr_step.type == "search") { 
            perform_search(index, curr_step);
        } else if(curr_step.type == "insert") { 
            perform_insert(index, curr_step);
        } else if(curr_step.type == "delete") { 
            perform_delete(index, curr_step);
        } else { 
            std::cerr << "Unsupport step type of " << curr_step.type << std::endl;
            exit(1);
        }

        // Run the mainteance
        std::cout << std::endl;
        perform_mainteance(index);
        std::cout << "------ FINISH: Step " << curr_step.step_number << " of type " << curr_step.type << " ------\n" << std::endl;

    }

    return 0;
}