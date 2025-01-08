//
// Created by Jason on 12/23/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#include <quake_index.h>
#include <clustering.h>

QuakeIndex::QuakeIndex(int current_level) {
    // Initialize the QuakeIndex
    parent_ = nullptr;
    partition_manager_ = nullptr;
    query_coordinator_ = nullptr;
    build_params_ = nullptr;
    maintenance_policy_params_ = nullptr;
    current_level_ = current_level;
}

QuakeIndex::~QuakeIndex() {
    parent_ = nullptr;
    partition_manager_ = nullptr;
    query_coordinator_ = nullptr;
    build_params_ = nullptr;
    maintenance_policy_params_ = nullptr;
}

shared_ptr<BuildTimingInfo> QuakeIndex::build(Tensor x, Tensor ids, shared_ptr<IndexBuildParams> build_params) {
    build_params_ = build_params;
    metric_ = build_params_->metric;

    x = x.contiguous().clone();
    ids = ids.contiguous();

    shared_ptr<BuildTimingInfo> timing_info = make_shared<BuildTimingInfo>();
    timing_info->n_vectors = x.size(0);
    timing_info->d = x.size(1);

    auto start_time = std::chrono::high_resolution_clock::now();
    if (build_params_->nlist > 1) {
        auto train_start = std::chrono::high_resolution_clock::now();
        shared_ptr<Clustering> clustering = kmeans(
            x,
            ids,
            build_params_->nlist,
            build_params_->metric,
            build_params_->niter
        );
        auto train_end = std::chrono::high_resolution_clock::now();

        auto add_start = std::chrono::high_resolution_clock::now();

        // create parent index over the centroids, assume is flat for now
        parent_ = make_shared<QuakeIndex>(current_level_ + 1);
        auto parent_build_params = make_shared<IndexBuildParams>();
        parent_build_params->metric = metric_;
        parent_->build(clustering->centroids, clustering->partition_ids, parent_build_params);

        // initialize the partition manager
        partition_manager_ = make_shared<PartitionManager>();
        partition_manager_->init_partitions(parent_, clustering);

        auto add_end = std::chrono::high_resolution_clock::now();
    } else {
        // flat index
        partition_manager_ = make_shared<PartitionManager>();

        shared_ptr<Clustering> clustering = make_shared<Clustering>();
        clustering->partition_ids = torch::tensor({0}, torch::kInt64);
        clustering->centroids = x.mean(0, true);
        clustering->vectors = {x};
        clustering->vector_ids = {ids};

        partition_manager_->init_partitions(parent_, clustering);
    }

    // create query coordinator
    query_coordinator_ = make_shared<QueryCoordinator>(parent_, partition_manager_, metric_, build_params_->num_workers);

    return timing_info;
}


shared_ptr<SearchResult>
QuakeIndex::search(Tensor x, shared_ptr<SearchParams> search_params) {
    if (!query_coordinator_) {
        throw std::runtime_error("[QuakeIndex::search()] No query coordinator. Did you build the index?");
    }
    return query_coordinator_->search(x, search_params);
}

Tensor QuakeIndex::get(Tensor ids) {
    if (!partition_manager_) {
        throw std::runtime_error("[QuakeIndex::get()] No partition manager. Index not built?");
    }
    return partition_manager_->get(ids);
}

shared_ptr<ModifyTimingInfo> QuakeIndex::add(Tensor x, Tensor ids) {
    if (!partition_manager_) {
        throw std::runtime_error("[QuakeIndex::add()] No partition manager. Build the index first.");
    }

    auto modify_info = std::make_shared<ModifyTimingInfo>();
    auto start_time = std::chrono::high_resolution_clock::now();

    partition_manager_->add(x, ids);

    auto end_time = std::chrono::high_resolution_clock::now();
    modify_info->n_vectors = x.size(0);
    modify_info->modify_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    return modify_info;
}

shared_ptr<ModifyTimingInfo> QuakeIndex::remove(Tensor ids) {
    if (!partition_manager_) {
        throw std::runtime_error("[QuakeIndex::remove()] No partition manager. Build the index first.");
    }

    auto modify_info = std::make_shared<ModifyTimingInfo>();
    auto start_time = std::chrono::high_resolution_clock::now();

    partition_manager_->remove(ids);

    auto end_time = std::chrono::high_resolution_clock::now();
    modify_info->n_vectors = ids.size(0);
    modify_info->modify_time_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    return modify_info;
}

shared_ptr<ModifyTimingInfo> QuakeIndex::modify(Tensor ids, Tensor x) {
    partition_manager_->remove(ids);
    return add(x, ids);
}


void QuakeIndex::initialize_maintenance_policy(shared_ptr<MaintenancePolicyParams> maintenance_policy_params) {
    maintenance_policy_params_ = maintenance_policy_params;
    maintenance_policy_ = make_shared<QueryCostMaintenance>(partition_manager_, maintenance_policy_params);
}

shared_ptr<MaintenanceTimingInfo> QuakeIndex::maintenance() {
    if (!maintenance_policy_) {
        throw std::runtime_error("[QuakeIndex::maintenance()] No maintenance policy set.");
    }

    return maintenance_policy_->maintenance();
}


void QuakeIndex::save(const std::string& dir_path) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir_path)) {
        fs::create_directories(dir_path);
    } else if (!fs::is_directory(dir_path)) {
        throw std::runtime_error("save path exists but is not a directory: " + dir_path);
    }

    // 2. Write metadata (metric, level)
    {
        std::string meta_file = (fs::path(dir_path) / "metadata.txt").string();
        std::ofstream ofs(meta_file);
        if (!ofs.is_open()) {
            throw std::runtime_error("Cannot open metadata file for writing: " + meta_file);
        }
        ofs << "metric=" << static_cast<int>(metric_) << "\n";
        ofs << "level=" << current_level_ << "\n";
        ofs << "ntotal=" << ntotal() << "\n";
        ofs << "nlist=" << nlist() << "\n";

        ofs.close();
    }

    {
        std::string partitions_path = (fs::path(dir_path) / "partitions").string();
        partition_manager_->save(partitions_path);
    }

    // 4. If parent_ exists, recursively save it into a "parent" subdirectory
    if (parent_) {
        std::string parent_dir = (fs::path(dir_path) / "parent").string();
        parent_->save(parent_dir);
    }

    std::cout << "[QuakeIndex::save] Index saved to directory: " << dir_path << "\n";
}

void QuakeIndex::load(const std::string& dir_path) {
    namespace fs = std::filesystem;

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        throw std::runtime_error("Cannot load QuakeIndex, directory does not exist: " + dir_path);
    }

    std::cout << "[QuakeIndex::load] Loading index from directory: " << dir_path << "\n";

    // 1. Read metadata.txt
    {
        std::string meta_file = (fs::path(dir_path) / "metadata.txt").string();
        std::ifstream ifs(meta_file);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open metadata file for reading: " + meta_file);
        }
        std::string line;
        while (std::getline(ifs, line)) {
            // parse lines like "metric=2" or "level=3"
            auto pos = line.find('=');
            if (pos == std::string::npos) continue; // skip malformed
            std::string key = line.substr(0, pos);
            std::string val = line.substr(pos + 1);
            if (key == "metric") {
                int m = std::stoi(val);
                metric_ = static_cast<MetricType>(m);
            } else if (key == "level") {
                current_level_ = std::stoi(val);
            }
            // if more fields exist, parse them here similarly
        }
        ifs.close();
    }

    // 2. Create partition manager if needed, and load it
    {
        partition_manager_ = std::make_shared<PartitionManager>();
        std::string partitions_path = (fs::path(dir_path) / "partitions").string();
        partition_manager_->load(partitions_path);
    }

    // 3. Check if a "parent" subdirectory exists, load it if so
    {
        std::string parent_dir = (fs::path(dir_path) / "parent").string();
        if (fs::exists(parent_dir) && fs::is_directory(parent_dir)) {
            parent_ = std::make_shared<QuakeIndex>();
            parent_->load(parent_dir);
        } else {
            parent_ = nullptr;
        }
    }

    // 4. Create query coordinator
    query_coordinator_ = std::make_shared<QueryCoordinator>(parent_, partition_manager_, metric_);
}

int64_t QuakeIndex::ntotal() {
    if (partition_manager_) {
        return partition_manager_->ntotal();
    }
    return 0;
}

int64_t QuakeIndex::nlist() {
    if (partition_manager_) {
        return partition_manager_->nlist();
    }
    return 0;
}

int QuakeIndex::d() {
    if (partition_manager_) {
        return partition_manager_->d();
    }
    return 0;
}