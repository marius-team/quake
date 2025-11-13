import quake
import torch

def mainteance_test():
    # Build the index
    code_size = 128
    build_params = quake.IndexBuildParams()
    build_params.metric = "l2"
    build_params.nlist = 1024
    build_params.num_workers = 8
    build_params.niter = 25

    parent_params = quake.IndexBuildParams()
    parent_params.nlist = 1
    parent_params.metric = "l2"
    parent_params.num_workers = 0
    build_params.parent_params = parent_params

    index = quake.QuakeIndex()
    num_initial_vectors = 10000
    insert_vectors = torch.randn(num_initial_vectors, code_size).to(torch.float32)
    insert_ids = torch.arange(num_initial_vectors).to(torch.int64)
    index.build(insert_vectors, insert_ids, build_params)
    print("Built index with", num_initial_vectors, "vectors")

    # Set the mainteance policy
    m_params = quake.MaintenancePolicyParams()
    m_params.window_size = 5000
    m_params.split_threshold_ns = 2100
    m_params.split_knn_iterations = 10
    m_params.delete_threshold_ns = 2750
    m_params.partition_reduction_threshold = 0.45
    m_params.refinement_radius = 0
    m_params.refinement_iterations = 5
    m_params.min_partition_size = 1024
    m_params.enable_split_rejection = True
    m_params.enable_delete_rejection = True
    index.initialize_maintenance_policy(m_params)
    print("Initialized mainteance policy")

    # Run a batch of queries against the index
    search_fraction = 0.1

    search_params = quake.SearchParams()
    search_params.k = 10
    search_params.nprobe = int(search_fraction * index.nlist())
    search_params.recall_target = -1.0
    search_params.batched_scan = True
    search_params.batch_size = 500
    search_params.track_hits = True

    query_vectors = torch.randn(2500, code_size).to(torch.float32)
    index.search(query_vectors, search_params)
    print("Finished a search")

    # Now run mainteance
    mainteance_result = index.maintenance()
    print("Finished mainteance in", mainteance_result.total_time_us, "us")

    # Now try to add some vectors into the index
    num_add_vectors = 5000
    add_vectors = torch.randn(num_add_vectors, code_size).to(torch.float32)
    add_ids = torch.arange(num_initial_vectors, num_initial_vectors + num_add_vectors).to(torch.int64)
    index.add(add_vectors, add_ids)
    print("Added", num_add_vectors, "vectors to index")

if __name__ == "__main__":
    quake.IndexPartition.delete_resize_threshold = 0.8
    quake.IndexPartition.capacity_resize_threshold = 1.2
    
    mainteance_test()