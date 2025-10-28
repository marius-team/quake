from quake import MaintenancePolicyParams, IndexPartition
from quake.index_wrappers.quake import QuakeWrapper
import torch

def mainteance_test():
    # Build the index
    num_search_workers = 64
    num_merge_workers = 4
    code_size = 128
    nc = 1024
    use_numa = True
    num_initial_vectors = 10000
    insert_vectors = torch.randn(num_initial_vectors, code_size).to(torch.float32)
    insert_ids = torch.arange(num_initial_vectors).to(torch.int64)

    index = QuakeWrapper()
    index.build(
        insert_vectors,
        nc,
        metric="l2",
        ids=insert_ids,
        num_workers=num_search_workers,
        code_size=code_size,
        num_merge_workers=num_merge_workers,
        use_numa=use_numa
    )

    m_params = MaintenancePolicyParams()
    index.index.initialize_maintenance_policy(m_params)
    print("Built index with", num_initial_vectors, "vectors")

    # Run a batch of queries against the index
    k = 5
    nprobe = 32
    recall_target = 0.9
    num_queries = 2500
    query_vectors = torch.randn(2500, code_size).to(torch.float32)
    index.search(query_vectors, k=k, nprobe=nprobe, batched_scan=True, recall_target=recall_target)

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
    IndexPartition.delete_resize_threshold = 0.8
    IndexPartition.capacity_resize_threshold = 1.2
    print(IndexPartition.delete_resize_threshold, IndexPartition.capacity_resize_threshold)
    # mainteance_test()