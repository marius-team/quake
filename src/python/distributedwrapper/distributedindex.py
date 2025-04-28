from typing import Any, List, Dict, Optional, Tuple
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from quake import QuakeIndex, IndexBuildParams, SearchParams
from quake.distributedwrapper import distributed
from collections import defaultdict

class DistributedIndex:
    """
    A distributed version of QuakeIndex that supports multiple servers.
    Each server maintains a full copy of the index, and queries are distributed
    across servers for parallel processing.
    """
    
    def __init__(self, server_addresses: List[str], num_partitions: int, build_params_kw_args: Dict[str, Any], search_params_kw_args: Dict[str, Any]):
        """
        Initialize the DistributedIndex with a list of server addresses.
        
        Args:
            server_addresses: List of server addresses in the format "host:port"
        """
        if not server_addresses:
            raise ValueError("At least one server address must be provided")
            
        self.server_addresses = server_addresses
        self.build_params_kw_args = build_params_kw_args
        self.search_params_kw_args = search_params_kw_args

        self.build_params: List[IndexBuildParams] = []
        self._initialize_build_params()

        self.indices: List[QuakeIndex] = []
        self._initialize_indices()

        self.search_params: List[SearchParams] = []
        self._initialize_search_params()

        self.k = self.search_params_kw_args["k"]

        # TODO if there are leftover servers, replicate most commonly accessed partitions
        assert len(self.server_addresses) % num_partitions == 0, "Number of servers must be divisible by number of partitions"

        self.num_partitions = num_partitions

    def _initialize_build_params(self):
        """Initialize IndexBuildParams instances for each server."""
        for address in self.server_addresses:
            params = distributed(IndexBuildParams, address)
            params.import_module(package="quake", item="IndexBuildParams")
            params.instantiate()
            params.nlist = self.build_params_kw_args["nlist"]
            params.metric = self.build_params_kw_args["metric"]
            self.build_params.append(params)

    def _initialize_indices(self):
        """Initialize QuakeIndex instances for each server."""
        for address in self.server_addresses:
            index = distributed(QuakeIndex, address)
            index.import_module(package="quake", item="QuakeIndex")
            index.register_function("build")
            index.register_function("search")
            index.register_function("add")
            index.register_function("remove")
            index.instantiate()
            self.indices.append(index)
    
    def _initialize_search_params(self):
        """Initialize SearchParams instances for each server."""
        for address in self.server_addresses:
            params = distributed(SearchParams, address)
            params.import_module(package="quake", item="SearchParams")
            params.instantiate()
            params.k = self.search_params_kw_args["k"]
            params.nprobe = self.search_params_kw_args["nprobe"]
            self.search_params.append(params)
    
    def _prepartition_vectors(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Prepartition the vectors and ids into num_partitions.
        
        Args:
            vectors: Tensor of vectors to partition
            ids: Tensor of vector IDs to partition
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of partitioned vectors and IDs
        """
            
        # Calculate partition sizes
        total_size = vectors.size(0)
        base_size = total_size // self.num_partitions
        remainder = total_size % self.num_partitions
        
        partitioned_vectors = []
        partitioned_ids = []
        
        start_idx = 0
        for i in range(self.num_partitions):
            # Calculate size for this partition
            partition_size = base_size + (1 if i < remainder else 0)
            
            # Slice vectors and ids
            end_idx = start_idx + partition_size
            partitioned_vectors.append(vectors[start_idx:end_idx])
            partitioned_ids.append(ids[start_idx:end_idx])
            
            start_idx = end_idx
            
        return partitioned_vectors, partitioned_ids

    def build(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Build the index on all servers. Each server gets a full copy of the index.
        
        Args:
            vectors: Tensor of vectors to index
            ids: Tensor of vector IDs
            build_params: Parameters for building the index
        """
        if vectors.size(0) != ids.size(0):
            raise ValueError("Number of vectors must match number of IDs")

        self.partition_to_server_map = defaultdict(list)

        partitioned_vectors, partitioned_ids = self._prepartition_vectors(vectors, ids)

        assert len(partitioned_vectors) == self.num_partitions, "Number of partitioned vectors must match number of partitions"

        # Create build_params for each server
        for i in range(len(self.server_addresses)):
            # Build the index on each server
            partition_idx = i % self.num_partitions
            self.indices[i].build(partitioned_vectors[partition_idx], partitioned_ids[partition_idx], self.build_params[i])
            self.partition_to_server_map[partition_idx].append(self.server_addresses[i])

        print("Partition to server map:")
        print(self.partition_to_server_map)

        
    def get_index_and_params(self, server_address: str):
        """
        Get the index and params for a given server address.
        """
        for i in range(len(self.server_addresses)):
            if self.server_addresses[i] == server_address:
                return self.indices[i], self.build_params[i], self.search_params[i]
            
    def _search_single_server(self, server_idx: int, queries: torch.Tensor) -> torch.Tensor:
        """Helper method to perform search on a single server."""
        return self.indices[server_idx].search(queries, self.search_params[server_idx])

    def _search_single_server_dist(self, server_address: str, queries: torch.Tensor) -> torch.Tensor:
        """Helper method to perform search on a single server."""
        index, _, search_params = self.get_index_and_params(server_address)
        return index.search(queries, search_params)

    def search(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Distribute queries across servers in parallel and merge results.
        
        Args:
            queries: Tensor of query vectors
            
        Returns:
            Search results from all servers merged and sorted
        """

        
        # Distribute queries across servers
        n_servers = len(self.server_addresses)
        n_queries = queries.size(0)
        
        # Calculate how many queries each server should handle
        queries_per_server = n_queries // n_servers
        remainder = n_queries % n_servers
        
        # Split queries among servers
        start_idx = 0
        futures = []
        
        with ThreadPoolExecutor(max_workers=n_servers) as executor:
            for i in range(n_servers):
                # Calculate number of queries for this server
                n_queries_for_server = queries_per_server + (1 if i < remainder else 0)
                if n_queries_for_server == 0:
                    continue
                    
                # Get queries for this server
                end_idx = start_idx + n_queries_for_server
                server_queries = queries[start_idx:end_idx]
                
                # Submit search task to thread pool
                future = executor.submit(self._search_single_server, i, server_queries)
                futures.append(future)
                start_idx = end_idx

            # Collect results as they complete
            results = [future.result() for future in futures]
        
        # Merge results
        return self._merge_search_results(results)

    def search_dist(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Distribute queries across servers in parallel and merge results.
        
        Args:
            queries: Tensor of query vectors
            
        Returns:
            Search results from all servers merged and sorted
        """
        # Distribute queries across servers
        n_servers = len(self.server_addresses)
        num_replicas = n_servers // self.num_partitions
        
        # Split queries among servers
        results_list = []
        
        with ThreadPoolExecutor(max_workers=n_servers) as executor:
            # Calculate base batch size and remainder
            num_queries = len(queries)
            base_batch_size = num_queries // num_replicas
            remainder = num_queries % num_replicas
            
            for i in range(num_replicas):
                futures = []
                # Calculate batch size for this partition
                batch_size = base_batch_size + (1 if i < remainder else 0)
                if batch_size == 0:
                    continue
                    
                # Get queries for this partition
                start_idx = i * base_batch_size + min(i, remainder)
                end_idx = start_idx + batch_size
                queries_for_partition_i = queries[start_idx:end_idx]
                
                # Submit to all servers handling this partition
                servers_to_submit = [value[i] for value in self.partition_to_server_map.values()]
                for server in servers_to_submit:
                    future = executor.submit(self._search_single_server_dist, server, queries_for_partition_i)
                    futures.append(future)
                results = [future.result() for future in futures]
                results_list.append(results)
        
        # Merge results
        final_results = self._merge_search_results_dist(results_list)
        return final_results

    def search_sync(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Synchronous wrapper for the async search method.
        """
        return asyncio.run(self.search(queries))

    def add(self, vectors: torch.Tensor, ids: torch.Tensor):
        """
        Add vectors to all servers' indices.
        
        Args:
            vectors: Tensor of vectors to add
            ids: Tensor of vector IDs
        """
        for index in self.indices:
            index.add(vectors, ids)
        
    def remove(self, ids: torch.Tensor):
        """
        Remove vectors from all servers' indices.
        
        Args:
            ids: Tensor of vector IDs to remove
        """
        for index in self.indices:
            index.remove(ids)
        
    def _merge_search_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge search results from multiple servers.
        Since each server handled a different subset of queries, we just concatenate the results.
        
        Args:
            results: A list of type distributedwrapper.rwrapper.Local, we can obtain the tensors from the ids
            
        Returns:
            Concatenated search results
        """
        # 
        ids = [result.ids for result in results]
        ids = torch.cat(ids, dim=0)
        return ids

    def _merge_search_results_dist(self, results_list: List[List]) -> torch.Tensor:
        """
        Merge search results from multiple servers.
        Since each server handled a different subset of queries, we just concatenate the results.
        
        Args:
            results: A list of type distributedwrapper.rwrapper.Local, we can obtain the tensors from the ids
            
        Returns:
            Concatenated search results of shape (num_queries, k)
        """
        full_ids = []
        for i in range(len(results_list)):
            # Get all IDs and distances for this partition
            ids = [result.ids for result in results_list[i]]
            distances = [result.distances for result in results_list[i]]
            
            # Concatenate along the k dimension (dim=1)
            ids = torch.cat(ids, dim=1)  # shape: (num_queries, total_k)
            distances = torch.cat(distances, dim=1)  # shape: (num_queries, total_k)
            
            # Sort by distances and get top k
            sorted_indices = torch.argsort(distances, dim=1)
            sorted_ids = torch.gather(ids, 1, sorted_indices)
            
            # Take top k results
            top_k_ids = sorted_ids[:, :self.k]
            full_ids.append(top_k_ids)
            
        # Concatenate results from all partitions
        final_ids = torch.cat(full_ids, dim=0)
        return final_ids
