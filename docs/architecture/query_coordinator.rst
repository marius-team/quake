.. _query_coordinator:

Query Coordinator
=====================================

Overview
--------
This document describes the NUMA‑aware design for the QueryCoordinator. In this design, the coordinator distributes query work to worker threads that scan index partitions in parallel. The design focuses on reducing memory latency by using NUMA‑aware allocation and thread affinity, minimizing synchronization overhead via per‑core resource pools and job queues, and uses adaptive partition scanning (APS) to enable early termination.

The design aims to:

- Minimize memory latency by pinning data (e.g. top‑K buffers, partitions) to the same cores that process them.
- Reduce synchronization overhead by using per‑core resource pools and per‑core job queues.
- Achieve nearly linear scalability with the number of cores while matching the serial scan performance for a single worker.
- Use APS (adaptive early termination) to terminate scanning early when the recall target is met.

Coordinator Architecture Diagram
----------------------------------

.. mermaid::

   %%{
     init: {
       "theme": "default",
       "themeVariables": {
         "fontSize": "20px",
         "fontFamily": "Arial"
         "fontWeight": "bold"
       }
     }
   }%%

   flowchart LR

   %% External inputs/nodes
   QIN["Input Query Vectors"]
   PSCAN["Partition IDs"]
   R["Final Results"]

   %% The main QueryCoordinator subgraph
   subgraph QC["QueryCoordinator"]
       direction LR

       %% Core 1
       subgraph Core1["Core 1"]
           direction TB
           JQ1[["Job Queue"]]
           QBA1[/Query Buffers/]
           IP1["Index Partitions"]
           WT1("Scan Thread")
           LTK1(("Local TopK"))
       end

       %% Core n
       subgraph CoreN["Core n"]
           direction TB
           JQN[["Job Queue"]]
           QBAN[/Query Buffers/]
           IPN["Index Partitions"]
           WTN("Scan Thread")
           LTKn(("Local TopK"))
       end

       GTopk(("Global TopK"))

       %% APS module that checks the global buffer and signals early termination
       APS["Adaptive Partition Scanning (APS)"]
   end

   %% Edges from input queries to each core's query buffers
   QIN -->|memcpy| QBA1
   QIN -->|memcpy| QBAN

   PSCAN -->|enqueue| JQ1
   PSCAN -->|enqueue| JQN

   %% Inside each core, show the flow
   JQ1 -->|dequeue| WT1
   QBA1 -->|read| WT1
   IP1 -->|scan| WT1
   WT1 -->|write| LTK1

   JQN -->|dequeue| WTN
   QBAN -->|read| WTN
   IPN -->|scan| WTN
   WTN -->|write| LTKn

   %% Merge local topK into global topK
   LTK1 -->|merge| GTopk
   LTKn -->|merge| GTopk

   %% APS periodically checks global buffer for recall progress
   GTopk <-->|check recall| APS
   APS -->|signal early termination| GTopk

   %% Finally, global topk to results
   GTopk -->|return| R

   %% Optional styling for clarity
   style QC fill:#fff7e6,stroke:#666,stroke-width:8px;
   style QIN fill:#ccf,stroke:#333,stroke-width:1px;
   style R fill:#ffecb3,stroke:#333,stroke-width:1px;
   style Core1 fill:#fff,stroke:#999,stroke-width:1px;
   style CoreN fill:#fff,stroke:#999,stroke-width:1px;
   style LTK1 fill:#eef,stroke:#333,stroke-width:1px,stroke-dasharray:2 2;
   style LTKn fill:#eef,stroke:#333,stroke-width:1px,stroke-dasharray:2 2;
   style GTopk fill:#eef,stroke:#333,stroke-width:1px,stroke-dasharray:2 2;
   style APS fill:#fff,stroke:#999,stroke-width:1px,stroke-dasharray:3 3;


Key Components
--------------
- **QueryCoordinator**
  The main class that distributes query work, manages worker threads, and merges local results from all cores into a final search result.

- **CoreResources**
  A per‑core structure that contains:

  - A pool of preallocated Top‑K buffers that are allocated using NUMA‑aware routines and pinned to local memory.
  - A local aggregator (query buffer) to collect intermediate results.
  - A dedicated job queue that holds scan jobs for that core.

- **ScanJob Structure**
  Each unit of work (a ScanJob) encapsulates:

  - Whether the job is batched or single‑query.
  - The partition ID (which is pinned to a specific core).
  - The number of neighbors (``k``) to return.
  - A pointer to the query vector(s).
  - Global query IDs and, for batched jobs, the number of queries.

- **Global Aggregator**
  A coordinator-managed Top‑K buffer that merges per‑core local aggregators to produce the final search result.

Workflow and Job Distribution
-------------------------------
1. **Distribute Partitions to Cores:**
   The PartitionManager assigns partitions to cores based on partition size. Each partition’s memory is allocated on the correct NUMA node using NUMA‑aware routines.

2. **Per‑Core Job Queues:**
   The QueryCoordinator creates a per‑core job queue inside each CoreResources structure. For each partition local to a core, a ScanJob is created (either for single-query or batched queries) and enqueued into that core’s job queue.

3. **Worker Processing:**
   Each worker thread (one per core) executes a stateless worker function that:

   - Sets affinity to the core it belongs to.
   - Dequeues jobs from its core’s job queue.
   - Processes each job (invoking the appropriate scan function).
   - Merges results into the core’s local aggregator.
   - Decrements a global atomic counter (or per‑core counter) and signals a condition variable for global coordination.

4. **Global Aggregation:**
   The coordinator periodically merges local aggregators into a global Top‑K buffer.

5. **APS and Early Termination:**
   The APS module periodically checks the global Top‑K buffer to determine if the recall target has been met. If so, it signals worker threads to stop scanning.
