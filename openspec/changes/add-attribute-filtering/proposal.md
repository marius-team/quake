# Change: Add Attribute Filtering Support

## Why

Quake currently supports pure vector similarity search but lacks the ability to filter results based on metadata attributes. Real-world applications frequently need to combine vector similarity with attribute predicates (e.g., "find similar products with price < $100" or "find similar documents from the last 30 days"). Without attribute filtering, users must either:
1. Over-fetch results and filter client-side (wasteful and slow)
2. Maintain separate indexes per attribute combination (explosion of indexes)
3. Use a different vector database entirely

Adding attribute support — including partition-level pruning and attribute-aware splits (relational cuts) — will make Quake competitive with production vector databases and enable the H-Quake hybrid index design.

## What Changes

Work is organized into three milestones, each independently shippable.

### M1: Bitmap Scan
Add an optional byte-mask parameter to partition scan paths so distance computation can be skipped for masked-out rows. This is the foundational primitive all filtering builds on.
- Modify scan functions in `list_scanning.h` and `QueryCoordinator` to accept `uint8_t* mask` (byte-mask: one byte per row, 1=include, 0=skip)
- All scan modes supported (serial, batched, worker)
- Zero overhead when mask is nullptr (no filter)
- No API changes, no new dependencies

### M2: DuckDB Attribute Store + Filtering
Full attribute support: storage, filtering, pruning, persistence, and Python API.

**Core Architecture:**
- Add `AttributeStore` component wrapping an in-memory DuckDB database (single `attributes` table with `partition_id` column + `partition_stats` table for bounds)
- DuckDB is the authoritative store for schema, types, persistence, and partition-level statistics
- NUMA-local raw typed arrays (`attr_data_`) in `IndexPartition` for scan-time filter evaluation (preserves NUMA locality on hot path)
- All mutations wrapped in DuckDB transactions for consistency

**Search Path:**
- Add partition-level pruning step: query `partition_stats` bounds to skip partitions whose attribute ranges don't overlap the query filter (before centroid routing)
- Add row-level filtered scan: evaluate filter predicates on NUMA-local `attr_columns_` arrays, generate byte-mask, pass to bitmap scan (M1)
- No filter = existing path, zero overhead

**Filter API:**
- Structured filter objects (`Eq`, `Lt`, `And`, `In`, `IdIn`, etc.) that render to SQL for DuckDB
- Support all DuckDB-native types (INT, FLOAT, VARCHAR, BOOL, DATE, etc.)
- ID-based filters (`IdIn`, `IdNotIn`) for restricting/excluding specific vectors
- Input attributes as pandas DataFrame or PyArrow table

**Persistence:**
- DuckDB database saved as `attributes.duckdb` alongside existing index files
- In-memory during operation, checkpointed on save
- On load, NUMA-local arrays materialized from DuckDB onto correct NUMA nodes
- Backward compatible: indexes without `attributes.duckdb` load normally

### M3: Relational Cuts
Extend maintenance to support attribute-aware partition splits.
- Extend maintenance policy to support relational cuts alongside spatial cuts (k-means)
- Relational cut: split partition on attribute threshold, same centroid, tightened bounds
- Evaluate both spatial and relational split profit; execute the better one
- V1 uses data-only heuristic for split-point selection (median of widest-range attribute)

## Impact

- **Affected code:**
  - `IndexPartition` — NUMA-local attribute column arrays (`attr_data_`)
  - `DynamicInvertedLists` — attribute operations on add/remove
  - `PartitionManager` — relational cuts, attribute propagation on splits
  - `QueryCoordinator` — partition pruning step, filtered scan
  - `MaintenancePolicy` — relational cut evaluation alongside spatial cuts
  - `QuakeIndex` — API surface (build/add/remove/search with attributes/filters)
  - `common.h` — `SearchParams` filter field, `FilterExpr` struct
  - Python bindings (`wrap.cpp`, `quake.py`) — filter objects, DataFrame input
  - New: `AttributeStore` class wrapping DuckDB

- **New dependency:** DuckDB C++ library (~20MB)

- **Performance implications:**
  - Memory: Dual storage (DuckDB + NUMA arrays) for attributes. Small relative to vectors (~1-3%).
  - Unfiltered search: Zero overhead (DuckDB not invoked)
  - Filtered search: Partition pruning reduces centroid routing cost. Row-level filtering reduces distance computations. Net positive for selective queries.
  - Build time: Minimal impact (DuckDB insert is fast for bulk data)

- **Backward compatibility:**
  - Indexes built without attributes continue to work
  - Attribute filtering is optional; default behavior unchanged
  - Loading old indexes (no `attributes.duckdb`) succeeds with empty schema

## Open Questions

1. **Relational cut split-point strategy**: V1 uses data-only heuristic. Workload-aware strategies (query reservoir, filter frequency counters, selectivity sketches) deferred — see design.md Open Questions.
2. **String filtering on NUMA arrays**: V1 may fall back to DuckDB for string predicates. Hash-based optimization deferred.
3. **DuckDB version pinning**: On-disk format may change between DuckDB versions. Need version tag or portable export strategy.
