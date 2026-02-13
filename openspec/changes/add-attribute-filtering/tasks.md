# Milestone 1: Bitmap Scan

Add optional byte-mask support to partition scan paths. A byte-mask is a `uint8_t*` array with one byte per row (1=include, 0=skip). When provided, skip distance computation for masked-out rows. This is the foundational primitive that all filtering builds on.

**Done when:** All scan modes (serial, batched, worker) support byte-mask. Unit tests pass with random masks. Benchmark confirms expected speedup (e.g., 10% selectivity → ~10x fewer distance ops). No API changes, no new dependencies.

## 1. Scan Path Changes
- [ ] 1.1 Add optional `uint8_t* mask` parameter to scan functions in `list_scanning.h`
- [ ] 1.2 Implement masked scan for serial scan path (skip distance computation when `mask[i] == 0`)
- [ ] 1.3 Implement masked scan for BLAS-accelerated batch distance path
- [ ] 1.4 Implement masked scan for batched scan in `QueryCoordinator`
- [ ] 1.5 Implement masked scan for worker scan in `QueryCoordinator`
- [ ] 1.6 Ensure unmasked path (mask == nullptr) has zero overhead vs current code

## 2. Testing (M1)
- [ ] 2.1 Unit tests: masked scan correctness (random masks, verify only included rows contribute to top-k)
- [ ] 2.2 Unit tests: masked scan with all-ones mask matches unmasked results exactly
- [ ] 2.3 Unit tests: masked scan with all-zeros mask returns no results
- [ ] 2.4 Performance benchmark: serial scan with 10%, 50%, 90% selectivity masks
- [ ] 2.5 Performance benchmark: verify zero overhead when mask is nullptr

---

# Milestone 2: DuckDB Attribute Store + Filtering

Add DuckDB-backed attribute storage, NUMA-local attribute arrays, structured filter expressions, partition-level pruning, and end-to-end filtered search. Extends build/add/remove/search APIs with attribute support.

**Done when:** Can build index with pandas DataFrame or PyArrow table of attributes, search with filter objects (`Eq`, `Lt`, `And`, `IdIn`, etc.), save/load round-trips with attributes, unfiltered search has zero overhead, filtered search returns correct results at various selectivities.

## 3. DuckDB Integration & AttributeStore
- [ ] 3.1 Add DuckDB as a C++ dependency (CMakeLists.txt)
- [ ] 3.2 Implement `AttributeStore` class (DuckDB wrapper: create in-memory DB, schema init, transaction methods)
- [ ] 3.3 Implement `attributes` table management (single table with partition_id, row_id, dynamic columns)
- [ ] 3.4 Implement `partition_stats` table management (per-partition, per-attribute min/max/row_count)
- [ ] 3.5 Implement `insert_rows()`, `delete_rows()`, `update_partition_ids()` with transaction support
- [ ] 3.6 Implement `recompute_stats()` and `update_stats_on_insert()`
- [ ] 3.7 Implement `save()` / `load()` (DuckDB checkpoint to/from `attributes.duckdb`)

## 4. NUMA-Local Attribute Arrays in IndexPartition
- [ ] 4.1 Add `AttrColumn` struct and `attr_columns_` field to `IndexPartition`
- [ ] 4.2 Implement `allocate_attr_columns()` using `quake_alloc()` on partition's NUMA node
- [ ] 4.3 Implement `append_attrs()` (mirrors existing `append()` for codes/ids)
- [ ] 4.4 Implement `remove_attr()` with swap-last semantics (mirrors existing `remove()`)
- [ ] 4.5 Implement `free_attr_columns()` and integrate with destructor/move
- [ ] 4.6 Implement `materialize_partition_attrs()` in `AttributeStore` (read from DuckDB, allocate NUMA-local)

## 5. Filter Expression System
- [ ] 5.1 Define `FilterExpr` struct in C++ (Op enum, column, value, children, `to_sql()` method)
- [ ] 5.2 Implement Python filter objects (`Eq`, `Lt`, `Gt`, `Le`, `Ge`, `Ne`, `In`, `NotIn`, `And`, `Or`, `Not`, `IdIn`, `IdNotIn`, `IsNull`, `IsNotNull`)
- [ ] 5.3 Implement `to_sql()` rendering for each filter type
- [ ] 5.4 Implement scan-time filter evaluation on NUMA-local arrays → byte-mask generation (numeric predicates: typed comparison loops)
- [ ] 5.5 Implement scan-time compound filter evaluation (AND/OR byte-mask combination)
- [ ] 5.6 Implement ID filter evaluation against `IndexPartition.ids_` → byte-mask

## 6. Search Integration
- [ ] 6.1 Add filter field to `SearchParams` (`FilterExpr`)
- [ ] 6.2 Implement `get_candidate_partitions()` in `AttributeStore` (query partition_stats for bounds overlap)
- [ ] 6.3 Modify `QueryCoordinator` search flow: insert partition pruning step before centroid routing
- [ ] 6.4 Integrate filter evaluation → byte-mask → bitmap scan (from M1) in scan loop
- [ ] 6.5 Ensure unfiltered queries bypass all attribute logic (zero overhead)

## 7. Write Path
- [ ] 7.1 Extend `QuakeIndex::build()` to accept attributes (call AttributeStore, populate NUMA arrays)
- [ ] 7.2 Extend `QuakeIndex::add()` to accept attributes (DuckDB transaction + NUMA array append)
- [ ] 7.3 Extend `QuakeIndex::remove()` to delete from DuckDB + NUMA array swap-last
- [ ] 7.4 Handle attribute propagation during `DynamicInvertedLists` batch operations

## 8. Persistence & Backward Compat
- [ ] 8.1 Extend `QuakeIndex::save()` to call `AttributeStore::save()`
- [ ] 8.2 Extend `QuakeIndex::load()` to call `AttributeStore::load()` + materialize NUMA arrays
- [ ] 8.3 Handle missing `attributes.duckdb` on load (backward compat: empty schema, no filter support)

## 9. Python Bindings (M2)
- [ ] 9.1 Expose `FilterExpr` / filter objects to Python via pybind11
- [ ] 9.2 Accept pandas DataFrame or PyArrow table in `build()` / `add()` Python bindings
- [ ] 9.3 Update `QuakeWrapper` Python class (build/add with attributes, search with filter)
- [ ] 9.4 Add `wrap.cpp` bindings for `AttributeStore` and filter parameters

## 10. Testing (M2)
- [ ] 10.1 Unit tests: `AttributeStore` (CRUD, transactions, stats, persistence)
- [ ] 10.2 Unit tests: filter expression rendering (to_sql) and scan-time evaluation
- [ ] 10.3 Unit tests: NUMA-local attr_data_ lifecycle (alloc, append, remove, free)
- [ ] 10.4 Integration tests: filtered search correctness (various selectivities, filter types)
- [ ] 10.5 Integration tests: partition pruning (verify partitions skipped correctly)
- [ ] 10.6 Integration tests: persistence (save/load with attributes, backward compat)
- [ ] 10.7 Performance benchmark: unfiltered search overhead (must be zero)
- [ ] 10.8 Performance benchmark: filtered search at various selectivities

---

# Milestone 3: Relational Cuts & Maintenance

Extend the maintenance policy to support relational cuts (attribute-aware partition splits). The index adapts its structure to filter workloads over time.

**Done when:** Maintenance can produce relational splits. After relational splits, partition pruning correctly skips children with non-overlapping bounds. Integration test: build → insert filtered workload → maintenance → verify partition structure adapted and filtered search improved.

## 11. Relational Cuts
- [ ] 11.1 Add `RELATIONAL` split type to `PartitionManager::split_partitions()`
- [ ] 11.2 Implement data-only split-point selection (median of widest-range attribute)
- [ ] 11.3 Implement relational split execution (same centroid, tightened bounds, row redistribution)
- [ ] 11.4 Extend `MaintenancePolicy::perform_maintenance()` to evaluate relational vs spatial split profit
- [ ] 11.5 Update `partition_stats` on split (tightened bounds for children)
- [ ] 11.6 Rebuild NUMA-local `attr_data_` arrays for child partitions after split

## 12. Testing (M3)
- [ ] 12.1 Unit tests: relational split correctness (vectors in correct children, bounds tightened)
- [ ] 12.2 Integration tests: relational cuts (split, search, verify pruning works)
- [ ] 12.3 Integration tests: maintenance chooses relational vs spatial split correctly
- [ ] 12.4 Integration tests: end-to-end workload (build → queries → maintenance → improved filtered search)

## 13. Documentation
- [ ] 13.1 Update API documentation (new parameters, filter syntax)
- [ ] 13.2 Add attribute filtering usage guide
- [ ] 13.3 Update quickstart example with filtered search
