## Context

Quake is a high-performance dynamic ANN index using an IVF (inverted file) structure. Vectors are clustered into partitions, and search probes the most promising partitions. The system supports real-time add/remove operations, automatic cost-based maintenance (split/merge), and NUMA-aware memory allocation.

This design adds **attribute filtering** to Quake, inspired by the H-Quake hybrid index design. The key insight from H-Quake is that per-partition predicate bounds enable **partition-level pruning before centroid distance computation**, making selective filtered queries significantly faster than a naive post-filter approach.

**Current data flow:**
```
Query → Centroid routing (nprobe) → Scan partitions → Top-k
```

**Proposed data flow with attributes:**
```
Query + Filter
  → Partition pruning (bounds check, skip non-overlapping partitions)
  → Centroid routing (over surviving candidates)
  → Filtered scan (evaluate predicate, compute distance only for matches)
  → Top-k
```

**Stakeholders:**
- Users who need filtered vector search
- Maintenance policy (relational cuts: attribute-aware partition splits)
- Persistence layer (must handle attribute storage)
- NUMA subsystem (scan-time attribute data must be NUMA-local)

## Goals / Non-Goals

### Goals
- Support attribute filtering during search with structured filter operators
- Support all DuckDB-native attribute types (INT, FLOAT, VARCHAR, BOOL, DATE, etc.)
- Support ID-based filters (`IdIn`, `IdNotIn`)
- Partition-level pruning via per-partition attribute bounds
- Relational cuts: attribute-aware partition splits in the maintenance policy
- Preserve NUMA locality on the scan hot path
- Maintain backward compatibility with attribute-free indexes
- Zero overhead when attributes are not used

### Non-Goals
- Full-text search within string attributes
- Complex secondary indexes (B-trees, bloom filters) in v1
- APS (Adaptive Partition Selection) integration with filtering in v1
- Multi-vector per entity (document chunking)
- Workload-aware relational cut split-point selection in v1 (see Open Questions)

## Decisions

### Decision 1: DuckDB as Attribute Authority + NUMA-Local Arrays for Scan

**What:** Use DuckDB as the authoritative store for attribute data (schema, types, persistence, partition-level statistics, SQL filter parsing). Maintain a separate copy of attribute columns as NUMA-local raw arrays in `IndexPartition` for scan-time filter evaluation.

**Architecture:**

```
┌──────────────────────────────────────────────────────────┐
│                      QuakeIndex                          │
│  build(vectors, ids, attributes)                         │
│  search(queries, params, filter)                         │
│  add(vectors, ids, attributes)                           │
│  remove(ids)                                             │
├──────────────┬───────────────┬────────────┬──────────────┤
│ Partition    │ Query         │ Maintenance│ Attribute    │
│ Manager      │ Coordinator   │ Policy     │ Store        │
│              │               │            │ (DuckDB)     │
│ IndexPartition               │            │              │
│  codes_ (NUMA-local vectors) │            │ Single table │
│  ids_   (NUMA-local IDs)     │            │ partition_   │
│  attr_data_ (NUMA-local      │            │   stats      │
│    typed column arrays)      │            │              │
└──────────────┴───────────────┴────────────┴──────────────┘
```

**Two-level design:**

| Responsibility | Component | NUMA-aware? | When used |
|---|---|---|---|
| Schema, types, SQL parsing | DuckDB | No | Build, add, schema queries |
| Partition-level bounds/stats | DuckDB `partition_stats` table | No | Query-time partition pruning (Step 1) |
| Row-level filter evaluation | NUMA-local `attr_data_` arrays | Yes | Scan-time filtering (hot path) |
| Persistence | DuckDB checkpoint | No | Save/load |
| Authoritative data source | DuckDB `attributes` table | No | Load → populate NUMA arrays |

**Why this split:**

Quake's NUMA model allocates partition data (`codes_`, `ids_`) on specific NUMA nodes and pins worker threads to cores. The scan inner loop reads only from NUMA-local memory.

- If attributes lived only in DuckDB, scan-time filter evaluation would read from DuckDB-managed memory with no NUMA placement, breaking locality.
- Vector data per row is ~3,072 bytes (768 dims × 4B). Attribute data is ~40-100 bytes per row. Although attributes are small relative to vectors (~1-3%), the scan loop is latency-sensitive and NUMA cross-node penalties are 2-3x.
- By keeping raw typed arrays in `IndexPartition`, the filter evaluation loop is:
  ```cpp
  // Fully NUMA-local: attr_data_ allocated with quake_alloc() on partition's NUMA node
  for (int64_t i = 0; i < n; i++) {
      if (eval_filter(attr_data_, i, filter)) {
          compute_distance(query, codes_ + i * code_size, ...);
      }
  }
  ```
- DuckDB handles everything else: SQL parsing, type coercion, schema management, statistics, persistence.

**NUMA-local attribute arrays (`attr_data_`):**
- Stored in `IndexPartition` alongside `codes_` and `ids_`
- Allocated with `quake_alloc()` on the partition's NUMA node
- Format: one contiguous typed array per attribute column (same row ordering as `codes_`/`ids_`)
- Mutations (append, swap-last remove) mirror existing `codes_`/`ids_` patterns
- Populated from DuckDB on `load()`, kept in sync on `add()`/`remove()`

**Alternatives considered:**
- DuckDB only (no NUMA arrays): Simpler, but breaks NUMA locality on scan hot path. Cross-NUMA memory access adds 2-3x latency penalty on every filter evaluation in the scan inner loop.
- DuckDB per NUMA node: Overengineered. No allocator hooks in DuckDB. Cross-node queries require merging N instances.
- Custom columnar storage (no DuckDB): Full NUMA control, but requires reimplementing schema management, SQL parsing, type system, persistence, and statistics.
- Apache Arrow as compute engine: Good filter kernels, but same NUMA problem. Also immutable RecordBatches conflict with Quake's mutable index model.

### Decision 2: Single DuckDB Table with Partition ID Column

**What:** Store all attribute data in a single DuckDB table with a `partition_id` column, rather than one table per partition.

**Schema:**
```sql
CREATE TABLE attributes (
    partition_id BIGINT NOT NULL,
    row_id BIGINT NOT NULL,       -- same as user-provided vector ID (ids_ in IndexPartition)
    -- user-defined attribute columns added dynamically
    -- e.g., price DOUBLE, category VARCHAR, in_stock BOOLEAN
    PRIMARY KEY (row_id)
);
CREATE INDEX idx_partition ON attributes(partition_id);
```

**Partition statistics table:**
```sql
CREATE TABLE partition_stats (
    partition_id BIGINT NOT NULL,
    attr_name VARCHAR NOT NULL,
    min_val DOUBLE,            -- NULL for non-numeric attributes
    max_val DOUBLE,            -- NULL for non-numeric attributes
    row_count BIGINT,
    PRIMARY KEY (partition_id, attr_name)
);
```

**Bounds limitation:** Partition-level pruning via min/max bounds applies only to orderable numeric types (DOUBLE, BIGINT) and types castable to DOUBLE (DATE, TIMESTAMP via epoch). VARCHAR and BOOLEAN attributes have NULL bounds and are not prunable — partitions are never skipped based on these columns. Row-level scan-time filtering still applies to all types.
```

**Why single table over table-per-partition:**

| Aspect | Table per partition | Single table |
|---|---|---|
| Scan locality | Explicit table isolation | Via DuckDB zone maps on `partition_id` (equivalent) |
| Schema changes | N ALTER TABLEs | 1 ALTER TABLE |
| Split/merge | DROP + CREATE + INSERT | UPDATE `partition_id` |
| Prepared statements | Cannot (dynamic table names) | Yes (parameterized `partition_id = ?`) |
| DuckDB catalog size | O(num_partitions) | O(1) |
| Scale to 10K+ partitions | Problematic (catalog + DDL churn) | Fine |
| Transaction semantics | Multi-table coordination | Single-table |

DuckDB is columnar with zone maps. Filtering `WHERE partition_id = 42` automatically skips row groups that don't contain partition 42. This gives equivalent scan locality to separate tables without DDL overhead.

Splits are `UPDATE attributes SET partition_id = ? WHERE row_id IN (...)`. DuckDB handles this as mark-old + append-new internally, compacted at checkpoint.

**Alternatives considered:**
- Table per partition: Clean isolation but DDL churn at scale (each split = DROP + 2x CREATE + INSERT). DuckDB catalog not designed for thousands of DDL ops. Prepared statements don't work with dynamic table names.

### Decision 3: Structured Filter Objects Rendered to SQL

**What:** Provide a structured filter API (operator objects) in Python and C++ that renders to SQL WHERE clauses for DuckDB execution. Support ID-based filters as a first-class filter type.

**Python API:**
```python
from quake import Eq, Lt, Gt, Le, Ge, Ne, In, NotIn, And, Or, Not, IdIn, IdNotIn

# Simple predicate
filter = Eq("category", "electronics")

# Compound filter
filter = And(Lt("price", 100.0), In("color", ["red", "blue"]))

# ID filter (restricts search to specific vector IDs)
filter = IdIn([42, 99, 137])

# ID exclusion (deduplication)
filter = IdNotIn([42, 99])

# Composition
filter = And(Lt("price", 100.0), IdNotIn([42]))

# Search
results = index.search(query, k=10, filter=filter)
```

**C++ representation:**
```cpp
struct FilterExpr {
    enum Op { EQ, NE, LT, LE, GT, GE, IN, NOT_IN,
              AND, OR, NOT, ID_IN, ID_NOT_IN, IS_NULL, IS_NOT_NULL };
    Op op;
    std::string column;                        // attribute name (empty for AND/OR/NOT)
    duckdb::Value value;                       // scalar comparand for EQ/NE/LT/LE/GT/GE
    std::vector<duckdb::Value> value_list;     // for IN/NOT_IN
    std::vector<int64_t> id_list;              // for ID_IN/ID_NOT_IN
    std::vector<FilterExpr> children;          // for AND/OR/NOT

    // Render to SQL WHERE clause fragment (computed on demand, not cached)
    std::string to_sql() const;
};
```

**How it works:**
1. User constructs filter in Python using operator objects
2. Python filter serializes to C++ `FilterExpr` (carries typed values, not raw SQL)
3. For partition pruning: `get_candidate_partitions()` renders the filter to SQL and queries `partition_stats` in DuckDB
4. For row-level filtering: C++ evaluates `FilterExpr` against NUMA-local `attr_columns_` arrays, producing a byte-mask, which feeds into bitmap scan (M1)

**ID filters:**
- `IdIn([42, 99])` → `row_id IN (42, 99)` — restricts search to only these vector IDs
- `IdNotIn([42, 99])` → `row_id NOT IN (42, 99)` — excludes these IDs from results
- ID filters compose with attribute filters normally via AND/OR
- At the scan level, ID filters are evaluated against `IndexPartition.ids_` (already NUMA-local)

**Why structured objects instead of raw SQL strings:**
- Type safety and validation before reaching DuckDB
- Enables scan-time evaluation on NUMA-local arrays (objects carry the column name + value for direct comparison, not just a SQL string)
- Composable and inspectable (the maintenance policy can examine what attributes are being filtered on)
- DuckDB SQL is the serialization/transport format, not the primary interface

**Alternatives considered:**
- Raw SQL strings only: Concise but no static validation, hard to inspect programmatically
- Custom AST + custom evaluator: Maximum control but reimplements DuckDB's type system and operator semantics
- Lambda/callback functions: Maximum flexibility but can't serialize, optimize, or inspect

### Decision 4: Per-Partition Bounds for Query-Time Pruning

**What:** Maintain per-partition, per-attribute min/max bounds in the DuckDB `partition_stats` table. Use these bounds to skip partitions whose attribute ranges don't overlap the query filter.

**Partition pruning at query time:**
```
1. Decompose filter into per-attribute range predicates
2. For each attribute predicate, query partition_stats:
     SELECT partition_id FROM partition_stats
     WHERE attr_name = 'price' AND min_val < 100.0
   → per-attribute candidate sets
3. Intersect candidate sets (AND) or union them (OR) → candidate_partition_ids
4. Centroid routing over candidates only → top-nprobe partitions
5. Filtered scan on selected partitions
```

**Compound filter pruning:** For compound filters like `And(Lt("price", 100), Gt("rating", 3))`, decompose into per-attribute predicates, query `partition_stats` separately for each, and intersect the resulting partition ID sets in C++. This avoids complex multi-attribute SQL joins on the stats table. For `Or` nodes, union the sets. Non-prunable attributes (VARCHAR, BOOLEAN) produce the full partition set, so they don't restrict candidates.

**Batch search:** When a batch of queries shares the same filter (common case), partition pruning runs once and the candidate set is reused across all queries in the batch.

**Bounds maintenance:**
- On `add()`: UPDATE min/max if new value extends range, increment row_count
- On `remove()`: Decrement row_count. **Bounds become potentially stale** — the removed value might have been the min or max, but we don't recompute. This is conservative: we never miss a true result (partition may be scanned unnecessarily, but never skipped incorrectly).
- On `split()`: Recompute tight bounds from actual child data
- On `refine()`/recluster: Recompute bounds

**Stale bounds are acceptable because:**
- Correctness is preserved: a stale (wider) bound means we scan the partition when we could have skipped it, never the reverse
- Bounds are refreshed naturally on splits, merges, and refinements
- The alternative (maintaining exact bounds on every remove) requires scanning the full column to find the new min/max, which is O(N) per remove

**Why partition_stats in DuckDB (rather than C++ cache):**
- DuckDB handles persistence automatically (save/load)
- Single source of truth for bounds
- The partition pruning query runs once per search call, not in the inner loop — ~100-200μs is acceptable
- For unfiltered queries (no filter predicate), this step is skipped entirely (zero overhead)

**Alternatives considered:**
- C++ bounds cache with DuckDB backup: Faster pruning (~1μs vs ~100μs) but two sources of truth to maintain. Premature optimization for v1.
- DuckDB zone maps only (no explicit stats): DuckDB zone maps are per-row-group, not per-partition_id. Would require scanning all row groups to determine partition overlap. Explicit stats table is more efficient.
- No partition pruning: Possible, but defeats the purpose of H-Quake's attribute-first routing. For a 0.01% selectivity query, pruning skips 99% of centroid distance computations.

### Decision 5: Relational Cuts in Maintenance Policy

**What:** Extend the maintenance policy to support **relational cuts** (attribute-aware partition splits) alongside existing spatial cuts (k-means splits). When the policy decides to split a partition, it evaluates both split types and picks the more profitable one.

**How relational cuts work:**
```
Parent:  centroid=C, bounds=[price: 10..1000], N rows
  →
Left:    centroid=C, bounds=[price: 10..500],  rows where price ≤ 500
Right:   centroid=C, bounds=[price: 500..1000], rows where price > 500
```

- **Centroid is inherited** (shared with parent). No new centroids created.
- **Bounds are tightened** and disjoint between children.
- **Pruning benefit:** Queries whose filter doesn't overlap `[10, 500]` skip the left child in the partition pruning step. Queries that don't overlap `[500, 1000]` skip the right child.

**Spatial cuts (existing behavior):**
```
Parent:  centroid=C, bounds=[price: 10..1000], N rows
  →
Left:    centroid=C₁, bounds=[price: 10..1000], rows nearest C₁
Right:   centroid=C₂, bounds=[price: 10..1000], rows nearest C₂
```

- **Centroids are new** (from k-means). Two new centroids replace the parent's.
- **Bounds are inherited** (both children keep parent's bounds).
- **Pruning benefit:** Queries route to the nearer centroid in spatial routing.

**Comparison:**

| Aspect | Relational Cut | Spatial Cut |
|---|---|---|
| Centroid | Inherited (shared) | New (k-means) |
| Bounds | Tightened | Inherited |
| Pruning via | Partition bounds check (Step 1) | Centroid routing (Step 2) |
| When profitable | Selective filter workloads | Spatially clustered data |

**Integration with existing maintenance flow:**

Current `perform_maintenance()`:
```
For each partition:
  1. Should we delete it? (cost model)
  2. Should we split it? (cost model → k-means split)
```

With relational cuts:
```
For each partition:
  1. Should we delete it? (cost model)
  2. Should we split it?
     a. Evaluate spatial split profit (existing k-means cost model)
     b. Evaluate relational split profit (attribute-based cost model)
     c. Execute whichever is more profitable (if profit > 0)
```

**Implementation in `PartitionManager::split_partitions()`:**

The method gains a `split_type` parameter:
- `SPATIAL`: existing k-means behavior. Creates 2 new centroids, same bounds.
- `RELATIONAL`: split on attribute threshold. Same centroid, tightened bounds. Requires `(attribute_name, threshold)`.

**Relational split execution:**
1. Query DuckDB: `SELECT row_id FROM attributes WHERE partition_id = ? AND <attr> <= <threshold>`
2. Partition the `codes_`, `ids_`, and `attr_data_` arrays into two children based on matching row_ids
3. Both children inherit the parent's centroid
4. Update `partition_stats` with tightened bounds
5. Update DuckDB `attributes` table with new partition_ids

**Data-only split-point selection for v1:**

For v1, use a simple heuristic to choose the split attribute and threshold:
- **Attribute:** Pick the attribute with the widest normalized range (max-min)/max across the partition's data
- **Threshold:** Median value of the chosen attribute (produces balanced children)

See **Open Questions** for workload-aware alternatives.

**Alternatives considered:**
- Relational cuts deferred to v2: Simpler, but partition-level pruning without relational cuts is limited — all partitions from the initial k-means have the same (global) bounds until data is split relationally.
- Always prefer relational over spatial: No — spatial cuts are better when queries have broad filters but spatially clustered data.

### Decision 6: DuckDB Transactions for Consistency

**What:** Wrap vector mutations and attribute mutations in DuckDB transactions to maintain consistency between the vector store (`IndexPartition`) and the attribute store (DuckDB).

**Pattern:**
```cpp
// In QuakeIndex::add()
attribute_store_->begin_transaction();
try {
    partition_manager_->add(vectors, ids, assignments);       // vector side
    attribute_store_->insert_rows(partition_id, ids, attrs);  // DuckDB side
    attribute_store_->update_stats(partition_id);             // bounds
    attribute_store_->commit();
} catch (...) {
    attribute_store_->rollback();
    partition_manager_->remove(ids);  // rollback vector side
    throw;
}

// In QuakeIndex::remove()
attribute_store_->begin_transaction();
try {
    partition_manager_->remove(ids);
    attribute_store_->delete_rows(ids);
    attribute_store_->update_stats_after_remove(partition_ids);
    attribute_store_->commit();
} catch (...) {
    attribute_store_->rollback();
    throw;
}
```

**For splits (maintenance):**
```
BEGIN TRANSACTION;
  1. Read attribute data for parent partition from DuckDB
  2. Determine child assignments (spatial or relational)
  3. UPDATE partition_id for rows going to each child
  4. DELETE old partition_stats rows, INSERT new ones with tight bounds
COMMIT;
Then: vector-side split (create new IndexPartitions with NUMA-local attr_data_)
```

**Why:**
- DuckDB has mature transaction support (ACID)
- Single-writer model matches Quake's existing mutex-based mutation model
- Searches see a consistent snapshot (either pre- or post-mutation)

### Decision 7: Persistence via DuckDB Checkpoint

**What:** Store the DuckDB database as part of the index save/load. DuckDB runs in-memory during operation and checkpoints to disk on save.

**On-disk layout:**
```
quake_index/
    metadata.txt          # existing: index metadata
    partitions            # existing: vector data (codes + ids)
    parent/               # existing: centroid index
    attributes.duckdb     # NEW: DuckDB database file
```

**Save:** DuckDB checkpoint (flush WAL to main database file).
**Load:** Open DuckDB file, then populate NUMA-local `attr_data_` arrays in each `IndexPartition` from DuckDB data, allocated on the correct NUMA nodes.

**Backward compatibility:** If `attributes.duckdb` does not exist on load, the index loads without attributes. Schema is empty. Vector search works normally. Filtered search returns an error if a filter is provided.

**Why in-memory with checkpoint (not file-backed):**
- In-memory DuckDB avoids WAL/fsync overhead during normal operation
- Checkpointing on save is explicit and predictable
- Load path reads from DuckDB and materializes into NUMA-local arrays — the DuckDB file is only read once

### Decision 8: Input Format for Attributes

**What:** Accept attributes as a pandas DataFrame or PyArrow table at the Python API level. Internally, attributes are inserted into DuckDB and materialized into NUMA-local arrays.

**Python API:**
```python
import pandas as pd

# Build with pandas DataFrame
attrs = pd.DataFrame({
    "price": [9.99, 19.99, 29.99, ...],
    "category": ["books", "electronics", "toys", ...],
    "in_stock": [True, False, True, ...]
})
index.build(vectors, ids, attributes=attrs)

# Build with PyArrow table
import pyarrow as pa
attrs = pa.table({"price": [9.99, 19.99], "category": ["books", "toys"]})
index.build(vectors, ids, attributes=attrs)

# Add with attributes
new_attrs = pd.DataFrame({"price": [39.99], "category": ["games"], "in_stock": [True]})
index.add(new_vectors, new_ids, attributes=new_attrs)
```

**Schema inference:** The attribute schema is inferred from the DataFrame/table dtypes on first `build()` call. Subsequent `add()` calls must conform to the established schema. If an `add()` call provides columns that don't match the schema (missing columns, extra columns, or incompatible types), the operation fails with a schema mismatch error.

**Why pandas + PyArrow:**
- pandas is the de facto standard for tabular data in Python ML workflows
- PyArrow enables zero-copy transfer to DuckDB (DuckDB natively reads Arrow)
- Both are likely already in the user's environment
- Type mapping is natural: pandas float64 → DuckDB DOUBLE, object/string → VARCHAR, bool → BOOLEAN, int64 → BIGINT, datetime64 → TIMESTAMP

## Search Flow (Complete)

```
search(queries, params, filter) {

    // STEP 0: No filter? Use existing unfiltered path (zero overhead)
    if (filter == null) {
        return existing_search(queries, params);
    }

    // STEP 1: Partition pruning via DuckDB partition_stats
    //   - Parse filter for range constraints
    //   - Query partition_stats to find partitions with overlapping bounds
    //   - O(1) DuckDB query on a small table (~num_partitions × num_attributes rows)
    candidate_ids = attribute_store.get_candidate_partitions(filter);

    // STEP 2: Centroid routing over candidates only
    //   - Compute distance from query to centroids of candidate partitions
    //   - Select top-nprobe nearest
    //   - This is cheaper than routing over ALL partitions when filter is selective
    routed_ids = centroid_routing(queries, candidate_ids, nprobe);

    // STEP 3: Filtered scan (NUMA-local hot path)
    for each partition_id in routed_ids:
        partition = get_partition(partition_id);

        // 3a. Evaluate filter on NUMA-local attr_columns_ → byte-mask
        //     One byte per row: 1 = passes filter, 0 = skip
        mask = eval_filter(partition.attr_columns_, partition.ids_, filter);

        // 3b. Bitmap scan (M1): compute distance only for mask[i] == 1
        bitmap_scan(partition.codes_, mask, query, topk_buffer);

    // STEP 4: Merge top-k results (existing)
    return merge_topk(topk_buffers);
}
```

**Performance characteristics by selectivity:**

| Selectivity | Step 1 effect | Step 2 effect | Step 3 effect |
|---|---|---|---|
| 0.01% (very selective) | Prunes ~99% of partitions | Routes over ~1% of centroids | Scans very few rows per partition |
| 10% (selective) | Prunes ~50-90% of partitions | Routes over reduced set | Skips most distance computations |
| 90% (broad) | Prunes few partitions | Routes over nearly all centroids | Most rows pass filter |
| 100% (no filter) | Skipped entirely | Full centroid set (existing) | All rows (existing) |

## Component: AttributeStore

```cpp
class AttributeStore {
public:
    // Lifecycle
    AttributeStore();  // creates in-memory DuckDB
    ~AttributeStore();

    // Schema management
    void initialize_schema(const std::vector<std::pair<std::string, std::string>>& columns);
    // columns: [(name, duckdb_type)] e.g., [("price", "DOUBLE"), ("category", "VARCHAR")]

    std::vector<std::pair<std::string, std::string>> get_schema() const;

    // Data operations (transactional)
    void begin_transaction();
    void commit();
    void rollback();

    void insert_rows(int64_t partition_id,
                     const std::vector<int64_t>& row_ids,
                     /* columnar attribute data */);
    void delete_rows(const std::vector<int64_t>& row_ids);
    void update_partition_ids(const std::vector<int64_t>& row_ids,
                              int64_t new_partition_id);

    // Partition stats
    void recompute_stats(int64_t partition_id);
    void recompute_stats_all();
    void update_stats_on_insert(int64_t partition_id, /* new values */);

    // Query: partition pruning
    std::vector<int64_t> get_candidate_partitions(const FilterExpr& filter);

    // Query: materialize NUMA-local arrays for a partition
    //   Called on load() and after splits to populate IndexPartition.attr_data_
    void materialize_partition_attrs(int64_t partition_id,
                                     IndexPartition& partition);

    // Persistence
    void save(const std::string& path);  // checkpoint to attributes.duckdb
    void load(const std::string& path);  // open from attributes.duckdb

    bool has_attributes() const;  // false if schema is empty

private:
    std::unique_ptr<duckdb::DuckDB> db_;
    std::unique_ptr<duckdb::Connection> conn_;
};
```

## Component: NUMA-Local Attribute Data in IndexPartition

```cpp
// Extension to IndexPartition
struct AttrColumn {
    std::string name;
    std::string type;        // DuckDB type name: "DOUBLE", "BIGINT", "VARCHAR", "BOOLEAN"
    void* data = nullptr;    // NUMA-allocated typed array. Cast to float*, int64_t*, etc.
    size_t elem_size = 0;    // bytes per element
};

class IndexPartition {
    // ... existing fields ...

    // NEW: NUMA-local attribute columns
    std::vector<AttrColumn> attr_columns_;

    // Append attribute values for new vectors (mirrors codes_/ids_ append)
    void append_attrs(int64_t n_entry, const std::vector<AttrColumn>& new_data);

    // Remove attribute row at index (swap-last, mirrors existing remove)
    void remove_attr(int64_t index);

    // Allocate/free attribute columns on NUMA node
    void allocate_attr_columns(const std::vector<std::pair<std::string, std::string>>& schema,
                                int64_t capacity);
    void free_attr_columns();
};
```

## Scan-Time Filter Evaluation

Filter evaluation on NUMA-local arrays is simple typed comparison loops:

```cpp
// Evaluate "price < 100.0" on a float column
void eval_lt_f64(const double* col, int64_t n, double threshold, uint8_t* mask) {
    for (int64_t i = 0; i < n; i++) {
        mask[i] &= (col[i] < threshold) ? 1 : 0;
    }
}

// String predicates: see Open Questions § String Filtering on NUMA-Local Arrays.
// V1 falls back to DuckDB for string predicates.

// Compound filters: AND byte-masks together
void eval_and(const uint8_t* a, const uint8_t* b, int64_t n, uint8_t* out) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = a[i] & b[i];
    }
}
```

Numeric predicates (EQ, NE, LT, LE, GT, GE) on contiguous arrays are trivially SIMD-izable. IN predicates use a hash set lookup. String predicates may fall back to DuckDB for complex operations.

## Risks / Trade-offs

### Risk 1: Two Copies of Attribute Data
**Risk:** NUMA-local arrays in `IndexPartition` and DuckDB table are two copies. Inconsistency if one is updated without the other.
**Mitigation:**
- DuckDB is the authoritative source. NUMA-local arrays are a materialized view.
- All mutations go through `AttributeStore` which updates both in a transaction.
- On `load()`, NUMA arrays are rebuilt from DuckDB. DuckDB is always correct.
- The only staleness risk is a crash between vector-side and DuckDB-side mutation. DuckDB transactions mitigate this.

### Risk 2: DuckDB Dependency Size and Overhead
**Risk:** DuckDB adds ~20MB to the binary. Query compilation adds ~100-200μs latency per search call (for partition pruning).
**Mitigation:**
- 20MB is acceptable for a production system.
- Partition pruning query is on a small table; prepared statements amortize compilation.
- For unfiltered queries, DuckDB is not invoked at all (zero overhead).

### Risk 3: DuckDB Single-Writer Model
**Risk:** DuckDB allows only one writer at a time. If maintenance and add/remove contend, one blocks.
**Mitigation:**
- Quake already uses a single-writer model for index mutations (mutex-guarded).
- DuckDB's concurrency model matches Quake's existing design.
- Concurrent reads (searches) are fine and do not block writes.

### Risk 4: NUMA Array Sync on Splits
**Risk:** After a relational or spatial split, NUMA-local attribute arrays must be rebuilt for both children.
**Mitigation:**
- Splits already rebuild vector data (`codes_`, `ids_`). Attribute arrays follow the same pattern.
- `materialize_partition_attrs()` reads from DuckDB and allocates on the correct NUMA node.

### Risk 5: Memory Overhead of Dual Storage
**Risk:** Storing attributes twice (DuckDB + NUMA arrays) increases memory usage.
**Mitigation:**
- Attribute data is small relative to vector data (~1-3% of total memory).
- DuckDB's columnar compression reduces its copy's footprint.
- The alternative (DuckDB-only) sacrifices NUMA locality on every filtered scan.

### Risk 6: Partition Pruning Limited to Numeric Types
**Risk:** `partition_stats` bounds (min/max DOUBLE) only support pruning on numeric attributes. VARCHAR and BOOLEAN attributes never contribute to partition pruning, so queries filtering only on non-numeric attributes scan all partitions.
**Mitigation:**
- Row-level scan-time filtering still applies to all types (correctness unaffected).
- Most ANN workloads with filtering use numeric predicates (price, date, rating). String equality filters are less common as the primary selective predicate.
- Future: add hash-based or lexicographic bounds for string pruning if profiling shows it matters.

### Risk 7: Schema Rigidity After Build
**Risk:** Attribute schema is fixed at `build()` time. Adding or removing attribute columns requires a full rebuild.
**Mitigation:**
- This matches the typical ANN index lifecycle (build once, query many times).
- DuckDB supports ALTER TABLE, so schema evolution could be added later without architectural changes.
- v1 focuses on correctness; schema evolution is a natural follow-up.

## Migration Plan

Work is organized into three milestones. Each milestone is independently shippable and has explicit "done" criteria.

### M1: Bitmap Scan
Modify scan paths in `list_scanning.h` and `QueryCoordinator` to accept an optional byte-mask (`uint8_t*`, one byte per row, 1=include, 0=skip). When provided, skip distance computation for masked-out rows. This is a small, self-contained change that establishes the primitive all filtering builds on.

**Scope:** `list_scanning.h`, `QueryCoordinator` scan loops. No API changes, no new dependencies.

**Done when:**
- All scan modes (serial, batched, worker) support byte-mask
- Unit tests pass with random masks
- Benchmark confirms expected speedup (10% selectivity → ~10x fewer distance ops)
- Zero overhead when mask is nullptr

### M2: DuckDB Attribute Store + Filtering
Add the `AttributeStore` class (DuckDB wrapper), NUMA-local attribute arrays in `IndexPartition`, structured filter expressions, partition-level pruning, filtered search integration, write path, persistence, and Python API.

**Scope:** New `AttributeStore` class. Modifications to `IndexPartition`, `DynamicInvertedLists`, `QueryCoordinator`, `QuakeIndex`, `SearchParams`, Python bindings. New dependency: DuckDB C++ library.

**Done when:**
- Can build index with pandas DataFrame of attributes
- Can search with filter objects (`Eq`, `Lt`, `And`, `IdIn`, etc.)
- Save/load round-trips with attributes
- Unfiltered search has zero overhead
- Filtered search returns correct results at various selectivities
- Backward compat: old indexes without attributes load normally

### M3: Relational Cuts
Extend maintenance policy to support relational cuts (attribute-aware partition splits). The index adapts its partition structure to filter workloads.

**Scope:** `PartitionManager::split_partitions()`, `MaintenancePolicy::perform_maintenance()`, partition stats updates.

**Done when:**
- Maintenance can produce relational splits (split on attribute threshold, same centroid, tightened bounds)
- Partition pruning correctly skips children with non-overlapping bounds after relational split
- Integration test: build → filtered workload → maintenance → verify partition structure adapted and filtered search improved

**Rollback:** Attribute support is additive. Existing APIs are unchanged. An index built without attributes loads and searches identically to today. Remove by reverting to prior version.

## Open Questions

### Relational Cut Split-Point Selection Strategy

The v1 implementation uses a **data-only heuristic** (split on median of widest-range attribute). Future versions should consider workload-aware strategies. Options recorded for future design:

**Option A: Query reservoir with filter recording**
Store recent queries with their filter predicates per partition. Evaluate candidate split points against the reservoir to estimate pruning benefit. Requires per-partition FIFO buffer of query filters.
- Pro: Accurate workload modeling
- Con: Memory overhead per partition, privacy concerns with stored queries

**Option B: Filter frequency counters (Level 2 stats)**
Track per partition, per attribute: how many queries filtered on it, and the running min/max of filter boundary values seen.
```cpp
struct PartitionFilterStats {
    std::unordered_map<std::string, int64_t> filter_hit_count;
    std::unordered_map<std::string, std::pair<double, double>> filter_boundary_range;
};
```
Split on the most-filtered attribute, at a threshold that maximizes expected pruning based on boundary distribution.
- Pro: Lightweight counters, low overhead
- Con: Doesn't capture full selectivity distribution

**Option C: Selectivity sketch (Level 3 stats)**
Per partition, per attribute, maintain approximate quantiles of both filter boundaries and data values. Estimate pruning benefit of any candidate threshold.
- Pro: Most accurate pruning estimates
- Con: Quantile sketch per attribute per partition, higher memory and compute

**Option D: DuckDB-derived statistics**
Use DuckDB's built-in `APPROX_QUANTILE` on the attributes table to compute candidate thresholds. Combine with filter frequency counters (Option B) to decide which attribute to split on.
- Pro: Leverages DuckDB's statistics engine, no custom sketches
- Con: Requires DuckDB query during maintenance, may add latency

**Decision deferred.** V1 ships with the data-only heuristic. The cost model infrastructure (partition sizes, hit rates) already exists in `MaintenanceCostEstimator`. Extending it to account for filter selectivity is a natural next step once the basic attribute filtering is proven.

### String Filtering on NUMA-Local Arrays
For numeric predicates, NUMA-local evaluation is straightforward (typed array comparison). String predicates are more complex:
- Equality: could store a hash alongside the string for fast NUMA-local check
- LIKE/substring: must fall back to DuckDB (acceptable since string pattern matching is inherently expensive)
- IN (string set): hash set lookup on precomputed hashes

Decision deferred to implementation. Start with DuckDB fallback for all string predicates; optimize with hashes if profiling shows it matters.

### DuckDB Version Pinning
DuckDB's on-disk format changes between major versions. The `attributes.duckdb` file may not be readable by a different DuckDB version. Options:
- Pin DuckDB version in build system
- Use DuckDB's `EXPORT DATABASE` (CSV/Parquet) for portable persistence, reimport on load
- Store a DuckDB version tag in `metadata.txt` and fail fast on mismatch

Decision deferred to implementation.
