## ADDED Requirements

### Requirement: DuckDB Attribute Store
The system SHALL use an embedded DuckDB database as the authoritative store for attribute data.

Attribute data SHALL be stored in a single DuckDB table with a `partition_id` column and a `row_id` column alongside user-defined attribute columns.

Per-partition attribute bounds (min, max, row_count per attribute) SHALL be maintained in a `partition_stats` table.

Bounds SHALL only be populated for orderable numeric types (BIGINT, DOUBLE) and types castable to DOUBLE (DATE, TIMESTAMP). Non-numeric attributes (VARCHAR, BOOLEAN) SHALL have NULL bounds and SHALL NOT be used for partition-level pruning.

DuckDB SHALL run in-memory during operation and checkpoint to disk on save.

The system SHALL support all DuckDB-native column types (BIGINT, DOUBLE, VARCHAR, BOOLEAN, DATE, TIMESTAMP, etc.).

#### Scenario: Schema inferred from input
- **WHEN** user builds an index with a pandas DataFrame containing columns "price" (float64) and "category" (object)
- **THEN** the DuckDB `attributes` table is created with columns `price DOUBLE` and `category VARCHAR`
- **AND** the schema is available via `get_schema()`

#### Scenario: Single table stores all partitions
- **WHEN** index has 500 partitions with attributes
- **THEN** all attribute rows are in one `attributes` table with `partition_id` distinguishing them
- **AND** DuckDB zone maps enable efficient per-partition queries

---

### Requirement: NUMA-Local Attribute Arrays
The system SHALL maintain NUMA-local copies of attribute column data in each `IndexPartition` for scan-time filter evaluation.

NUMA-local attribute arrays SHALL be allocated using `quake_alloc()` on the partition's NUMA node.

NUMA-local attribute arrays SHALL follow the same row ordering as `codes_` and `ids_`.

Mutations (append, swap-last remove) SHALL mirror existing `codes_`/`ids_` patterns.

On `load()`, NUMA-local arrays SHALL be materialized from DuckDB data onto the correct NUMA nodes.

#### Scenario: Scan-time filter uses NUMA-local data
- **WHEN** a filtered search scans partition P on NUMA node 1
- **THEN** filter predicate evaluation reads from `attr_data_` arrays allocated on NUMA node 1
- **AND** no cross-NUMA memory access occurs during the filter+distance inner loop

#### Scenario: Attributes follow swap-last remove
- **WHEN** vector at index 5 is removed from a partition with 100 vectors
- **THEN** the attribute values at index 5 are overwritten with values from index 99
- **AND** attribute array length decrements to 99

#### Scenario: Non-numeric attribute skips pruning
- **WHEN** user searches with `filter=Eq("category", "electronics")`
- **AND** category is a VARCHAR column
- **THEN** partition pruning does not eliminate any partitions (all are candidates)
- **AND** row-level scan-time filtering still returns only matching vectors

---

### Requirement: Structured Filter Expressions
The system SHALL support structured filter objects for specifying search predicates.

Filter objects SHALL support the following operators:
- Comparison: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`
- Logical: `And`, `Or`, `Not`
- Set membership: `In`, `NotIn`
- Null checks: `IsNull`, `IsNotNull`
- ID filters: `IdIn`, `IdNotIn`

Filter objects SHALL render to SQL WHERE clause fragments for DuckDB execution.

Filter objects SHALL also be evaluable on NUMA-local typed arrays at scan time.

#### Scenario: Simple comparison filter
- **WHEN** user searches with `filter=Lt("price", 100.0)`
- **THEN** results only include vectors where the price attribute is less than 100.0

#### Scenario: Compound filter
- **WHEN** user searches with `filter=And(Lt("price", 100.0), In("category", ["electronics", "books"]))`
- **THEN** results only include vectors matching both conditions

#### Scenario: ID filter
- **WHEN** user searches with `filter=IdIn([42, 99, 137])`
- **THEN** results only include vectors with those IDs

#### Scenario: ID exclusion filter
- **WHEN** user searches with `filter=IdNotIn([42])`
- **THEN** results exclude vector with ID 42

#### Scenario: Composed attribute and ID filter
- **WHEN** user searches with `filter=And(Lt("price", 100.0), IdNotIn([42]))`
- **THEN** results match the price filter AND exclude ID 42

---

### Requirement: Partition-Level Pruning via Bounds
The system SHALL maintain per-partition, per-attribute bounds (min, max) in DuckDB `partition_stats`.

During filtered search, the system SHALL query `partition_stats` to identify candidate partitions whose attribute bounds overlap the query filter.

Only candidate partitions SHALL be considered during centroid routing.

Bounds SHALL be recomputed on partition splits and refinements.

On remove, bounds MAY become stale (wider than actual data range). Stale bounds SHALL NOT cause incorrect results (only unnecessary scans).

#### Scenario: Partition pruning skips non-overlapping partitions
- **WHEN** user searches with `filter=Lt("price", 50.0)`
- **AND** partition P has `partition_stats` showing `price: min=100, max=500`
- **THEN** partition P is excluded from candidate set before centroid routing

#### Scenario: Stale bounds do not miss results
- **WHEN** the maximum price value (1000) is removed from a partition
- **AND** bounds still show `price: max=1000`
- **THEN** the partition is still considered for queries with `price < 1000`
- **AND** no matching vectors are missed

#### Scenario: Unfiltered search skips pruning
- **WHEN** user searches without a filter
- **THEN** partition pruning step is skipped entirely
- **AND** search performance is identical to the unfiltered baseline

---

### Requirement: Relational Cuts
The maintenance policy SHALL support relational cuts (attribute-aware partition splits) alongside spatial cuts (k-means splits).

A relational cut SHALL split a partition on an attribute threshold, inheriting the parent centroid and tightening attribute bounds for the children.

When deciding to split a partition, the system SHALL evaluate both spatial and relational split profit and execute whichever is more profitable.

In v1, the split attribute and threshold SHALL be selected using a data-only heuristic (median of the widest-range attribute).

#### Scenario: Relational cut on price attribute
- **WHEN** maintenance decides to split partition P with price range [10, 1000]
- **AND** relational cut is more profitable than spatial cut
- **THEN** partition P is split into two children with same centroid
- **AND** left child has bounds `price: [10, 500]`, right child has bounds `price: (500, 1000]`
- **AND** vectors are distributed to children based on their price values

#### Scenario: Spatial cut preferred for broad-filter workload
- **WHEN** maintenance evaluates partition P
- **AND** spatial split profit exceeds relational split profit
- **THEN** partition P is split via k-means (existing behavior)
- **AND** both children inherit the parent's attribute bounds

#### Scenario: Relational cut enables partition pruning
- **WHEN** partition P has been relationally split on price at threshold 500
- **AND** query has filter `price < 200`
- **THEN** only the left child (price: [10, 500]) is a candidate
- **AND** the right child (price: (500, 1000]) is pruned in partition pruning

---

### Requirement: Transactional Consistency
All mutations (add, remove, split) SHALL be wrapped in DuckDB transactions to maintain consistency between vector data and attribute data.

If the DuckDB operation fails, the vector-side operation SHALL be rolled back.

Concurrent searches SHALL see a consistent snapshot (either fully pre- or post-mutation).

#### Scenario: Add with attributes is atomic
- **WHEN** user adds vectors with attributes
- **AND** the DuckDB insert fails
- **THEN** the vector-side add is rolled back
- **AND** the index state is unchanged

#### Scenario: Search during mutation
- **WHEN** a search executes concurrently with an add operation
- **THEN** the search sees either all or none of the new vectors and attributes

---

### Requirement: Attribute Persistence
The system SHALL persist attribute data as a DuckDB database file (`attributes.duckdb`) alongside the existing index files.

Loading an index saved without `attributes.duckdb` SHALL succeed with an empty schema.

#### Scenario: Schema mismatch on add raises error
- **WHEN** index was built with attributes "price" (DOUBLE) and "category" (VARCHAR)
- **AND** user calls `add()` with attributes "price" (DOUBLE) and "color" (VARCHAR)
- **THEN** the operation fails with a schema mismatch error
- **AND** the index state is unchanged

#### Scenario: Save and load index with attributes
- **WHEN** user saves an index with attributes to disk
- **AND** user loads index from disk
- **THEN** all attribute values are restored in DuckDB
- **AND** NUMA-local attribute arrays are materialized on correct NUMA nodes
- **AND** filtered search works correctly

#### Scenario: Backward compatible load
- **WHEN** user loads an index saved before attribute support was added
- **THEN** load succeeds
- **AND** attribute schema is empty
- **AND** vector search works normally
- **AND** providing a filter to search raises an error

---

### Requirement: Python API for Attributes
The system SHALL expose attribute functionality through the Python API.

Build and add SHALL accept attributes as a pandas DataFrame or PyArrow table.

Filter expressions SHALL be constructible in Python using operator objects (`Eq`, `Lt`, `And`, etc.).

#### Scenario: Build with pandas DataFrame
- **WHEN** user calls `index.build(vectors, ids, attributes=pd.DataFrame({"price": prices, "category": categories}))`
- **THEN** index is built with attribute values
- **AND** filtered search is available

#### Scenario: Build with PyArrow table
- **WHEN** user calls `index.build(vectors, ids, attributes=pa.table({"price": prices}))`
- **THEN** index is built with attribute values

#### Scenario: Filter objects in Python
- **WHEN** user constructs `filter=And(Lt("price", 100), Eq("in_stock", True))`
- **AND** passes it to `index.search(query, k=10, filter=filter)`
- **THEN** search returns only matching vectors
