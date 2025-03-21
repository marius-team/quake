Quake Index Architecture
=========================

Overview
--------
The Quake Index is an adaptive, hierarchical indexing system for vector search. It uses a partitioned
index design in which the dataset is recursively clustered into a multi-level structure. The **QuakeIndex**
class coordinates index construction, query processing, and maintenance. Underneath, a **DynamicInvertedLists**
object stores multiple **IndexPartition** instances, each representing a subset of vectors. A **PartitionManager**
controls how vectors are assigned to partitions, while a **QueryCoordinator** manages query execution.
Finally, a **MaintenancePolicy** adapts the index to evolving data and query patterns over time.

QuakeIndex Class Components
---------------------

The diagram below illustrates the main components of the QuakeIndex class and the primary operations each is responsible for:

- **PartitionManager**: Handles modifications of vectors and partitions. Responsible for `add()` and `remove()`
- **QueryCoordinator**: Manages search operations. Used in `search()` and `add()`.
- **CentroidIndex**: A QuakeIndex over the centroids for efficient searching of centroids.
- **MaintenancePolicy**: Maintains partition access counts and oversees periodic index maintenance.

.. image:: quake_arch_diagram.png
  :width: 1600
  :alt: Architecture Diagram

Detailed Components
-------------------

.. toctree::
   :maxdepth: 2

   query_coordinator
   maintenance_policy
   partition_manager
   dynamic_inverted_list
   index_partition


Data Layout
-----------
The index is organized hierarchically. Each level uses centroids to summarize the partitions of the next level.
Every centroid corresponds to a cluster that can be recursively subdivided. At the base level, partitions store
the actual encoded vectors and their IDs. The figure below shows a toy 3-level Quake Index with three partitions
per level and 27 base vectors. Although this example uses uniform partition sizes, real-world sizes vary based
on data distribution.

.. graphviz::
   :caption: Example with 3 levels and 3 partitions per level and 27 base vectors.
   :name: fig-quake-index

    digraph QuakeIndex {
        rankdir=TB;
        // Global node defaults: using HTML labels so shape is none
        node [shape=none, style=filled, fontname=Helvetica, fontsize=18];

        // =============================
        // Level 2 (Top Level)
        // =============================
        subgraph cluster_level2 {
            label = "Level 2 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            topP [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0"><FONT POINT-SIZE="18">C<sup>2</sup><sub>0</sub></FONT></TD>
                  <TD PORT="f1"><FONT POINT-SIZE="18">C<sup>2</sup><sub>1</sub></FONT></TD>
                  <TD PORT="f2"><FONT POINT-SIZE="18">C<sup>2</sup><sub>2</sub></FONT></TD>
                </TR>
              </TABLE>
            >, fillcolor="#45b594"];
            { rank=same; topP }
        }

        // =============================
        // Level 1 (Middle Level)
        // =============================
        // Partition 0 of Level 1
        subgraph cluster_level1_0 {
            label = "Level 1 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            midP0 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>0</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>1</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>2</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#5aaed5"];
            { rank=same; midP0 }
        }
        // Partition 1 of Level 1
        subgraph cluster_level1_1 {
            label = "Level 1 - Partition 1";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            midP1 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>3</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>4</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>5</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#5aaed5"];
            { rank=same; midP1 }
        }
        // Partition 2 of Level 1
        subgraph cluster_level1_2 {
            label = "Level 1 - Partition 2";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            midP2 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>6</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>7</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>8</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#5aaed5"];
            { rank=same; midP2 }
        }

        // =============================
        // Level 0 (Base Level)
        // =============================
        // Partition 0 of Level 0
        subgraph cluster_level0_0 {
            label = "Level 0 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP0 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>0</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>1</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>2</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP0 }
        }
        // Partition 1 of Level 0
        subgraph cluster_level0_1 {
            label = "Level 0 - Partition 1";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP1 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>3</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>4</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>5</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP1 }
        }
        // Partition 2 of Level 0
        subgraph cluster_level0_2 {
            label = "Level 0 - Partition 2";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP2 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>6</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>7</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>8</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP2 }
        }
        // Partition 3 of Level 0
        subgraph cluster_level0_3 {
            label = "Level 0 - Partition 3";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP3 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>9</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>10</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>11</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP3 }
        }
        // Partition 4 of Level 0
        subgraph cluster_level0_4 {
            label = "Level 0 - Partition 4";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP4 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>12</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>13</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>14</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP4 }
        }
        // Partition 5 of Level 0
        subgraph cluster_level0_5 {
            label = "Level 0 - Partition 5";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP5 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>15</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>16</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>17</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP5 }
        }
        // Partition 6 of Level 0
        subgraph cluster_level0_6 {
            label = "Level 0 - Partition 6";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP6 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>18</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>19</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>20</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP6 }
        }
        // Partition 7 of Level 0
        subgraph cluster_level0_7 {
            label = "Level 0 - Partition 7";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP7 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>21</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>22</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>23</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP7 }
        }
        // Partition 8 of Level 0
        subgraph cluster_level0_8 {
            label = "Level 0 - Partition 8";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            baseP8 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>24</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>25</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>26</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];
            { rank=same; baseP8 }
        }

        // =============================
        // Edges: Connect parents to children.
        // =============================
        // Top-level -> Middle-level
        topP:f0 -> midP0;
        topP:f1 -> midP1;
        topP:f2 -> midP2;

        // Middle-level -> Base-level
        midP0:f0 -> baseP0;
        midP0:f1 -> baseP1;
        midP0:f2 -> baseP2;

        midP1:f0 -> baseP3;
        midP1:f1 -> baseP4;
        midP1:f2 -> baseP5;

        midP2:f0 -> baseP6;
        midP2:f1 -> baseP7;
        midP2:f2 -> baseP8;
    }

Query Processing Example
------------------------
When a query is issued, it descends the index hierarchy from the top level to the base level where vectors are
stored. At each level, the query vector is compared against centroids to determine which partitions to scan next.
This continues until the base level is reached, where the actual vectors are compared with the query. The figure
below illustrates a 3-level Quake Index example. Scanned partitions are shown in full color, while non-scanned
partitions are faded.

.. graphviz::
   :caption: Query Processing Example (3-Level Quake Index)
   :name: fig-quake-index-query

    digraph QuakeIndexQueryVector {
        rankdir=TB;
        node [shape=none, style=filled, fontname=Helvetica, fontsize=18];

        // Query vector node: use soft red color. hex = #fa4251
        query [label="Query Vector", shape=rectangle, style=filled, fillcolor="#fa4251"];

        // =============================
        // Level 2 (Top Level) - Fully scanned
        // =============================
        subgraph cluster_level2 {
            label = "Level 2 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";

            topP [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0"><FONT POINT-SIZE="18">C<sup>2</sup><sub>0</sub></FONT></TD>
                  <TD PORT="f1"><FONT POINT-SIZE="18">C<sup>2</sup><sub>1</sub></FONT></TD>
                  <TD PORT="f2"><FONT POINT-SIZE="18">C<sup>2</sup><sub>2</sub></FONT></TD>
                </TR>
              </TABLE>
            >, fillcolor="#45b594"];
            { rank=same; topP }
        }

        { rank=same; query, topP }

        // =============================
        // Level 1 (Middle Level)
        // Only 2 of 3 partitions are scanned.
        // Here, Partition 1 and Partition 2 are scanned, Partition 0 is faded.
        // =============================
        subgraph cluster_level1_0 {
            label = "Level 1 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            midP0 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>0</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>1</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>2</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; midP0 }
        }
        subgraph cluster_level1_1 {
            label = "Level 1 - Partition 1";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            midP1 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>3</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>4</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>5</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#5aaed5"];  // full color
            { rank=same; midP1 }
        }
        subgraph cluster_level1_2 {
            label = "Level 1 - Partition 2";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            midP2 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">C<sup>1</sup><sub>6</sub></TD>
                  <TD PORT="f1">C<sup>1</sup><sub>7</sub></TD>
                  <TD PORT="f2">C<sup>1</sup><sub>8</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#5aaed5"];  // full color
            { rank=same; midP2 }
        }

        // =============================
        // Level 0 (Base Level)
        // 9 partitions total; only 4 are scanned.
        // Partitions under non-scanned middle (Partition 0) are faded.
        // For Partition 1 (child of midP1): scan 2 out of 3 base partitions.
        // For Partition 2 (child of midP2): scan 2 out of 3 base partitions.
        // =============================
        // Base partitions for middle level Partition 0 (faded)
        subgraph cluster_level0_0 {
            label = "Level 0 - Partition 0";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP0 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>0</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>1</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>2</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; baseP0 }
        }
        subgraph cluster_level0_1 {
            label = "Level 0 - Partition 1";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP1 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>3</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>4</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>5</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; baseP1 }
        }
        subgraph cluster_level0_2 {
            label = "Level 0 - Partition 2";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP2 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>6</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>7</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>8</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; baseP2 }
        }
        // Base partitions for middle level Partition 1 (scanned: 2 of 3)
        subgraph cluster_level0_3 {
            label = "Level 0 - Partition 3";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP3 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>9</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>10</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>11</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];  // scanned
            { rank=same; baseP3 }
        }
        subgraph cluster_level0_4 {
            label = "Level 0 - Partition 4";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP4 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>12</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>13</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>14</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];  // scanned
            { rank=same; baseP4 }
        }
        subgraph cluster_level0_5 {
            label = "Level 0 - Partition 5";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP5 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>15</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>16</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>17</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; baseP5 }
        }
        // Base partitions for middle level Partition 2 (scanned: 2 of 3)
        subgraph cluster_level0_6 {
            label = "Level 0 - Partition 6";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP6 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>18</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>19</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>20</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#f5f5f5"];  // faded
            { rank=same; baseP6 }
        }
        subgraph cluster_level0_7 {
            label = "Level 0 - Partition 7";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP7 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>21</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>22</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>23</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];  // scanned
            { rank=same; baseP7 }
        }
        subgraph cluster_level0_8 {
            label = "Level 0 - Partition 8";
            style="rounded";
            color="gray";
            labelloc="t";
            labeljust="c";
            baseP8 [label=<
              <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                <TR>
                  <TD PORT="f0">V<sup>0</sup><sub>24</sub></TD>
                  <TD PORT="f1">V<sup>0</sup><sub>25</sub></TD>
                  <TD PORT="f2">V<sup>0</sup><sub>26</sub></TD>
                </TR>
              </TABLE>
            >, fillcolor="#7173f9"];  // scanned
            { rank=same; baseP8 }
        }

        // =============================
        // Edges: Connect query -> top level -> middle level -> base level.
        // =============================
        query -> topP;

        // Top-level -> Middle-level
        topP:f0 -> midP0 [dir=none;penwidth=0];
        topP:f1 -> midP1;
        topP:f2 -> midP2;

        // Middle-level -> Base-level
        midP0:f0 -> baseP0 [dir=none;penwidth=0];
        midP0:f1 -> baseP1 [dir=none;penwidth=0];
        midP0:f2 -> baseP2 [dir=none;penwidth=0];

        midP1:f0 -> baseP3;
        midP1:f1 -> baseP4;
        midP1:f2 -> baseP5 [dir=none;penwidth=0];

        midP2:f0 -> baseP6 [dir=none;penwidth=0];
        midP2:f1 -> baseP7;
        midP2:f2 -> baseP8;
    }

Index Memory Layout
-------------------
The following figure demonstrates how the index can be distributed across three NUMA nodes, each with three CPUs.
Partitions in each NUMA node’s memory region are pinned to specific CPUs, enabling efficient local access. Level-2
partitions are shown in green, Level-1 in blue, and Level-0 in purple. In order to distribute the top-level,
the single top-level partition is sharded across NUMA nodes.

.. graphviz::
   :caption: Hardware Layout: Index Distributed Across 3 NUMA Nodes (each with 3 CPUs)
   :name: fig-hw-layout-distributed

    digraph QuakeHW {
        compound=true;
        // Lay out nodes left-to-right
        rankdir="same";
        fontsize=18;
        node [fontname=Helvetica, style=filled];

        // =============================
        // Entire System
        // =============================
        subgraph cluster_system {
            label="Hardware: 3 NUMA Nodes, each with 3 CPUs";
            style="bold,rounded";
            color="black";
            labelloc="t";
            labeljust="c";

            // =============================
            // NUMA Node 0
            // =============================
            subgraph cluster_node0 {
                label="NUMA Node 0";
                style="rounded";
                color="gray";
                labelloc="t";
                labeljust="c";

                // Place memory (left) and CPU (right) side by side
                rankdir="same";

                // ---------- Memory Sub-Cluster ----------
                subgraph cluster_mem0 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    // One Level 2 partition
                    p2_0 [label="Partition 0\n(Level 2)", shape=box, fillcolor="#45b594"];
                    // One Level 1 partition
                    p1_0 [label="Partition 0\n(Level 1)", shape=box, fillcolor="#5aaed5"];
                    // Three Level 0 (base) partitions
                    p0_0 [label="Partition 0\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_1 [label="Partition 1\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_2 [label="Partition 2\n(Level 0)", shape=box, fillcolor="#7173f9"];
                }

                // ---------- CPU Sub-Cluster ----------
                subgraph cluster_cpus0 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    cpu0_0 [label="CPU 0", shape=box, fillcolor="#ffffff"];
                    cpu0_1 [label="CPU 1", shape=box, fillcolor="#ffffff"];
                    cpu0_2 [label="CPU 2", shape=box, fillcolor="#ffffff"];
                }

                // ---------- Pinned Edges ----------
                // Example: pin top-level partition to CPU0_0, mid-level to CPU0_1, base-levels to CPU0_2
                p2_0 -> cpu0_0 [dir=None, constraint=false];
                p1_0 -> cpu0_1 [dir=None, constraint=false];
                p0_0 -> cpu0_1 [dir=None, constraint=true];
                p0_1 -> cpu0_2 [dir=None, constraint=false];
                p0_2 -> cpu0_2 [dir=None, constraint=false];
            }

            // =============================
            // NUMA Node 1
            // =============================
            subgraph cluster_node1 {
                label="NUMA Node 1";
                style="rounded";
                color="gray";
                labelloc="t";
                labeljust="c";

                rankdir="same";

                // ---------- Memory Sub-Cluster ----------
                subgraph cluster_mem1 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    // One Level 2 partition
                    p2_1 [label="Partition 1\n(Level 2)", shape=box, fillcolor="#45b594"];
                    // One Level 1 partition
                    p1_1 [label="Partition 1\n(Level 1)", shape=box, fillcolor="#5aaed5"];
                    // Three Level 0 (base) partitions
                    p0_3 [label="Partition 3\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_4 [label="Partition 4\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_5 [label="Partition 5\n(Level 0)", shape=box, fillcolor="#7173f9"];
                }

                // ---------- CPU Sub-Cluster ----------
                subgraph cluster_cpus1 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    cpu1_0 [label="CPU 0", shape=box, fillcolor="#ffffff"];
                    cpu1_1 [label="CPU 1", shape=box, fillcolor="#ffffff"];
                    cpu1_2 [label="CPU 2", shape=box, fillcolor="#ffffff"];
                }

                // ---------- Pinned Edges ----------
                p2_1 -> cpu1_0 [dir=None, constraint=false];
                p1_1 -> cpu1_1 [dir=None, constraint=false];
                p0_3 -> cpu1_1 [dir=None, constraint=true];
                p0_4 -> cpu1_2 [dir=None, constraint=false];
                p0_5 -> cpu1_2 [dir=None, constraint=false];
            }

            // =============================
            // NUMA Node 2
            // =============================
            subgraph cluster_node2 {
                label="NUMA Node 2";
                style="rounded";
                color="gray";
                labelloc="t";
                labeljust="c";

                rankdir="same";

                // ---------- Memory Sub-Cluster ----------
                subgraph cluster_mem2 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    // One Level 2 partition
                    p2_2 [label="Partition 2\n(Level 2)", shape=box, fillcolor="#45b594"];
                    // One Level 1 partition
                    p1_2 [label="Partition 2\n(Level 1)", shape=box, fillcolor="#5aaed5"];
                    // Three Level 0 (base) partitions
                    p0_6 [label="Partition 6\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_7 [label="Partition 7\n(Level 0)", shape=box, fillcolor="#7173f9"];
                    p0_8 [label="Partition 8\n(Level 0)", shape=box, fillcolor="#7173f9"];
                }

                // ---------- CPU Sub-Cluster ----------
                subgraph cluster_cpus2 {
                    label="";
                    style="rounded";
                    color="lightgray";

                    cpu2_0 [label="CPU 0", shape=box, fillcolor="#ffffff"];
                    cpu2_1 [label="CPU 1", shape=box, fillcolor="#ffffff"];
                    cpu2_2 [label="CPU 2", shape=box, fillcolor="#ffffff"];
                }

                // ---------- Pinned Edges ----------
                p2_2 -> cpu2_0 [dir=None, constraint=false];
                p1_2 -> cpu2_1 [dir=None, constraint=false];
                p0_6 -> cpu2_1 [dir=None, constraint=true];
                p0_7 -> cpu2_2 [dir=None, constraint=false];
                p0_8 -> cpu2_2 [dir=None, constraint=false];
            }
            edge[constraint=false, style=solid];
            cpu0_0 -> cpu1_0 [ style=invis];
            cpu1_0 -> cpu2_0 [ style=invis];
        }
    }
