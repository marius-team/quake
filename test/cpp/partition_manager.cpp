#include <gtest/gtest.h>
#include "partition_manager.h"
#include "quake_index.h"
#include "clustering.h"

class PartitionManagerTest : public ::testing::Test {
 protected:
  std::shared_ptr<QuakeIndex> parent_;
  std::unique_ptr<PartitionManager> partition_manager_;
  int dim_ = 4;

  void SetUp() override {
    parent_ = std::make_shared<QuakeIndex>();
    partition_manager_ = std::make_unique<PartitionManager>();
  }
};

TEST_F(PartitionManagerTest, InitPartitionsSuccess) {
  auto clustering = std::make_shared<Clustering>();
  clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);

  auto v0 = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f},
                           {5.0f, 6.0f, 7.0f, 8.0f}}, torch::kFloat32);
  auto v1 = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}}, torch::kFloat32);
  auto v2 = torch::tensor({{9.0f,  9.1f,  9.2f,  9.3f},
                           {10.0f, 10.1f, 10.2f, 10.3f}}, torch::kFloat32);

  auto i0 = torch::tensor({10, 11}, torch::kInt64);
  auto i1 = torch::tensor({100}, torch::kInt64);
  auto i2 = torch::tensor({1000, 1001}, torch::kInt64);

  clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  clustering->centroids = torch::tensor({{0.0f, 0.0f, 0.0f, 0.0f},
                                         {1.0f, 1.0f, 1.0f, 1.0f},
                                         {2.0f, 2.0f, 2.0f, 2.0f}}, torch::kFloat32);
  clustering->vectors = {v0, v1, v2};
  clustering->vector_ids = {i0, i1, i2};

  // Make sure ntotal/dim/nlist match the actual data
  EXPECT_EQ(clustering->ntotal(), 5);
  EXPECT_EQ(clustering->dim(), 4);
  EXPECT_EQ(clustering->nlist(), 3);

  auto build_params = std::make_shared<IndexBuildParams>();
  parent_->build(clustering->centroids, clustering->partition_ids, build_params);

  partition_manager_->init_partitions(parent_, clustering);

  EXPECT_EQ(partition_manager_->nlist(), 3);
  EXPECT_EQ(partition_manager_->ntotal(), 5);
}

TEST_F(PartitionManagerTest, AddVectors) {
  // Init with empty partitions: 3 IDs, each empty
  auto clustering = std::make_shared<Clustering>();
  clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  clustering->centroids = torch::tensor({{0.0f, 0.0f, 0.0f, 0.0f},
                                         {1.0f, 1.0f, 1.0f, 1.0f},
                                         {2.0f, 2.0f, 2.0f, 2.0f}}, torch::kFloat32);
  clustering->vectors = {
    Tensor(),
    Tensor(),
    Tensor()
  };
  clustering->vector_ids = {
    Tensor(),
    Tensor(),
    Tensor()
  };
  auto build_params = std::make_shared<IndexBuildParams>();
  parent_->build(clustering->centroids, clustering->partition_ids, build_params);

  partition_manager_->init_partitions(parent_, clustering);
  EXPECT_EQ(partition_manager_->ntotal(), 0);

  auto new_vectors = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f},
                                    {1.1f, 1.2f, 1.3f, 1.4f},
                                    {2.1f, 2.2f, 2.3f, 2.4f}}, torch::kFloat32);
  auto new_ids = torch::tensor({10, 11, 12}, torch::kInt64);

  // Assign them to partitions [0, 0, 2]
  auto assignments = torch::tensor({0, 0, 2}, torch::kInt64);
  partition_manager_->add(new_vectors, new_ids, assignments);
  EXPECT_EQ(partition_manager_->ntotal(), 3);
}

TEST_F(PartitionManagerTest, RemoveVectors) {
  auto clustering = std::make_shared<Clustering>();
  clustering->partition_ids = torch::tensor({0, 1}, torch::kInt64);

  auto v0 = torch::tensor({{1.0f,1.0f,1.0f,1.0f},
                           {2.0f,2.0f,2.0f,2.0f}}, torch::kFloat32);
  auto v1 = torch::tensor({{9.0f,9.1f,9.2f,9.3f}}, torch::kFloat32);

  auto i0 = torch::tensor({10, 11}, torch::kInt64);
  auto i1 = torch::tensor({99}, torch::kInt64);

  clustering->centroids = torch::tensor({{0.0f, 0.0f, 0.0f, 0.0f},
                                         {1.0f, 1.0f, 1.0f, 1.0f}}, torch::kFloat32);
  clustering->partition_ids = torch::tensor({0, 1}, torch::kInt64);
  clustering->vectors = {v0, v1};
  clustering->vector_ids = {i0, i1};
  parent_->build(clustering->centroids, clustering->partition_ids, std::make_shared<IndexBuildParams>());

  partition_manager_->init_partitions(parent_, clustering);
  EXPECT_EQ(partition_manager_->ntotal(), 3);

  partition_manager_->remove(torch::tensor({11, 99}, torch::kInt64));
  EXPECT_EQ(partition_manager_->ntotal(), 1);
}

TEST_F(PartitionManagerTest, ThrowsIfPartitionsNotInitted) {
  auto new_vectors = torch::randn({5, dim_}, torch::kFloat32);
  auto new_ids = torch::arange(5, torch::kInt64);
  EXPECT_THROW(partition_manager_->add(new_vectors, new_ids), std::runtime_error);

  auto remove_ids = torch::tensor({100, 101}, torch::kInt64);
  EXPECT_THROW(partition_manager_->remove(remove_ids), std::runtime_error);
}

TEST_F(PartitionManagerTest, RefinePartitions) {

  int64_t n_total = 1000;
  int64_t n_list = 5;

  Tensor vectors = torch::randn({n_total, dim_}, torch::kFloat32);
  Tensor vector_ids = torch::arange(n_total, torch::kInt64);
  Tensor initial_centroids = vectors.index_select(0, torch::randperm(n_total).slice(0, 0, n_list));

  // get the assignments
  Tensor dists = torch::cdist(vectors, initial_centroids, 2);
  Tensor assignments = torch::argmin(dists, 1);

  // init the clustering
  auto clustering = std::make_shared<Clustering>();
  clustering->centroids = initial_centroids;
  clustering->partition_ids = torch::arange(n_list, torch::kInt64);
  clustering->vectors = {};
  clustering->vector_ids = {};

  for (int i = 0; i < n_list; i++) {
    Tensor mask = assignments == i;
    Tensor ids = vector_ids.masked_select(mask);
    Tensor vecs = vectors.index_select(0, mask.nonzero().squeeze(1));
    clustering->vectors.push_back(vecs);
    clustering->vector_ids.push_back(ids);
  }

  parent_->build(clustering->centroids, clustering->partition_ids, std::make_shared<IndexBuildParams>());
  partition_manager_->init_partitions(parent_, clustering);

  // run refinement and make sure the number of vectors and partitions have not changed
  Tensor p_sizes_before = partition_manager_->get_partition_sizes(torch::arange(n_list, torch::kInt64));
  partition_manager_->refine_partitions({}, 0); // reassign all vectors
  ASSERT_EQ(partition_manager_->ntotal(), n_total);
  ASSERT_EQ(partition_manager_->nlist(), n_list);
  Tensor p_sizes_after = partition_manager_->get_partition_sizes(torch::arange(n_list, torch::kInt64));
  ASSERT_TRUE(torch::allclose(p_sizes_before, p_sizes_after));

  partition_manager_->refine_partitions(torch::tensor({0, 1, 2}, torch::kInt64), 0); // reassign partitions 0, 1, 2
  ASSERT_EQ(partition_manager_->ntotal(), n_total);
  ASSERT_EQ(partition_manager_->nlist(), n_list);

  partition_manager_->refine_partitions(torch::tensor({0, 1, 2}, torch::kInt64), 3); // run 3 iterations of refinement on partitions 0, 1, 2
  ASSERT_EQ(partition_manager_->ntotal(), n_total);
  ASSERT_EQ(partition_manager_->nlist(), n_list);
}

// Test: Verify that add_partitions correctly adds new partitions.
TEST_F(PartitionManagerTest, AddPartitionsTest) {
  // Initialize with a base clustering containing 3 empty partitions.
  auto base_clustering = std::make_shared<Clustering>();
  base_clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  base_clustering->centroids = torch::rand({3, dim_}, torch::kFloat32);
  base_clustering->vectors = {torch::empty({0, dim_}, torch::kFloat32),
                              torch::empty({0, dim_}, torch::kFloat32),
                              torch::empty({0, dim_}, torch::kFloat32)};
  base_clustering->vector_ids = {torch::empty({0}, torch::kInt64),
                                  torch::empty({0}, torch::kInt64),
                                  torch::empty({0}, torch::kInt64)};
  parent_->build(base_clustering->centroids, base_clustering->partition_ids, std::make_shared<IndexBuildParams>());
  partition_manager_->init_partitions(parent_, base_clustering);

  // Create new clustering for adding partitions.
  auto clustering = std::make_shared<Clustering>();
  clustering->partition_ids = torch::tensor({3, 4}, torch::kInt64);
  clustering->centroids = torch::tensor({{3.0f, 3.0f, 3.0f, 3.0f},
                                          {4.0f, 4.0f, 4.0f, 4.0f}}, torch::kFloat32);
  clustering->vectors = {
    torch::tensor({{3.1f, 3.1f, 3.1f, 3.1f}}, torch::kFloat32),
    torch::tensor({{4.1f, 4.1f, 4.1f, 4.1f}}, torch::kFloat32)
  };
  clustering->vector_ids = {
    torch::tensor({31}, torch::kInt64),
    torch::tensor({41}, torch::kInt64)
  };

  partition_manager_->add_partitions(clustering);
  // Base clustering had 3 partitions; now 2 more are added.
  EXPECT_EQ(partition_manager_->nlist(), 3 + 2);
  Tensor p_ids = partition_manager_->get_partition_ids();
  auto p_ids_acc = p_ids.accessor<int64_t,1>();
  EXPECT_EQ(partition_manager_->partition_store_->list_size(3), 1);
  EXPECT_EQ(partition_manager_->partition_store_->list_size(4), 1);
}

// Test: Verify that select_partitions returns the expected data.
TEST_F(PartitionManagerTest, SelectPartitionsTest) {
  // Initialize with a base clustering containing 3 partitions.
  auto base_clustering = std::make_shared<Clustering>();
  base_clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  base_clustering->centroids = torch::rand({3, dim_}, torch::kFloat32);
  base_clustering->vectors = {torch::rand({10, dim_}, torch::kFloat32),
                              torch::rand({8, dim_}, torch::kFloat32),
                              torch::rand({12, dim_}, torch::kFloat32)};
  base_clustering->vector_ids = {torch::arange(10, torch::kInt64),
                                  torch::arange(10, 18, torch::kInt64),
                                  torch::arange(18, 30, torch::kInt64)};
  parent_->build(base_clustering->centroids, base_clustering->partition_ids, std::make_shared<IndexBuildParams>());
  partition_manager_->init_partitions(parent_, base_clustering);

  Tensor p_ids = partition_manager_->get_partition_ids();
  auto selected = partition_manager_->select_partitions(p_ids, true);
  Tensor parent_centroids = parent_->get(p_ids);
  EXPECT_TRUE(torch::allclose(selected->centroids, parent_centroids));
}

// Test: Verify that get_partition_sizes returns correct sizes.
TEST_F(PartitionManagerTest, GetPartitionSizesTest) {
  // Initialize with a base clustering containing 3 partitions.
  auto base_clustering = std::make_shared<Clustering>();
  base_clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  base_clustering->centroids = torch::rand({3, dim_}, torch::kFloat32);
  base_clustering->vectors = {torch::rand({5, dim_}, torch::kFloat32),
                              torch::rand({7, dim_}, torch::kFloat32),
                              torch::rand({9, dim_}, torch::kFloat32)};
  base_clustering->vector_ids = {torch::arange(5, torch::kInt64),
                                  torch::arange(5, 12, torch::kInt64),
                                  torch::arange(12, 21, torch::kInt64)};
  parent_->build(base_clustering->centroids, base_clustering->partition_ids, std::make_shared<IndexBuildParams>());
  partition_manager_->init_partitions(parent_, base_clustering);

  Tensor p_ids = partition_manager_->get_partition_ids();
  Tensor sizes = partition_manager_->get_partition_sizes(p_ids);
  auto p_ids_acc = p_ids.accessor<int64_t,1>();
  auto sizes_acc = sizes.accessor<int64_t,1>();
  for (int i = 0; i < p_ids.size(0); i++) {
    EXPECT_EQ(sizes_acc[i], partition_manager_->partition_store_->list_size(p_ids_acc[i]));
  }
}

// Test: Save and load roundtrip.
TEST_F(PartitionManagerTest, SaveLoadRoundTripTest) {
  // Initialize with a base clustering containing 3 partitions.
  auto base_clustering = std::make_shared<Clustering>();
  base_clustering->partition_ids = torch::tensor({0, 1, 2}, torch::kInt64);
  base_clustering->centroids = torch::rand({3, dim_}, torch::kFloat32);
  base_clustering->vectors = {torch::rand({5, dim_}, torch::kFloat32),
                              torch::rand({7, dim_}, torch::kFloat32),
                              torch::rand({9, dim_}, torch::kFloat32)};
  base_clustering->vector_ids = {torch::arange(5, torch::kInt64),
                                  torch::arange(5, 12, torch::kInt64),
                                  torch::arange(12, 21, torch::kInt64)};
  parent_->build(base_clustering->centroids, base_clustering->partition_ids, std::make_shared<IndexBuildParams>());
  partition_manager_->init_partitions(parent_, base_clustering);

  std::string filename = "temp_partmgr.dat";
  EXPECT_NO_THROW(partition_manager_->save(filename));

  auto new_mgr = std::make_unique<PartitionManager>();
  EXPECT_NO_THROW(new_mgr->load(filename));

  EXPECT_EQ(new_mgr->ntotal(), partition_manager_->ntotal());
  EXPECT_EQ(new_mgr->nlist(), partition_manager_->nlist());

  Tensor old_ids = partition_manager_->get_partition_ids();
  Tensor new_ids = new_mgr->get_partition_ids();

  // sort the ids, they may be in different order
  old_ids = std::get<0>(torch::sort(old_ids));
  new_ids = std::get<0>(torch::sort(new_ids));

  EXPECT_TRUE(torch::allclose(old_ids, new_ids));

  remove(filename.c_str());
}