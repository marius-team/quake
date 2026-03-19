
#include <gtest/gtest.h>
#include "quake_index.h"

using torch::Tensor;

//---------------------------------------------------------------
// Helpers
//---------------------------------------------------------------
static Tensor rand_vectors(int64_t n, int64_t d, uint64_t seed)
{
    torch::manual_seed(seed);
    return torch::randn({n, d}, torch::kFloat32);
}
static Tensor seq_ids(int64_t n, int64_t start = 0)
{
    return torch::arange(start, start + n, torch::kInt64);
}

//---------------------------------------------------------------
// Constants for all tests
//---------------------------------------------------------------
constexpr int64_t DIM   = 32;
constexpr int64_t NVEC  = 20'000;
constexpr int64_t LEAF  = 64;
constexpr int64_t MID   = 16;
constexpr int64_t ROOT  = 4;

//---------------------------------------------------------------
// 1. Build a 3‑level index, search, maintain, search again.
//---------------------------------------------------------------
TEST(QuakeMultiLevel, EndToEndMaintenance)
{
    Tensor data = rand_vectors(NVEC, DIM, 0);
    Tensor ids  = seq_ids(NVEC);

    QuakeIndex ix;
    auto build = std::make_shared<IndexBuildParams>();
    build->nlist = LEAF;
    build->metric = "l2";
    ix.build(data, ids, build);

    // add 2nd level
    auto mid = std::make_shared<IndexBuildParams>();
    mid->nlist = MID;
    mid->metric = "l2";
    ix.add_level(mid);

    // add 3rd (root) level
    auto root = std::make_shared<IndexBuildParams>();
    root->nlist = ROOT;
    root->metric = "l2";
    ix.add_level(root);

    // quick sanity on hierarchy depth
    ASSERT_NE(ix.parent_, nullptr);
    ASSERT_NE(ix.parent_->parent_, nullptr);

    auto sp = std::make_shared<SearchParams>();
    sp->k = 10;

    // baseline search
    auto baseline = ix.search(rand_vectors(50, DIM, 1), sp);
    ASSERT_EQ(baseline->ids.size(1), sp->k);

    // hammer with 2000 queries to populate hit tracker then run maintenance
    ix.search(rand_vectors(2000, DIM, 2), sp);
    auto mi = ix.maintenance();
    EXPECT_GE(mi->total_time_us, 0);
    EXPECT_EQ(ix.ntotal(), NVEC);

    // final search should still work
    auto after = ix.search(rand_vectors(50, DIM, 3), sp);
    EXPECT_EQ(after->ids.size(1), sp->k);
}

//---------------------------------------------------------------
// 2. Calling add_level() on a 1‑level index must throw.
//---------------------------------------------------------------
TEST(QuakeMultiLevel, AddLevelWithoutParentThrows)
{
    Tensor data = rand_vectors(1000, DIM, 4);
    Tensor ids  = seq_ids(1000);

    QuakeIndex flat;
    flat.build(data, ids, std::make_shared<IndexBuildParams>()); // nlist=0 ⇒ flat

    EXPECT_THROW(flat.add_level(std::make_shared<IndexBuildParams>()), std::runtime_error);
}

//---------------------------------------------------------------
// 3. Root maintenance should be a safe no‑op (flat index).
//---------------------------------------------------------------
TEST(QuakeMultiLevel, RootMaintenanceSafe)
{
    Tensor data = rand_vectors(800, DIM, 5);
    Tensor ids  = seq_ids(800);

    QuakeIndex flat;
    flat.build(data, ids, std::make_shared<IndexBuildParams>()); // nlist=0

    EXPECT_NO_THROW(flat.maintenance());
}