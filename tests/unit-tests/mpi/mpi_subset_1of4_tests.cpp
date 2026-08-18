#include "SPECFEM_Environment.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>

/**
 * @brief MPI tests for specfem::MPI with most restrictive subset communicator
 * (4 world processes, nprocs=1, so only rank 0 active, ranks 1-3 excluded)
 *
 * Tests the extreme case where only 1 rank is active.
 */

class MPI_Subset_1of4 : public ::testing::Test {
protected:
};

TEST_F(MPI_Subset_1of4, OnlyRank0Active) {
  // World rank 0: is_active() == true
  // World ranks 1-3: is_active() == false

  bool active = specfem::MPI::is_active();
  int world_rank;
#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
#else
  world_rank = 0;
#endif

  if (world_rank == 0) {
    EXPECT_TRUE(active);
  } else {
    EXPECT_FALSE(active);
  }
}

TEST_F(MPI_Subset_1of4, Rank0HasRankZeroAndSize1) {
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for rank 0 only";
  }

  EXPECT_EQ(specfem::MPI::get_rank(), 0);
  EXPECT_EQ(specfem::MPI::get_size(), 1);
}

TEST_F(MPI_Subset_1of4, Rank0IsMainProc) {
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for rank 0 only";
  }

  EXPECT_TRUE(specfem::MPI::main_proc());
}

TEST_F(MPI_Subset_1of4, FormatFilenameSize1ReturnedUnchanged) {
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for rank 0 only";
  }

  // With size=1, format_proc_filename should return filename unchanged
  std::string input = "foo/bar.bin";
  std::string result = specfem::MPI::format_proc_filename(input);
  EXPECT_EQ(result, input);
}

TEST_F(MPI_Subset_1of4, ExcludedRanksInactive) {
  // Ranks 1-3 should be inactive
  int world_rank;
#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
#else
  world_rank = 0;
#endif

  if (world_rank > 0) {
    EXPECT_FALSE(specfem::MPI::is_active());
    EXPECT_FALSE(specfem::MPI::check_context());
  } else {
    // Rank 0 is active
    EXPECT_TRUE(specfem::MPI::is_active());
    EXPECT_TRUE(specfem::MPI::check_context());
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Create Context with nprocs=1 (subset of 4)
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(1));
  return RUN_ALL_TESTS();
}
