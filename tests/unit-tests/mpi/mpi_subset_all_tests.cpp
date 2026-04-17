#include "SPECFEM_Environment.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>

/**
 * @brief MPI tests for specfem::MPI with full subset communicator
 * (4 world processes, nprocs=4, so all ranks active - equivalent to standard
 * init)
 *
 * Verifies that when nprocs equals world size, all ranks remain active
 * and behavior matches standard initialization.
 */

class MPI_Subset_All4of4 : public ::testing::Test {
protected:
};

TEST_F(MPI_Subset_All4of4, AllRanksActive) {
  // All 4 ranks should be active when nprocs=4
  EXPECT_TRUE(specfem::MPI::is_active());
}

TEST_F(MPI_Subset_All4of4, SizeEquals4) {
  EXPECT_EQ(specfem::MPI::get_size(), 4);
}

TEST_F(MPI_Subset_All4of4, RankInRange) {
  int rank = specfem::MPI::get_rank();
  EXPECT_GE(rank, 0);
  EXPECT_LT(rank, 4);
}

TEST_F(MPI_Subset_All4of4, MainProcOnlyRank0) {
  bool is_main = specfem::MPI::main_proc();
  int rank = specfem::MPI::get_rank();

  if (rank == 0) {
    EXPECT_TRUE(is_main);
  } else {
    EXPECT_FALSE(is_main);
  }
}

TEST_F(MPI_Subset_All4of4, CheckContextReturnsTrue) {
  EXPECT_TRUE(specfem::MPI::check_context());
}

TEST_F(MPI_Subset_All4of4, SyncWorks) {
  // Barrier should complete on all ranks
  specfem::MPI::sync();
  EXPECT_TRUE(true);
}

TEST_F(MPI_Subset_All4of4, FormatProcFilenameWorks) {
  std::string filename = "data/test.bin";
  std::string result = specfem::MPI::format_proc_filename(filename);

  int rank = specfem::MPI::get_rank();
  std::string expected = "data/test/proc_" + std::to_string(rank) + ".bin";

  EXPECT_EQ(result, expected);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Create Context with nprocs=4 (full world)
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
