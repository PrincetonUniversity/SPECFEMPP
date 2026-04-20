#include "SPECFEM_Environment.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>

/**
 * @brief MPI tests for specfem::MPI with subset communicator
 * (4 world processes, nprocs=2, so ranks 0-1 active, 2-3 excluded)
 *
 * Tests that the subset communicator correctly filters ranks and provides
 * appropriate behavior for excluded ranks.
 */

class MPI_Subset_2of4 : public ::testing::Test {
protected:
};

TEST_F(MPI_Subset_2of4, ActiveRanksCorrect) {
  // Ranks 0-1: is_active() == true
  // Ranks 2-3: is_active() == false (MPI_COMM_NULL)

  bool active = specfem::MPI::is_active();
  int world_rank;
#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
#else
  world_rank = 0;
#endif

  if (world_rank < 2) {
    EXPECT_TRUE(active);
  } else {
    EXPECT_FALSE(active);
  }
}

TEST_F(MPI_Subset_2of4, ActiveRankAndSize) {
  // For active ranks (0-1):
  //   - get_rank() should be 0 or 1
  //   - get_size() should be 2
  // For excluded ranks (2-3):
  //   - check_context() returns false, so API calls are no-ops or skip
  //   - get_rank() and get_size() should not be called without first checking

  if (specfem::MPI::is_active()) {
    int rank = specfem::MPI::get_rank();
    int size = specfem::MPI::get_size();

    EXPECT_GE(rank, 0);
    EXPECT_LT(rank, 2);
    EXPECT_EQ(size, 2);
  } else {
    // Excluded rank: cannot safely call get_rank() or get_size()
    // They call check_context() which returns false and doesn't abort
    // but the returned values are stale (rank_ and size_ are -1)
    EXPECT_FALSE(specfem::MPI::check_context());
  }
}

TEST_F(MPI_Subset_2of4, ExcludedRanksCommunicatorNull) {
  // For excluded ranks: calling communicator() will abort (correct behavior)
  // Only active ranks can safely call communicator()

  if (specfem::MPI::is_active()) {
    MPI_Comm comm = specfem::MPI::communicator();
#ifdef SPECFEM_ENABLE_MPI
    EXPECT_NE(comm, MPI_COMM_NULL); // Active ranks have valid communicator
#endif
  } else {
    // Excluded rank: cannot call communicator(), it will abort
    GTEST_SKIP() << "Excluded rank: communicator() would abort";
  }
}

TEST_F(MPI_Subset_2of4, CheckContextBehavior) {
  // For active ranks: check_context() returns true
  // For excluded ranks: check_context() returns false (no abort)

  bool is_active = specfem::MPI::is_active();
  bool check_result = specfem::MPI::check_context();

  if (is_active) {
    EXPECT_TRUE(check_result);
  } else {
    EXPECT_FALSE(check_result);
  }
}

TEST_F(MPI_Subset_2of4, MainProcInSubset) {
  // Active ranks: main_proc() true only for rank 0
  // Excluded ranks: calling main_proc() will abort (correct behavior)

  if (specfem::MPI::is_active()) {
    bool is_main = specfem::MPI::main_proc();
    int rank = specfem::MPI::get_rank();

    if (rank == 0) {
      EXPECT_TRUE(is_main);
    } else {
      EXPECT_FALSE(is_main);
    }
  } else {
    // Excluded rank: cannot call main_proc(), it will abort
    GTEST_SKIP() << "Excluded rank: main_proc() would abort";
  }
}

TEST_F(MPI_Subset_2of4, ExcludedRanksSkipTests) {
  // Excluded ranks should skip all subsequent tests that require active
  // communicator. Test this by using GTEST_SKIP() pattern.

  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for active ranks in subset communicator";
  }

  // If we reach here, we're an active rank
  EXPECT_TRUE(specfem::MPI::is_active());
}

// Tests that only run on active ranks

TEST_F(MPI_Subset_2of4, ActiveRanksSyncWorks) {
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for active ranks";
  }

  // Barrier among active ranks should complete
  specfem::MPI::sync();
  EXPECT_TRUE(true);
}

TEST_F(MPI_Subset_2of4, FormatProcFilenameInSubset) {
  if (!specfem::MPI::is_active()) {
    GTEST_SKIP() << "Test designed for active ranks";
  }

  std::string filename = "data/output.bin";
  std::string result = specfem::MPI::format_proc_filename(filename);

  int rank = specfem::MPI::get_rank();
  int size = specfem::MPI::get_size();

  // With 2 active ranks, ndigits = log10(1) + 1 = 0 + 1 = 1
  // Format: "data/output/proc_N.bin" where N = rank (0 or 1)

  std::string expected = "data/output/proc_" + std::to_string(rank) + ".bin";
  EXPECT_EQ(result, expected);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Create Context with nprocs=2 (subset of 4)
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(2));
  return RUN_ALL_TESTS();
}
