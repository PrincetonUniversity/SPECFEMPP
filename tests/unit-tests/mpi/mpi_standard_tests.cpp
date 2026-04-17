#include "SPECFEM_Environment.hpp"
#include "specfem/mpi.hpp"
#include <gtest/gtest.h>
#include <set>
#include <vector>

/**
 * @brief MPI tests for specfem::MPI class with standard initialization
 * (4 processes, MPI_COMM_WORLD)
 *
 * Tests rank/size queries, synchronization, and macro functionality
 * in multi-process environment.
 */

class MPI_Standard_Init : public ::testing::Test {
protected:
};

// Test rank and size queries

TEST_F(MPI_Standard_Init, GetRankInRange) {
  int rank = specfem::MPI::get_rank();
  EXPECT_GE(rank, 0);
  EXPECT_LT(rank, 4);
}

TEST_F(MPI_Standard_Init, GetSizeEquals4) {
  EXPECT_EQ(specfem::MPI::get_size(), 4);
}

TEST_F(MPI_Standard_Init, MainProcOnlyRankZero) {
  bool is_main = specfem::MPI::main_proc();
  int rank = specfem::MPI::get_rank();

  if (rank == 0) {
    EXPECT_TRUE(is_main);
  } else {
    EXPECT_FALSE(is_main);
  }
}

TEST_F(MPI_Standard_Init, IsActiveTrueAllRanks) {
  // In standard init, all ranks should be active (no filtering)
  EXPECT_TRUE(specfem::MPI::is_active());
}

TEST_F(MPI_Standard_Init, CommunicatorValid) {
  // Verify the communicator is valid by querying its size
  MPI_Comm comm = specfem::MPI::communicator();

#ifdef SPECFEM_ENABLE_MPI
  EXPECT_NE(comm, MPI_COMM_NULL);
  int size;
  SPECFEM_MPI_SAFECALL(MPI_Comm_size(comm, &size));
  EXPECT_EQ(size, 4);
#endif
}

TEST_F(MPI_Standard_Init, CheckContextReturnsTrue) {
  // Inside valid Context, check_context() should return true
  EXPECT_TRUE(specfem::MPI::check_context());
}

// Test synchronization

TEST_F(MPI_Standard_Init, SyncBarrier) {
  // All ranks should pass through sync() (MPI_Barrier)
  // Without a barrier, ranks could progress faster than others
  // We verify indirectly: if any rank hangs, the test will timeout

  int rank = specfem::MPI::get_rank();
  std::vector<int> values(4, -1);

  // Set value
  values[rank] = rank * 10;

  // Barrier — all ranks must reach this point
  specfem::MPI::sync();

  // After barrier, values array is still local, but we proved no deadlock
  EXPECT_EQ(values[rank], rank * 10);
}

TEST_F(MPI_Standard_Init, SyncAllAlias) {
  // sync_all() should behave identically to sync()
  // Both should complete without hanging
  specfem::MPI::sync_all();
  EXPECT_TRUE(true); // If we reach here, no deadlock
}

// Test macros

TEST_F(MPI_Standard_Init, MacroSafeCallSucceeds) {
  // SPECFEM_MPI_SAFECALL should wrap MPI calls successfully
#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Barrier(specfem::MPI::communicator())); // Should not
                                                                   // error
  EXPECT_TRUE(true);
#endif
}

TEST_F(MPI_Standard_Init, MacroOnRootExecutesOnRank0Only) {
  bool flag_was_set = false;

  SPECFEM_MPI_ON_ROOT({ flag_was_set = true; });

  // Only rank 0 should set the flag
  if (specfem::MPI::get_rank() == 0) {
    EXPECT_TRUE(flag_was_set);
  } else {
    EXPECT_FALSE(flag_was_set);
  }
}

// Test format_proc_filename with 4 processes

TEST_F(MPI_Standard_Init, FormatProcFilenameBasic) {
  std::string filename = "foo/bar.bin";
  std::string result = specfem::MPI::format_proc_filename(filename);

  // With 4 procs, ndigits = log10(3) + 1 = 1
  // So format should be "foo/bar/proc_N.bin" where N = rank (no padding)
  int rank = specfem::MPI::get_rank();

  std::string expected = "foo/bar/proc_" + std::to_string(rank) + ".bin";
  EXPECT_EQ(result, expected);
}

TEST_F(MPI_Standard_Init, FormatProcFilenameNoExtension) {
  std::string filename = "data/mesh";
  std::string result = specfem::MPI::format_proc_filename(filename);

  int rank = specfem::MPI::get_rank();
  std::string expected = "data/mesh/proc_" + std::to_string(rank);

  EXPECT_EQ(result, expected);
}

TEST_F(MPI_Standard_Init, FormatProcFilenameNoDirectory) {
  std::string filename = "file.dat";
  std::string result = specfem::MPI::format_proc_filename(filename);

  int rank = specfem::MPI::get_rank();
  std::string expected = "file/proc_" + std::to_string(rank) + ".dat";

  EXPECT_EQ(result, expected);
}

TEST_F(MPI_Standard_Init, FormatProcFilenameNoPadding) {
  // For 4 procs: ndigits = log10(4-1) + 1 = log10(3) + 1 = 0 + 1 = 1
  // So no padding, proc_0 through proc_3

  std::string filename = "test.bin";
  std::string result = specfem::MPI::format_proc_filename(filename);

  int rank = specfem::MPI::get_rank();
  // Should be single digit, no zero-padding
  std::string single_digit = std::to_string(rank);
  EXPECT_EQ(single_digit.length(), 1);
  EXPECT_TRUE(result.find("proc_" + single_digit) != std::string::npos);
}

TEST_F(MPI_Standard_Init, FormatProcFilenameEachRankUnique) {
  std::string filename = "data/output.bin";
  std::string my_filename = specfem::MPI::format_proc_filename(filename);

  // Gather all filenames on rank 0
  std::vector<char> my_name(256, '\0');
  std::copy(my_filename.begin(), my_filename.end(), my_name.begin());

  std::vector<char> all_names(256 * 4);

#ifdef SPECFEM_ENABLE_MPI
  SPECFEM_MPI_SAFECALL(MPI_Gather(my_name.data(), 256, MPI_CHAR,
                                  all_names.data(), 256, MPI_CHAR, 0,
                                  specfem::MPI::communicator()));
#endif

  // On rank 0, verify uniqueness
  SPECFEM_MPI_ON_ROOT({
    std::set<std::string> unique_names;
    for (int i = 0; i < 4; ++i) {
      std::string name(all_names.data() + i * 256);
      unique_names.insert(name);
    }
    EXPECT_EQ(unique_names.size(), 4);
  });

  // Barrier to ensure all ranks complete
  specfem::MPI::sync();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
