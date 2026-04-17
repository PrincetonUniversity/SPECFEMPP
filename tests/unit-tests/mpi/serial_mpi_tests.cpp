#include "specfem/mpi.hpp"
#include "specfem/program/context.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

/**
 * @brief Serial tests for specfem::MPI class (1 process, no MPI launcher)
 *
 * Tests non-context, single-process, and format_proc_filename behavior.
 * Death tests verify error conditions when MPI is accessed outside Context
 * scope.
 */

// Tests 1-4: Verify that API calls outside Context scope cause process exit

TEST(MPI_Serial_ContextCheck, GetRankOutsideContextDeath) {
  // Attempting to call get_rank() without a Context should cause exit
  ASSERT_DEATH(
      { specfem::MPI::get_rank(); }, "ERROR: MPI used outside Context scope");
}

TEST(MPI_Serial_ContextCheck, GetSizeOutsideContextDeath) {
  // Attempting to call get_size() without a Context should cause exit
  ASSERT_DEATH(
      { specfem::MPI::get_size(); }, "ERROR: MPI used outside Context scope");
}

TEST(MPI_Serial_ContextCheck, MainProcOutsideContextDeath) {
  // Attempting to call main_proc() without a Context should cause exit
  ASSERT_DEATH(
      { specfem::MPI::main_proc(); }, "ERROR: MPI used outside Context scope");
}

TEST(MPI_Serial_ContextCheck, SyncOutsideContextReturnsNoOp) {
  // sync() calls check_context() which returns false for excluded ranks
  // But outside Context, rank_ and size_ are both -1, so it exits
  ASSERT_DEATH(
      { specfem::MPI::sync(); }, "ERROR: MPI used outside Context scope");
}

// Test with Context

class MPI_Serial_WithContext : public ::testing::Test {
protected:
  void SetUp() override {
    std::vector<std::string> args = { "test_program" };
    context_ = std::make_unique<specfem::program::Context>(args);
  }

  void TearDown() override { context_.reset(); }

  std::unique_ptr<specfem::program::Context> context_;
};

TEST_F(MPI_Serial_WithContext, ContextLifecycleGetRank) {
  // In single-process mode, rank should be 0
  EXPECT_EQ(specfem::MPI::get_rank(), 0);
}

TEST_F(MPI_Serial_WithContext, ContextLifecycleGetSize) {
  // In single-process mode, size should be 1
  EXPECT_EQ(specfem::MPI::get_size(), 1);
}

TEST_F(MPI_Serial_WithContext, ContextLifecycleMainProc) {
  // In single-process mode, main_proc() should be true
  EXPECT_TRUE(specfem::MPI::main_proc());
}

TEST_F(MPI_Serial_WithContext, ContextLifecycleIsActive) {
  // In single-process mode, is_active() should be true
  EXPECT_TRUE(specfem::MPI::is_active());
}

TEST_F(MPI_Serial_WithContext, ContextLifecycleCommunicator) {
  // communicator() should return a valid communicator (MPI_COMM_WORLD)
  // We can't do much with it in a single-process non-MPI build,
  // but we can verify it's not MPI_COMM_NULL
#ifdef SPECFEM_ENABLE_MPI
  EXPECT_NE(specfem::MPI::communicator(), MPI_COMM_NULL);
#else
  // In non-MPI build, it's the dummy value MPI_COMM_WORLD (0)
  EXPECT_EQ(specfem::MPI::communicator(), 0);
#endif
}

// Tests for format_proc_filename in single-process mode

TEST_F(MPI_Serial_WithContext, FormatProcFilenameBasic) {
  // Single process (size=1): filename should be returned unchanged
  std::string input = "foo/bar.bin";
  std::string expected = "foo/bar.bin";
  EXPECT_EQ(specfem::MPI::format_proc_filename(input), expected);
}

TEST_F(MPI_Serial_WithContext, FormatProcFilenameNoExtension) {
  // Single process: "data/mesh" should return unchanged
  std::string input = "data/mesh";
  std::string expected = "data/mesh";
  EXPECT_EQ(specfem::MPI::format_proc_filename(input), expected);
}

TEST_F(MPI_Serial_WithContext, FormatProcFilenameNoDirectory) {
  // Single process: "file.dat" should return unchanged
  std::string input = "file.dat";
  std::string expected = "file.dat";
  EXPECT_EQ(specfem::MPI::format_proc_filename(input), expected);
}

TEST_F(MPI_Serial_WithContext, FormatProcFilenameComplexPath) {
  // Single process: complex path unchanged
  std::string input = "path/to/data/file.bin";
  std::string expected = "path/to/data/file.bin";
  EXPECT_EQ(specfem::MPI::format_proc_filename(input), expected);
}

// Test sync/sync_all in single process (should complete without error)

TEST_F(MPI_Serial_WithContext, SyncDoesNotHang) {
  // In single process, sync() should complete without hanging
  EXPECT_NO_THROW(specfem::MPI::sync());
}

TEST_F(MPI_Serial_WithContext, SyncAllDoesNotHang) {
  // In single process, sync_all() should also complete
  EXPECT_NO_THROW(specfem::MPI::sync_all());
}

// Test check_context returns true inside valid Context

TEST_F(MPI_Serial_WithContext, CheckContextReturnsTrue) {
  // Inside a valid Context, check_context() should return true
  EXPECT_TRUE(specfem::MPI::check_context());
}

// Test that Context cleanup causes subsequent calls to fail

TEST(MPI_Serial_ContextCheck, GetRankAfterContextDestructionDeath) {
  // Create context in subscope
  {
    std::vector<std::string> args = { "test_program" };
    specfem::program::Context context(args);
    // Inside scope, get_rank() works
    EXPECT_EQ(specfem::MPI::get_rank(), 0);
  }

  // After context destruction, get_rank() should fail
  ASSERT_DEATH(
      { specfem::MPI::get_rank(); }, "ERROR: MPI used outside Context scope");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
