#include "specfem/program/abort.hpp"
#include "specfem/program/context.hpp"
#include <gtest/gtest.h>
#include <string>

// Test abort without context (should call std::exit)
TEST(AbortTests, AbortWithoutContext) {
  // ASSERT_DEATH expects the statement to terminate the process
  // The second argument is a regex that should match the error output
  ASSERT_DEATH(
      { specfem::program::abort("Test error without context", 42); },
      "ERROR: Test error without context");
}

// Test abort without message
TEST(AbortTests, AbortWithoutMessage) {
  ASSERT_DEATH({ specfem::program::abort("", 30); }, ".*");
}

// Test abort with context (MPI initialized)
// Note: This test is more complex because MPI_Abort behaves differently
TEST(AbortTests, AbortWithContext) {
  // The death test will fork, create a context, and abort
  // We just check that it dies, as the exact behavior depends on MPI
  ASSERT_DEATH(
      {
        std::vector<std::string> args = { "test_program" };
        specfem::program::Context context(args);
        specfem::program::abort("Test error with context", 42);
      },
      ".*"); // Match any output since MPI_Abort output varies
}

// Test abort macro with file and line information
TEST(AbortTests, AbortMacroWithLocation) {
  ASSERT_DEATH(
      { specfem_abort("Test error with location", 50); },
      "abort_tests\\.cpp:[0-9]+: Test error with location");
}

// Test abort with context and custom error code
TEST(AbortTests, AbortWithCustomErrorCode) {
  ASSERT_DEATH(
      {
        std::vector<std::string> args = { "test_program" };
        specfem::program::Context context(args);
        specfem::program::abort("Custom error code test", 99);
      },
      ".*"); // Match any output since MPI_Abort output varies
}

// Test that abort actually terminates (doesn't just return)
TEST(AbortTests, AbortDoesNotReturn) {
  ASSERT_DEATH(
      {
        specfem::program::abort("This should terminate");
        // This line should never execute
        std::cerr << "This should never be printed" << std::endl;
      },
      "ERROR: This should terminate");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Note: ASSERT_DEATH tests run in a forked process, so they won't
  // interfere with each other or the main test process

  return RUN_ALL_TESTS();
}
