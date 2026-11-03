#include "SPECFEM_Environment.hpp"
#include "gtest/gtest.h"

// Six-rank globe MPI tests: one rank per cubed-sphere chunk.
int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(6));
  return RUN_ALL_TESTS();
}
