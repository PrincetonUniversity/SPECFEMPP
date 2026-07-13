#include "../SPECFEM_Environment.hpp"
#include "specfem/linear_system/trilinos_smoke.hpp"
#include <gtest/gtest.h>

TEST(LinearSystemTrilinosSmoke, BuildsAndFillsOneByOneMatrix) {
#ifdef SPECFEM_ENABLE_TRILINOS
  EXPECT_EQ(specfem::linear_system::trilinos_smoke_test(), 1)
      << "Expected the 1x1 Tpetra::CrsMatrix to report a single global row";
#else
  EXPECT_EQ(specfem::linear_system::trilinos_smoke_test(), 0)
      << "Expected the no-op stub to return 0 when Trilinos is disabled";
#endif
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
