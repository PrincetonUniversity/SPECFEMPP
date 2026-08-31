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

TEST(LinearSystemTrilinosSmoke, SolvesWithBelosGmresAndIfpack2) {
#ifdef SPECFEM_ENABLE_TRILINOS
  // Gate for the implicit solver (issue #1984): proves the install provides
  // Belos + Ifpack2 instantiations for type_real (cluster installs are
  // float-only). Throws on non-convergence or a wrong solution.
  int iterations = -1;
  EXPECT_NO_THROW(iterations =
                      specfem::linear_system::linear_solver_smoke_test());
  EXPECT_GE(iterations, 0)
      << "Expected a non-negative GMRES iteration count from the smoke solve";
#else
  EXPECT_EQ(specfem::linear_system::linear_solver_smoke_test(), -1)
      << "Expected the no-op stub to return -1 when Trilinos is disabled";
#endif
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
