#include "../../../SPECFEM_Environment.hpp"

#include "test_fixture.hpp"
#include <gtest/gtest.h>
#include <string>

INSTANTIATE_TEST_SUITE_P(MPIMesh2DTests, MPIMesh2DTest,
                         ::testing::Values("HomogeneousMediumMPI4Procs"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
