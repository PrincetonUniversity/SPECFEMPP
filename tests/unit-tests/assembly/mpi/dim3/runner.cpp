#include "SPECFEM_Environment.hpp"
#include "fixture.hpp"
#include "gtest/gtest.h"

INSTANTIATE_TEST_SUITE_P(Assembly3DTests, AssemblyMPI3DTest,
                         ::testing::Values("HomogeneousMediumMPI4x4"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
