#include "SPECFEM_Environment.hpp"

#include "read_mesh_test_fixture.hpp"
#include <gtest/gtest.h>

INSTANTIATE_TEST_SUITE_P(Read2DMeshMPITests, Read2DMeshMPITest,
                         ::testing::Values("HomogeneousMediumMPI4Procs"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(4));
  return RUN_ALL_TESTS();
}
