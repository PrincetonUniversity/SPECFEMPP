#include "../../SPECFEM_Environment.hpp"

#include "test_fixture.hpp"
#include <gtest/gtest.h>
#include <string>

INSTANTIATE_TEST_SUITE_P(Mesh3DTests, Mesh3DTest,
                         ::testing::Values("EightNodeElastic"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
