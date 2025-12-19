#include "../SPECFEM_Environment.hpp"
#include "dim3/test_fixture.hpp"
#include "gtest/gtest.h"

INSTANTIATE_TEST_SUITE_P(Assembly3DTests, Assembly3DTest,
                         ::testing::Values("EightNodeElastic"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
