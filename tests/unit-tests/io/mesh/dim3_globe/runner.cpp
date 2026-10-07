#include "SPECFEM_Environment.hpp"

#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(1));
  return RUN_ALL_TESTS();
}
