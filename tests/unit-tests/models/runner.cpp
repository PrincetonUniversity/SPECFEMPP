#include "../SPECFEM_Environment.hpp"

#include <gtest/gtest.h>

// The globe model oracle requires MPI (fortran/meshfem3d_globe has no serial
// stub), so the environment is brought up with a single rank rather than run
// without a communicator at all.
int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(1));
  return RUN_ALL_TESTS();
}
