#include "SPECFEM_Environment.hpp"
#include "fixture.hpp"
#include "gtest/gtest.h"

// 8-rank MPI assembly tests. Uses the HomogeneousElasticMPI2x2x2 fixture: a
// 4x4x4 elastic cube decomposed with METIS into 8 partitions, which produces
// the general set of MPI interfaces (top/bottom faces, horizontal edges, and
// single-node corner connections) with mixed orientations per neighbor. This
// exercises the cross-rank connection pairing in the packer/unpacker that the
// structured 4-rank fixtures cannot. See the fixture's provenance/README.md.
INSTANTIATE_TEST_SUITE_P(Assembly3DTests8Proc, AssemblyMPI3DTest,
                         ::testing::Values("HomogeneousElasticMPI2x2x2"));

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment(8));
  return RUN_ALL_TESTS();
}
