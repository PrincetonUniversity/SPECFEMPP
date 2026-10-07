#include <gtest/gtest.h>
#include <string>
#include <unordered_map>

#include "test_fixture.hpp"

namespace specfem::test_configuration {

struct ExpectedMeshSize {
  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of GLL points in z direction
  int nglly; ///< Number of GLL points in y direction
  int ngllx; ///< Number of GLL points in x direction

  ExpectedMeshSize(int nspec, int ngllz, int nglly, int ngllx)
      : nspec(nspec), ngllz(ngllz), nglly(nglly), ngllx(ngllx) {}
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

std::unordered_map<std::string, ExpectedMeshSize> expected_mesh_sizes = {
  { "EightNodeElastic", ExpectedMeshSize(8, 5, 5, 5) },
  { "EightNodeElasticCosserat", ExpectedMeshSize(8, 5, 5, 5) },
  // Additional test cases can be added here
};

TEST_P(Mesh3DTest, MeshSize) {
  const auto &param_name = GetParam();
  if (expected_mesh_sizes.find(param_name) == expected_mesh_sizes.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name
                 << std::endl;
    return;
  }

  const auto &mesh = getMesh();
  const auto &expected_size = expected_mesh_sizes.at(param_name);

  EXPECT_EQ(mesh.nspec, expected_size.nspec)
      << "Number of spectral elements mismatch. Expected: "
      << expected_size.nspec << ", Got: " << mesh.nspec << std::endl;
  EXPECT_EQ(mesh.element_grid.ngllz, expected_size.ngllz)
      << "Number of GLL points in z direction mismatch. Expected: "
      << expected_size.ngllz << ", Got: " << mesh.element_grid.ngllz
      << std::endl;
  EXPECT_EQ(mesh.element_grid.nglly, expected_size.nglly)
      << "Number of GLL points in y direction mismatch. Expected: "
      << expected_size.nglly << ", Got: " << mesh.element_grid.nglly
      << std::endl;
  EXPECT_EQ(mesh.element_grid.ngllx, expected_size.ngllx)
      << "Number of GLL points in x direction mismatch. Expected: "
      << expected_size.ngllx << ", Got: " << mesh.element_grid.ngllx
      << std::endl;

  return;
}
