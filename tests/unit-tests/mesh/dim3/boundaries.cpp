#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/setup.hpp"
#include "test_fixture.hpp"

namespace specfem::test_configuration {

/**
 * @brief Expected counts and face entries for a 3D mesh boundary check.
 */
struct ExpectedBoundary3DEntry {
  int element_id;
  bool expect_in_absorbing;
  specfem::mesh_entity::dim3::type face_type;
};

struct ExpectedBoundaries3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  int n_absorbing;    ///< Expected number of absorbing boundary faces
  int n_free_surface; ///< Expected number of top-surface faces
  std::vector<ExpectedBoundary3DEntry> entries;

  ExpectedBoundaries3D(int n_absorbing, int n_free_surface,
                       std::initializer_list<ExpectedBoundary3DEntry> e)
      : n_absorbing(n_absorbing), n_free_surface(n_free_surface), entries(e) {}

  void check(const specfem::mesh::boundaries<dimension> &boundaries) const {
    const auto &abs = boundaries.absorbing_boundary;
    const auto &fs = boundaries.acoustic_free_surface;

    EXPECT_EQ(abs.nelements, n_absorbing)
        << "absorbing_boundary element count mismatch";
    EXPECT_EQ(fs.nelem_acoustic_surface, n_free_surface)
        << "acoustic_free_surface element count mismatch";

    for (const auto &entry : entries) {
      if (entry.expect_in_absorbing) {
        bool found = false;
        for (int i = 0; i < abs.nelements; ++i) {
          if (abs.index_mapping(i) == entry.element_id &&
              abs.type(i) == entry.face_type) {
            found = true;
            break;
          }
        }
        EXPECT_TRUE(found) << "Absorbing boundary face not found for element "
                           << entry.element_id << " with face type "
                           << specfem::mesh_entity::dim3::to_string(
                                  entry.face_type);
      } else {
        bool found = false;
        for (int i = 0; i < fs.nelem_acoustic_surface; ++i) {
          if (fs.index_mapping(i) == entry.element_id &&
              fs.type(i) == entry.face_type) {
            found = true;
            break;
          }
        }
        EXPECT_TRUE(found) << "Free-surface face not found for element "
                           << entry.element_id << " with face type "
                           << specfem::mesh_entity::dim3::to_string(
                                  entry.face_type);
      }
    }
  }
};

} // namespace specfem::test_configuration

using namespace specfem::test_configuration;

// 2×2×2 mesh: 6 directions × 4 faces = 24 total boundary faces.
// 5 non-top directions × 4 = 20 absorbing, 1 top direction × 4 = 4 free
// surface.
static const std::unordered_map<std::string, ExpectedBoundaries3D>
    expected_map = {
      { "EightNodeElastic",
        ExpectedBoundaries3D(
            20, 4,
            {
                // Non-top (absorbing) faces
                { 0, true, specfem::mesh_entity::dim3::type::left },
                { 1, true, specfem::mesh_entity::dim3::type::right },
                { 0, true, specfem::mesh_entity::dim3::type::front },
                { 2, true, specfem::mesh_entity::dim3::type::back },
                { 0, true, specfem::mesh_entity::dim3::type::bottom },
                // Top (free-surface) face
                { 4, false, specfem::mesh_entity::dim3::type::top },
            }) }
    };

TEST_P(Mesh3DTest, Boundaries) {
  const auto &param_name = GetParam();
  if (expected_map.find(param_name) == expected_map.end()) {
    GTEST_SKIP() << "No ground truth defined for test case: " << param_name;
    return;
  }

  const auto &mesh = getMesh();
  expected_map.at(param_name).check(mesh.boundaries);
}
