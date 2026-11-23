#include <array>
#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>
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

/**
 * @brief Expected wavefield-injection faces for a 3D mesh boundary check.
 */
struct ExpectedInjection3D {
  specfem::mesh::InjectionSurfaceKind kind =
      specfem::mesh::InjectionSurfaceKind::none;
  std::array<bool, 6> sides = {}; ///< xmin, xmax, ymin, ymax, bottom, top
  int n_faces = 0;                ///< Expected number of injection faces
  /// Faces that must be on the injection surface: (element, face type)
  std::vector<std::pair<int, specfem::mesh_entity::dim3::type>> entries;
};

struct ExpectedBoundaries3D {
  constexpr static specfem::element::dimension_tag dimension =
      specfem::element::dimension_tag::dim3;

  int n_absorbing;    ///< Expected number of absorbing boundary faces
  int n_free_surface; ///< Expected number of top-surface faces
  std::vector<ExpectedBoundary3DEntry> entries;
  ExpectedInjection3D injection; ///< Expected injection faces (none default)

  ExpectedBoundaries3D(int n_absorbing, int n_free_surface,
                       std::initializer_list<ExpectedBoundary3DEntry> e,
                       ExpectedInjection3D injection = {})
      : n_absorbing(n_absorbing), n_free_surface(n_free_surface), entries(e),
        injection(std::move(injection)) {}

  void check(const specfem::mesh::boundaries<dimension> &boundaries) const {
    const auto &abs = boundaries.absorbing_boundary;
    const auto &fs = boundaries.acoustic_free_surface;
    const auto &inj = boundaries.injection_surface;

    EXPECT_EQ(abs.nelements, n_absorbing)
        << "absorbing_boundary element count mismatch";
    EXPECT_EQ(fs.nelem_acoustic_surface, n_free_surface)
        << "acoustic_free_surface element count mismatch";

    EXPECT_EQ(inj.kind, injection.kind) << "injection_surface kind mismatch";
    EXPECT_EQ(inj.sides, injection.sides) << "injection_surface sides mismatch";
    EXPECT_EQ(inj.nfaces, injection.n_faces)
        << "injection_surface face count mismatch";

    const auto in_absorbing =
        [&abs](const int ispec, const specfem::mesh_entity::dim3::type face) {
          for (int i = 0; i < abs.nelements; ++i) {
            if (abs.index_mapping(i) == ispec && abs.type(i) == face)
              return true;
          }
          return false;
        };

    // v1 invariant: every injection face is an absorbing face, on a box side
    for (int i = 0; i < inj.nfaces; ++i) {
      EXPECT_TRUE(in_absorbing(inj.index_mapping(i), inj.type(i)))
          << "Injection face is not an absorbing face: element "
          << inj.index_mapping(i) << ", face "
          << specfem::mesh_entity::dim3::to_string(inj.type(i));
      EXPECT_EQ(inj.side(i), 1) << "Box-side injection face must have side +1";
    }

    for (const auto &[ispec, face] : injection.entries) {
      bool found = false;
      for (int i = 0; i < inj.nfaces; ++i) {
        if (inj.index_mapping(i) == ispec && inj.type(i) == face) {
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Injection face not found for element " << ispec
                         << " with face type "
                         << specfem::mesh_entity::dim3::to_string(face);
    }

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
            }) },
      // Generated with STACEY_ABSORBING_CONDITIONS = .false. (no Stacey impl
      // for Cosserat media): no absorbing faces, only the 4 top faces.
      { "EightNodeElasticCosserat",
        ExpectedBoundaries3D(
            0, 4,
            {
                // Top (free-surface) face
                { 4, false, specfem::mesh_entity::dim3::type::top },
            }) },
      // Same mesh as EightNodeElastic with STACEY_AS_INJECTION_BOUNDARY =
      // .true.: all 20 Stacey faces (xmin…bottom) are injection faces.
      { "EightNodeElasticInjectionStacey",
        ExpectedBoundaries3D(
            20, 4,
            {
                { 0, true, specfem::mesh_entity::dim3::type::left },
                { 1, true, specfem::mesh_entity::dim3::type::right },
                { 0, true, specfem::mesh_entity::dim3::type::front },
                { 2, true, specfem::mesh_entity::dim3::type::back },
                { 0, true, specfem::mesh_entity::dim3::type::bottom },
                { 4, false, specfem::mesh_entity::dim3::type::top },
            },
            { specfem::mesh::InjectionSurfaceKind::box_sides,
              { true, true, true, true, true, false },
              20,
              {
                  { 0, specfem::mesh_entity::dim3::type::left },
                  { 1, specfem::mesh_entity::dim3::type::right },
                  { 0, specfem::mesh_entity::dim3::type::front },
                  { 2, specfem::mesh_entity::dim3::type::back },
                  { 0, specfem::mesh_entity::dim3::type::bottom },
              } }) },
      // Same mesh with STACEY_AS_INJECTION_BOUNDARY = .false. and only
      // INJECTION_XMIN and INJECTION_BOTTOM set: 4 + 4 = 8 injection faces.
      { "EightNodeElasticInjectionSides",
        ExpectedBoundaries3D(
            20, 4,
            {
                { 0, true, specfem::mesh_entity::dim3::type::left },
                { 1, true, specfem::mesh_entity::dim3::type::right },
                { 0, true, specfem::mesh_entity::dim3::type::bottom },
                { 4, false, specfem::mesh_entity::dim3::type::top },
            },
            { specfem::mesh::InjectionSurfaceKind::box_sides,
              { true, false, false, false, true, false },
              8,
              {
                  { 0, specfem::mesh_entity::dim3::type::left },
                  { 0, specfem::mesh_entity::dim3::type::bottom },
              } }) }
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
