#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mesh_entity.hpp"
#include <gtest/gtest.h>
#include <ostream>
#include <stdexcept>
#include <string>

namespace globe_checks_test_impl {

/**
 * @brief One absorbing-boundary case: a two-element mesh with a single Stacey
 * face on element 0.
 */
struct StaceyCase {
  int nchunks;
  specfem::element::region_tag stacey_region;
  bool expect_throw;
  std::string name;
};

std::ostream &operator<<(std::ostream &os, const StaceyCase &params) {
  return os << params.name;
}

/**
 * @brief A two-element globe mesh with no interfaces or surfaces, so only the
 * chunk-count and Stacey rules have anything to check.
 *
 * @param nchunks Chunk count to record
 * @param stacey_region Region of element 0, which owns the Stacey face
 * @param nstacey Number of Stacey faces (0 or 1)
 */
specfem::mesh::globe3d_mesh
make_mesh(const int nchunks, const specfem::element::region_tag stacey_region,
          const int nstacey) {
  specfem::mesh::globe3d_mesh mesh;
  mesh.nspec = 2;
  mesh.adjacency_graph =
      specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>(
          mesh.nspec);
  mesh.globe.model_config.nchunks = nchunks;
  mesh.globe.element_context.resize(mesh.nspec);
  mesh.globe.element_context[0].region = stacey_region;
  mesh.globe.element_context[1].region =
      specfem::element::region_tag::crust_mantle;

  mesh.boundaries.absorbing_boundary =
      specfem::mesh::absorbing_boundary<specfem::element::dimension_tag::dim3>(
          nstacey);
  if (nstacey > 0) {
    mesh.boundaries.absorbing_boundary.index_mapping(0) = 0;
    mesh.boundaries.absorbing_boundary.type(0) =
        specfem::mesh_entity::dim3::type::left;
  }
  return mesh;
}

} // namespace globe_checks_test_impl

class GlobeStaceyConsistencyTest
    : public ::testing::TestWithParam<globe_checks_test_impl::StaceyCase> {};

// The globe reader cannot produce Stacey faces -- the database has no block for
// them -- so the mesher's Stacey rules are exercised on a hand-built mesh.
TEST_P(GlobeStaceyConsistencyTest, EnforcesMesherStaceyRules) {
  const auto &params = GetParam();
  const auto mesh = globe_checks_test_impl::make_mesh(
      params.nchunks, params.stacey_region, /* nstacey = */ 1);
  if (params.expect_throw) {
    EXPECT_THROW(mesh.check_consistency(), std::runtime_error);
  } else {
    EXPECT_NO_THROW(mesh.check_consistency());
  }
}

INSTANTIATE_TEST_SUITE_P(
    GlobeStaceyRules, GlobeStaceyConsistencyTest,
    ::testing::Values(
        // The mesher refuses absorbing conditions on the full Earth.
        globe_checks_test_impl::StaceyCase{
            6, specfem::element::region_tag::crust_mantle, true,
            "FullEarthRejectsStacey" },
        // ... and does not support them for three chunks.
        globe_checks_test_impl::StaceyCase{
            3, specfem::element::region_tag::crust_mantle, true,
            "ThreeChunksRejectStacey" },
        globe_checks_test_impl::StaceyCase{
            1, specfem::element::region_tag::crust_mantle, false,
            "SingleChunkAllowsCrustMantleStacey" },
        globe_checks_test_impl::StaceyCase{
            2, specfem::element::region_tag::outer_core, false,
            "TwoChunksAllowOuterCoreStacey" },
        // Stacey faces are placed on the crust/mantle and outer core only.
        globe_checks_test_impl::StaceyCase{
            1, specfem::element::region_tag::inner_core, true,
            "InnerCoreRejectsStacey" }),
    [](const ::testing::TestParamInfo<globe_checks_test_impl::StaceyCase>
           &info) { return info.param.name; });

TEST(GlobeStaceyConsistency, FullEarthWithoutStaceyIsConsistent) {
  const auto mesh = globe_checks_test_impl::make_mesh(
      6, specfem::element::region_tag::crust_mantle, /* nstacey = */ 0);
  EXPECT_NO_THROW(mesh.check_consistency());
}
