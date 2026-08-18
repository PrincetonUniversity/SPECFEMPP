#include "specfem/assembly/resolve_coordinates.hpp"

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/utm.hpp"

// resolve_coordinates() resolves depth/geographic coordinates via
// specfem::algorithms::locate_point, which uses specfem::MPI. That requires an
// initialized MPI Context, so this test installs a SPECFEMEnvironment (rather
// than relying on gtest_main). Without it the 3D depth/geographic cases abort
// with "MPI used outside Context scope" in MPI-enabled builds.
#include "SPECFEM_Environment.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace {

// Default-constructed mesh — resolve_coordinates doesn't use it for
// the currently implemented coordinate types.
const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> mesh2d{};
const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> mesh3d{};

// Empty free surface → flat (z = 0) topography fallback.
const specfem::mesh::acoustic_free_surface<
    specfem::element::dimension_tag::dim2>
    surface2d{};
const specfem::mesh::acoustic_free_surface<
    specfem::element::dimension_tag::dim3>
    surface3d{};

} // namespace

// ── 2D Tests ────────────────────────────────────────────────────────────────

TEST(ResolveCoordinates2D, CartesianAbsolute) {
  // Origin set to {0,0} → absolute coordinates
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim2>
      coords(100.0, 200.0, std::array<double, 2>{ 0.0, 0.0 });

  const auto result =
      specfem::assembly::resolve_coordinates(coords, mesh2d, surface2d);
  const auto &gc = result.global;

  EXPECT_FLOAT_EQ(gc.x, 100.0);
  EXPECT_FLOAT_EQ(gc.z, 200.0);
  EXPECT_FALSE(result.topography.has_value()); // dim2 has no topography
}

TEST(ResolveCoordinates2D, CartesianDepthFlatFallback) {
  // Origin nullopt → depth-based, flat fallback sets origin to {0,0}
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim2>
      coords(100.0, -50.0, std::nullopt);

  const auto result =
      specfem::assembly::resolve_coordinates(coords, mesh2d, surface2d);
  const auto &gc = result.global;

  EXPECT_FLOAT_EQ(gc.x, 100.0);
  EXPECT_FLOAT_EQ(gc.z, -50.0);
  // Verify origin was set
  ASSERT_TRUE(coords.origin.has_value());
  EXPECT_FALSE(result.topography.has_value()); // dim2 has no topography
}

// ── 3D Tests ────────────────────────────────────────────────────────────────

TEST(ResolveCoordinates3D, CartesianAbsolute) {
  // z given directly → origin = {0,0,0}
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>
      coords(1.0, 2.0, 3.0, std::array<double, 3>{ 0.0, 0.0, 0.0 });

  const auto result =
      specfem::assembly::resolve_coordinates(coords, mesh3d, surface3d);
  const auto &gc = result.global;

  EXPECT_FLOAT_EQ(gc.x, 1.0);
  EXPECT_FLOAT_EQ(gc.y, 2.0);
  EXPECT_FLOAT_EQ(gc.z, 3.0);
  // Absolute coords (origin set) skip the topographic lookup.
  EXPECT_FALSE(result.topography.has_value());
}

TEST(ResolveCoordinates3D, CartesianDepthFlatFallback) {
  // depth = 5000 m → stored as z = -5000, origin nullopt
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>
      coords(10.0, 20.0, -5000.0, std::nullopt);

  const auto result =
      specfem::assembly::resolve_coordinates(coords, mesh3d, surface3d);
  const auto &gc = result.global;

  EXPECT_FLOAT_EQ(gc.x, 10.0);
  EXPECT_FLOAT_EQ(gc.y, 20.0);
  EXPECT_FLOAT_EQ(gc.z, -5000.0);
  // Verify origin was set by flat fallback
  ASSERT_TRUE(coords.origin.has_value());
  EXPECT_DOUBLE_EQ((*coords.origin)[2], 0.0);
  // Depth-based input: topography reported as the flat-fallback elevation.
  ASSERT_TRUE(result.topography.has_value());
  EXPECT_FLOAT_EQ(*result.topography, 0.0);
}

TEST(ResolveCoordinates3D, GeographicThrowsWithoutUTMConfig) {
  specfem::coordinate_systems::geographic_coordinates coords(0.0, 0.0, 0.0);

  EXPECT_THROW(
      specfem::assembly::resolve_coordinates(coords, mesh3d, surface3d),
      std::runtime_error);
}

TEST(ResolveCoordinates3D, GeographicWithUTMConfig) {
  // Reference: lon=2.6741959317615298, lat=51.561449479910003, depth=0
  specfem::coordinate_systems::geographic_coordinates coords(
      2.6741959317615298, 51.561449479910003, 0.0);
  specfem::coordinate_systems::utm_projection_config utm_config{ 31 };

  const auto result = specfem::assembly::resolve_coordinates(
      coords, mesh3d, surface3d, utm_config);
  const auto &gc = result.global;

  // UTM zone 31: easting ~477415.5, northing ~5712313.5
  EXPECT_NEAR(gc.x, 477415.5, 0.01);
  EXPECT_NEAR(gc.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(gc.z, 0.0); // depth = 0 → z = 0
  // Geographic input resolves depth against the (flat) surface.
  ASSERT_TRUE(result.topography.has_value());
  EXPECT_FLOAT_EQ(*result.topography, 0.0);
}

TEST(ResolveCoordinates3D, GeographicWithDepth) {
  // Same reference location, 5 km depth
  specfem::coordinate_systems::geographic_coordinates coords(
      2.6741959317615298, 51.561449479910003, 5000.0);
  specfem::coordinate_systems::utm_projection_config utm_config{ 31 };

  const auto result = specfem::assembly::resolve_coordinates(
      coords, mesh3d, surface3d, utm_config);
  const auto &gc = result.global;

  EXPECT_NEAR(gc.x, 477415.5, 0.01);
  EXPECT_NEAR(gc.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(gc.z, -5000.0); // flat fallback: z = 0 - 5000
  ASSERT_TRUE(result.topography.has_value());
  EXPECT_FLOAT_EQ(*result.topography, 0.0);
}

TEST(ResolveCoordinates3D, GeocentricThrows) {
  specfem::coordinate_systems::geocentric_coordinates coords(6371000.0, 0.5,
                                                             1.0);

  EXPECT_THROW(
      specfem::assembly::resolve_coordinates(coords, mesh3d, surface3d),
      std::runtime_error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
