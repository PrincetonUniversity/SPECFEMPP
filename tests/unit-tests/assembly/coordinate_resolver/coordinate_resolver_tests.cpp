#include "specfem/assembly/coordinate_resolver.hpp"

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"

// The resolver locates depth/geographic coordinates via
// specfem::algorithms::project_onto_surface, which uses specfem::MPI. That
// requires an initialized MPI Context, so this test installs a
// SPECFEMEnvironment (rather than relying on gtest_main).
#include "SPECFEM_Environment.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace {

// Default-constructed meshes/surfaces: the resolver does not use them for the
// coordinate types exercised here (absolute cartesian, globe spherical).
const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> mesh2d{};
const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> mesh3d{};
const specfem::mesh::acoustic_free_surface<
    specfem::element::dimension_tag::dim2>
    surface2d{};
const specfem::mesh::acoustic_free_surface<
    specfem::element::dimension_tag::dim3>
    surface3d{};

constexpr double earth_radius = 6371000.0; // meters

double radius_of(const specfem::point::global_coordinates<
                 specfem::element::dimension_tag::dim3> &point) {
  return std::sqrt(static_cast<double>(point.x) * point.x +
                   static_cast<double>(point.y) * point.y +
                   static_cast<double>(point.z) * point.z);
}

} // namespace

// dim2 cartesian (base resolver)

TEST(CoordinateResolver2D, CartesianAbsolute) {
  const specfem::assembly::coordinate_resolver<
      specfem::element::dimension_tag::dim2>
      resolver;
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim2>
      coords(100.0, 200.0, std::array<double, 2>{ 0.0, 0.0 });

  const auto result = resolver.resolve(coords, mesh2d, surface2d);
  EXPECT_FLOAT_EQ(result.global.x, 100.0);
  EXPECT_FLOAT_EQ(result.global.z, 200.0);
  EXPECT_FALSE(result.topography.has_value());
}

TEST(CoordinateResolver2D, CartesianDepthFlatFallback) {
  const specfem::assembly::coordinate_resolver<
      specfem::element::dimension_tag::dim2>
      resolver;
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim2>
      coords(100.0, -50.0, std::nullopt);

  const auto result = resolver.resolve(coords, mesh2d, surface2d);
  EXPECT_FLOAT_EQ(result.global.x, 100.0);
  EXPECT_FLOAT_EQ(result.global.z, -50.0);
  ASSERT_TRUE(coords.origin.has_value());
}

// dim3 cartesian (base resolver)

TEST(CoordinateResolver3D, CartesianAbsolute) {
  const specfem::assembly::coordinate_resolver<
      specfem::element::dimension_tag::dim3>
      resolver;
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>
      coords(1.0, 2.0, 3.0, std::array<double, 3>{ 0.0, 0.0, 0.0 });

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_FLOAT_EQ(result.global.x, 1.0);
  EXPECT_FLOAT_EQ(result.global.y, 2.0);
  EXPECT_FLOAT_EQ(result.global.z, 3.0);
  EXPECT_FALSE(result.topography.has_value());
}

TEST(CoordinateResolver3D, CartesianDepthFlatFallback) {
  const specfem::assembly::coordinate_resolver<
      specfem::element::dimension_tag::dim3>
      resolver;
  specfem::coordinate_systems::cartesian_coordinates<
      specfem::element::dimension_tag::dim3>
      coords(10.0, 20.0, -5000.0, std::nullopt);

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_FLOAT_EQ(result.global.x, 10.0);
  EXPECT_FLOAT_EQ(result.global.y, 20.0);
  EXPECT_FLOAT_EQ(result.global.z, -5000.0);
  ASSERT_TRUE(coords.origin.has_value());
  EXPECT_DOUBLE_EQ((*coords.origin)[2], 0.0);
  ASSERT_TRUE(result.topography.has_value());
  EXPECT_FLOAT_EQ(*result.topography, 0.0);
}

TEST(CoordinateResolver3D, GeographicThrowsWithoutProjection) {
  const specfem::assembly::coordinate_resolver<
      specfem::element::dimension_tag::dim3>
      resolver;
  specfem::coordinate_systems::geographic_coordinates coords(0.0, 0.0, 0.0);

  EXPECT_THROW(resolver.resolve(coords, mesh3d, surface3d), std::runtime_error);
}

// Regional (UTM) resolver

TEST(CoordinateResolverUtm, GeographicZone31) {
  const specfem::assembly::utm_resolver resolver(
      specfem::coordinate_systems::utm_projection_config{ 31 });
  specfem::coordinate_systems::geographic_coordinates coords(
      2.6741959317615298, 51.561449479910003, 0.0);

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_NEAR(result.global.x, 477415.5, 0.01);
  EXPECT_NEAR(result.global.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(result.global.z, 0.0);
  ASSERT_TRUE(result.topography.has_value());
  EXPECT_FLOAT_EQ(*result.topography, 0.0);
}

TEST(CoordinateResolverUtm, GeographicWithDepth) {
  const specfem::assembly::utm_resolver resolver(
      specfem::coordinate_systems::utm_projection_config{ 31 });
  specfem::coordinate_systems::geographic_coordinates coords(
      2.6741959317615298, 51.561449479910003, 5000.0);

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_NEAR(result.global.x, 477415.5, 0.01);
  EXPECT_NEAR(result.global.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(result.global.z, -5000.0);
}

// Globe (spherical) resolver

TEST(CoordinateResolverSpherical, GeocentricResolvesOnSphere) {
  const specfem::assembly::spherical_resolver resolver(earth_radius, false,
                                                       false);
  specfem::coordinate_systems::geocentric_coordinates coords(earth_radius, 0.5,
                                                             1.0);

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_NEAR(radius_of(result.global), earth_radius, 1.0);
  EXPECT_FALSE(result.topography.has_value());
}

TEST(CoordinateResolverSpherical, GeographicPlacement) {
  const specfem::assembly::spherical_resolver resolver(earth_radius, false,
                                                       false);
  // lat=0, lon=0, depth=10 km -> radius r_planet - depth, on the +x axis.
  specfem::coordinate_systems::geographic_coordinates coords(0.0, 0.0, 10000.0);

  const auto result = resolver.resolve(coords, mesh3d, surface3d);
  EXPECT_NEAR(radius_of(result.global), earth_radius - 10000.0, 1.0);
  EXPECT_NEAR(result.global.x, earth_radius - 10000.0, 1.0);
  EXPECT_NEAR(result.global.y, 0.0, 1.0);
  EXPECT_NEAR(result.global.z, 0.0, 1.0);
}

TEST(CoordinateResolverSpherical, EllipticityThrowsNaming2058) {
  const specfem::assembly::spherical_resolver resolver(earth_radius, true,
                                                       false);
  specfem::coordinate_systems::geographic_coordinates coords(0.0, 0.0, 0.0);

  try {
    resolver.resolve(coords, mesh3d, surface3d);
    FAIL() << "expected elliptical globe resolution to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("2058"), std::string::npos)
        << "throw message should name issue #2058: " << error.what();
  }
}

TEST(CoordinateResolverSpherical, TopographyThrowsNaming2058) {
  const specfem::assembly::spherical_resolver resolver(earth_radius, false,
                                                       true);
  specfem::coordinate_systems::geographic_coordinates coords(0.0, 0.0, 0.0);

  try {
    resolver.resolve(coords, mesh3d, surface3d);
    FAIL() << "expected topographic globe resolution to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("2058"), std::string::npos)
        << "throw message should name issue #2058: " << error.what();
  }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
