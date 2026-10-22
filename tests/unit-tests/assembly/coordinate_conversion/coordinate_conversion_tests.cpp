#include "specfem/assembly/coordinate_conversion.hpp"

#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/utm_projection.hpp"

// to<> resolves depth/geographic coordinates via
// specfem::algorithms::project_onto_surface, which uses specfem::MPI. That
// requires an initialized MPI Context, so this test installs a
// SPECFEMEnvironment (rather than relying on gtest_main).
#include "SPECFEM_Environment.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

namespace coordinate_conversion_tests_impl {

namespace cs = specfem::coordinate_systems;
namespace mesh_impl = specfem::assembly::mesh_impl;
constexpr auto dim2 = specfem::element::dimension_tag::dim2;
constexpr auto dim3 = specfem::element::dimension_tag::dim3;

// Default-constructed dim2 assembly mesh: dim2 placement is pass-through and
// does not read the mesh.
const specfem::assembly::mesh<dim2> mesh2d{};

constexpr double earth_radius = 6371000.0; // meters

// A plain Cartesian 3-D assembly mesh with no angular projection. The default
// (empty) free surface means depth-based coordinates use the flat fallback.
specfem::assembly::mesh<dim3> make_cartesian3d_no_projection() {
  specfem::assembly::mesh<dim3> mesh{};
  mesh.projection.value = mesh_impl::cartesian_projection{};
  return mesh;
}

// A regional Cartesian 3-D assembly mesh that projects geographic input via
// the UTM projection.
specfem::assembly::mesh<dim3> make_cartesian3d_utm(const int zone) {
  specfem::assembly::mesh<dim3> mesh{};
  mesh.projection.value =
      specfem::coordinate_systems::utm_projection_config{ zone, false };
  return mesh;
}

// A perfect-sphere globe assembly mesh with Earth's radius.
specfem::assembly::mesh<dim3> make_globe(const bool ellipticity = false,
                                         const bool topography = false) {
  specfem::assembly::mesh<dim3> mesh{};
  mesh.projection.value =
      mesh_impl::globe_projection{ earth_radius, ellipticity, topography };
  return mesh;
}

double radius_of(const specfem::point::global_coordinates<dim3> &point) {
  return std::sqrt(static_cast<double>(point.x) * point.x +
                   static_cast<double>(point.y) * point.y +
                   static_cast<double>(point.z) * point.z);
}

double dot_product(const std::array<double, 3> &first,
                   const std::array<double, 3> &second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

std::array<double, 3> cross_product(const std::array<double, 3> &first,
                                    const std::array<double, 3> &second) {
  return { first[1] * second[2] - first[2] * second[1],
           first[2] * second[0] - first[0] * second[2],
           first[0] * second[1] - first[1] * second[0] };
}

void expect_basis_near(const cs::GeographicBasis &basis,
                       const std::array<double, 3> &east,
                       const std::array<double, 3> &north,
                       const std::array<double, 3> &up,
                       const double tolerance) {
  for (int component = 0; component < 3; ++component) {
    EXPECT_NEAR(basis.east[component], east[component], tolerance);
    EXPECT_NEAR(basis.north[component], north[component], tolerance);
    EXPECT_NEAR(basis.up[component], up[component], tolerance);
  }
}

// ---- Cartesian target, dim2 (plain Cartesian mesh) ------------------------

TEST(To2D, CartesianAbsolute) {
  const cs::cartesian_coordinates<dim2> coords(
      100.0, 200.0, std::array<double, 2>{ 0.0, 0.0 });
  const auto global = specfem::assembly::to_cartesian(coords, mesh2d);
  EXPECT_FLOAT_EQ(global.x, 100.0);
  EXPECT_FLOAT_EQ(global.z, 200.0);
}

TEST(To2D, CartesianDepthFlatFallback) {
  const cs::cartesian_coordinates<dim2> coords(100.0, -50.0, std::nullopt);
  const auto global = specfem::assembly::to_cartesian(coords, mesh2d);
  EXPECT_FLOAT_EQ(global.x, 100.0);
  EXPECT_FLOAT_EQ(global.z, -50.0);
}

// ---- Cartesian target, dim3 (plain Cartesian mesh) ------------------------

TEST(To3D, CartesianAbsolute) {
  const auto mesh = make_cartesian3d_no_projection();
  const cs::cartesian_coordinates<dim3> coords(
      1.0, 2.0, 3.0, std::array<double, 3>{ 0.0, 0.0, 0.0 });
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_FLOAT_EQ(global.x, 1.0);
  EXPECT_FLOAT_EQ(global.y, 2.0);
  EXPECT_FLOAT_EQ(global.z, 3.0);
}

TEST(To3D, CartesianDepthFlatFallback) {
  const auto mesh = make_cartesian3d_no_projection();
  const cs::cartesian_coordinates<dim3> coords(10.0, 20.0, -5000.0,
                                               std::nullopt);
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_FLOAT_EQ(global.x, 10.0);
  EXPECT_FLOAT_EQ(global.y, 20.0);
  EXPECT_FLOAT_EQ(global.z, -5000.0);
}

TEST(To3D, GeographicThrowsWithoutProjection) {
  const auto mesh = make_cartesian3d_no_projection();
  const cs::geographic_coordinates coords(0.0, 0.0, 0.0);
  EXPECT_THROW((specfem::assembly::to_cartesian(coords, mesh)),
               std::runtime_error);
}

// ---- Cartesian target, regional UTM mesh ----------------------------------

TEST(ToUtm, GeographicZone31) {
  const auto mesh = make_cartesian3d_utm(31);
  const cs::geographic_coordinates coords(2.6741959317615298,
                                          51.561449479910003, 0.0);
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_NEAR(global.x, 477415.5, 0.01);
  EXPECT_NEAR(global.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(global.z, 0.0);
}

TEST(ToUtm, GeographicWithDepth) {
  const auto mesh = make_cartesian3d_utm(31);
  const cs::geographic_coordinates coords(2.6741959317615298,
                                          51.561449479910003, 5000.0);
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_NEAR(global.x, 477415.5, 0.01);
  EXPECT_NEAR(global.y, 5712313.5, 0.01);
  EXPECT_FLOAT_EQ(global.z, -5000.0);
}

// ---- Cartesian target, globe mesh -----------------------------------------

TEST(ToSpherical, GeocentricResolvesOnSphere) {
  const auto mesh = make_globe();
  const cs::geocentric_coordinates coords(earth_radius, 0.5, 1.0);
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_NEAR(radius_of(global), earth_radius, 1.0);
}

TEST(ToSpherical, GeographicPlacement) {
  const auto mesh = make_globe();
  // lat=0, lon=0, depth=10 km -> radius r_planet - depth, on the +x axis.
  const cs::geographic_coordinates coords(0.0, 0.0, 10000.0);
  const auto global = specfem::assembly::to_cartesian(coords, mesh);
  EXPECT_NEAR(radius_of(global), earth_radius - 10000.0, 1.0);
  EXPECT_NEAR(global.x, earth_radius - 10000.0, 1.0);
  EXPECT_NEAR(global.y, 0.0, 1.0);
  EXPECT_NEAR(global.z, 0.0, 1.0);
}

TEST(ToSpherical, EllipticityThrowsNaming2058) {
  const auto mesh = make_globe(/*ellipticity=*/true);
  const cs::geographic_coordinates coords(0.0, 0.0, 0.0);
  try {
    specfem::assembly::to_cartesian(coords, mesh);
    FAIL() << "expected elliptical globe resolution to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("2058"), std::string::npos)
        << "throw message should name issue #2058: " << error.what();
  }
}

TEST(ToSpherical, TopographyThrowsNaming2058) {
  const auto mesh = make_globe(/*ellipticity=*/false, /*topography=*/true);
  const cs::geographic_coordinates coords(0.0, 0.0, 0.0);
  try {
    specfem::assembly::to_cartesian(coords, mesh);
    FAIL() << "expected topographic globe resolution to throw";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find("2058"), std::string::npos)
        << "throw message should name issue #2058: " << error.what();
  }
}

// ---- Cartesian -> geocentric / geographic targets -------------------------

TEST(ToGeocentric, FromCartesianOnSphere) {
  const auto mesh = make_globe();
  // A point on the +x axis at r = earth_radius: theta = pi/2, phi = 0.
  const cs::cartesian_coordinates<dim3> cartesian(
      earth_radius, 0.0, 0.0, std::array<double, 3>{ 0.0, 0.0, 0.0 });
  const auto geocentric = specfem::assembly::to_geocentric(cartesian, mesh);
  EXPECT_NEAR(geocentric.r, earth_radius, 1.0);
  EXPECT_NEAR(geocentric.theta, M_PI / 2.0, 1e-6);
  EXPECT_NEAR(geocentric.phi, 0.0, 1e-6);
}

TEST(ToGeographic, FromCartesianOnSphere) {
  const auto mesh = make_globe();
  // +x axis at r = earth_radius - 10 km -> lon=0, lat=0, depth=10 km.
  const cs::cartesian_coordinates<dim3> cartesian(
      earth_radius - 10000.0, 0.0, 0.0, std::array<double, 3>{ 0.0, 0.0, 0.0 });
  const auto geographic = specfem::assembly::to_geographic(cartesian, mesh);
  // geocentric reduce() nudges points off the exact polar axis by ~1e-7 rad
  // (~5.7e-6 deg), so lon/lat are near (not exactly) zero on the axis.
  EXPECT_NEAR(geographic.longitude, 0.0, 1e-4);
  EXPECT_NEAR(geographic.latitude, 0.0, 1e-4);
  EXPECT_NEAR(geographic.depth, 10000.0, 1.0);
}

TEST(ToGeographic, FromCartesianRoundTripUtm) {
  const auto mesh = make_cartesian3d_utm(31);
  const cs::geographic_coordinates original(2.6741959317615298,
                                            51.561449479910003, 0.0);
  // geographic -> cartesian (mesh space) -> geographic.
  const auto global = specfem::assembly::to_cartesian(original, mesh);
  const cs::cartesian_coordinates<dim3> cartesian(
      global.x, global.y, global.z, std::array<double, 3>{ 0.0, 0.0, 0.0 });
  const auto recovered = specfem::assembly::to_geographic(cartesian, mesh);
  EXPECT_NEAR(recovered.longitude, original.longitude, 1e-4);
  EXPECT_NEAR(recovered.latitude, original.latitude, 1e-4);
}

// ---- Local geographic basis -------------------------------------------------

TEST(LocalGeographicBasis, CartesianMeshIsIdentity) {
  const auto mesh = make_cartesian3d_no_projection();
  const specfem::point::global_coordinates<dim3> point{ 1234.0, -567.0, 89.0 };
  const auto basis = specfem::assembly::local_geographic_basis(point, mesh);
  expect_basis_near(basis, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
                    { 0.0, 0.0, 1.0 }, 0.0);
}

TEST(LocalGeographicBasis, UtmMeshIsIdentity) {
  const auto mesh = make_cartesian3d_utm(31);
  const specfem::point::global_coordinates<dim3> point{ 500000.0, 5700000.0,
                                                        -100.0 };
  const auto basis = specfem::assembly::local_geographic_basis(point, mesh);
  expect_basis_near(basis, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
                    { 0.0, 0.0, 1.0 }, 0.0);
}

TEST(LocalGeographicBasis, GlobeUpIsRadial) {
  const auto mesh = make_globe();
  const std::array<std::array<double, 2>, 5> longitude_latitude = {
    { { 0.0, 0.0 },
      { 10.0, 40.0 },
      { -66.15, 18.1 },
      { 146.25, -20.09 },
      { 9.6, 59.65 } }
  };
  for (const auto &entry : longitude_latitude) {
    const auto point = specfem::assembly::to_cartesian(
        cs::geographic_coordinates(entry[0], entry[1], 0.0), mesh);
    const auto basis = specfem::assembly::local_geographic_basis(point, mesh);
    const double radius = radius_of(point);
    const std::array<double, 3> radial = { point.x / radius, point.y / radius,
                                           point.z / radius };
    EXPECT_NEAR(dot_product(basis.up, radial), 1.0, 1e-6);
    EXPECT_NEAR(dot_product(basis.east, basis.east), 1.0, 1e-6);
    EXPECT_NEAR(dot_product(basis.north, basis.north), 1.0, 1e-6);
    EXPECT_NEAR(dot_product(basis.up, basis.up), 1.0, 1e-6);
    const auto east_cross_north = cross_product(basis.east, basis.north);
    for (int component = 0; component < 3; ++component)
      EXPECT_NEAR(east_cross_north[component], basis.up[component], 1e-6);
  }
}

TEST(LocalGeographicBasis, GlobeEquatorPrimeMeridian) {
  const auto mesh = make_globe();
  const specfem::point::global_coordinates<dim3> point{ earth_radius, 0.0,
                                                        0.0 };
  const auto basis = specfem::assembly::local_geographic_basis(point, mesh);
  expect_basis_near(basis, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 },
                    { 1.0, 0.0, 0.0 }, 1e-6);
}

TEST(LocalGeographicBasis, GlobeNorthPoleIsFinite) {
  const auto mesh = make_globe();
  const specfem::point::global_coordinates<dim3> point{ 0.0, 0.0,
                                                        earth_radius };
  const auto basis = specfem::assembly::local_geographic_basis(point, mesh);
  for (int component = 0; component < 3; ++component) {
    EXPECT_TRUE(std::isfinite(basis.east[component]));
    EXPECT_TRUE(std::isfinite(basis.north[component]));
    EXPECT_TRUE(std::isfinite(basis.up[component]));
  }
  expect_basis_near(basis, { 0.0, 1.0, 0.0 }, { -1.0, 0.0, 0.0 },
                    { 0.0, 0.0, 1.0 }, 1e-6);
}

TEST(LocalGeographicBasis, GlobeEllipticityThrows) {
  const auto mesh = make_globe(/*ellipticity=*/true);
  const specfem::point::global_coordinates<dim3> point{ earth_radius, 0.0,
                                                        0.0 };
  EXPECT_THROW(specfem::assembly::local_geographic_basis(point, mesh),
               std::runtime_error);
}

} // namespace coordinate_conversion_tests_impl

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
