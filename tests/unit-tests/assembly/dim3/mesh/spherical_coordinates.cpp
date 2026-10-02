#include "../test_fixture.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace spherical_coordinates_test_impl {
using Geocentric = specfem::coordinate_systems::geocentric_coordinates;
constexpr auto dimension = specfem::element::dimension_tag::dim3;
using Cache = specfem::assembly::mesh_impl::SphericalCoordinates;
const std::string fixture_path = "data/dim3_globe/GlobalSmallMesh/";
} // namespace spherical_coordinates_test_impl

TEST(GeocentricCoordinates, AxesAndBranchCuts) {
  using spherical_coordinates_test_impl::Geocentric;
  constexpr double pi = std::numbers::pi;
  const std::array<std::array<double, 3>, 11> xyz = { { { 0, 0, 6371000 },
                                                        { 0, 0, -6371000 },
                                                        { -6371000, 0, 0 },
                                                        { -6371000, -0.0, 0 },
                                                        { -6371000, 1e-6, 0 },
                                                        { -6371000, -1e-6, 0 },
                                                        { 6371000, 0, 0 },
                                                        { 6371000, -1e-20, 0 },
                                                        { 0, 6371000, 0 },
                                                        { 0, -6371000, 0 },
                                                        { 0, 0, 0 } } };
  for (const auto &point : xyz) {
    const auto spherical =
        Geocentric::from_cartesian(point[0], point[1], point[2]);
    EXPECT_TRUE(std::isfinite(spherical.r));
    EXPECT_GE(spherical.theta, 0);
    EXPECT_LE(spherical.theta, pi);
    EXPECT_GE(spherical.phi, 0);
    EXPECT_LT(spherical.phi, 2 * pi);
    const auto actual = spherical.to_cartesian();
    for (int dim = 0; dim < 3; ++dim) {
      EXPECT_NEAR(actual[dim], point[dim], 1e-9 * std::max(1.0, spherical.r));
    }
  }
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, 0, 1).theta, 0);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, 0, -1).theta, pi);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, 0, -1).phi, 0);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(-1, -0.0, 0).phi, pi);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, -1, 0).phi, 1.5 * pi);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, 0, 0).r, 0);
  EXPECT_DOUBLE_EQ(Geocentric::from_cartesian(0, 0, 0).theta, 0);
  EXPECT_THROW(
      Geocentric::from_cartesian(std::numeric_limits<double>::infinity(), 0, 0),
      std::invalid_argument);
  EXPECT_THROW(Geocentric::from_cartesian(
                   0, std::numeric_limits<double>::quiet_NaN(), 0),
               std::invalid_argument);
}

TEST_P(Assembly3DTest, CartesianSphericalCacheIsEmpty) {
  const auto &cache = getAssembly().mesh.spherical_coordinates;
  EXPECT_EQ(cache.h_coord.size(), 0);
  EXPECT_EQ(cache.h_coord.data(), nullptr);
}

TEST(SphericalCoordinates, GlobeRoundTripAndRelease) {
  namespace test_impl = spherical_coordinates_test_impl;
  const auto raw_mesh = specfem::io::read_globe_mesh(
      test_impl::fixture_path +
          "DATABASES_MPI/proc000000_specfempp_database.bin",
      specfem::attenuation::Setup{});
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  specfem::assembly::mesh<test_impl::dimension> mesh{
    raw_mesh.nspec,
    raw_mesh.control_nodes.ngnod,
    raw_mesh.element_grid.ngllz,
    raw_mesh.element_grid.nglly,
    raw_mesh.element_grid.ngllx,
    raw_mesh.tags,
    raw_mesh.adjacency_graph,
    raw_mesh.control_nodes,
    quadrature,
    raw_mesh.globe.reference_coordinates
  };
  mesh.spherical_coordinates = test_impl::Cache(mesh);
  auto &cache = mesh.spherical_coordinates;
  ASSERT_EQ(cache.h_coord.size(), mesh.h_coord.size());
  ASSERT_NE(cache.h_coord.data(), nullptr);
  double max_relative_error = 0;
  double max_reference_difference = 0;
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    for (int iz = 0; iz < mesh.element_grid.ngllz; ++iz) {
      for (int iy = 0; iy < mesh.element_grid.nglly; ++iy) {
        for (int ix = 0; ix < mesh.element_grid.ngllx; ++ix) {
          const test_impl::Geocentric spherical{
            cache.h_coord(ispec, iz, iy, ix, 0),
            cache.h_coord(ispec, iz, iy, ix, 1),
            cache.h_coord(ispec, iz, iy, ix, 2)
          };
          ASSERT_TRUE(std::isfinite(spherical.r));
          ASSERT_TRUE(std::isfinite(spherical.theta));
          ASSERT_TRUE(std::isfinite(spherical.phi));
          const auto xyz = spherical.to_cartesian();
          for (int dim = 0; dim < 3; ++dim) {
            max_relative_error = std::max(
                max_relative_error,
                std::abs(xyz[dim] - mesh.h_coord(ispec, iz, iy, ix, dim)) /
                    std::max(1.0, spherical.r));
          }
          const double reference_radius =
              std::hypot(mesh.h_reference_coord(ispec, iz, iy, ix, 0),
                         mesh.h_reference_coord(ispec, iz, iy, ix, 1),
                         mesh.h_reference_coord(ispec, iz, iy, ix, 2));
          max_reference_difference =
              std::max(max_reference_difference,
                       std::abs(reference_radius - spherical.r));
        }
      }
    }
  }
  EXPECT_LT(max_relative_error, 1e-9);
  EXPECT_GT(max_reference_difference, 500.0);

  // Dumped by globe's xyz_2_rthetaphi_dble + reduce for corner GLL points.
  // Radius in the dump is nondimensional, as in globe's rstore.
  std::ifstream reference(test_impl::fixture_path +
                          "spherical_coordinates.txt");
  ASSERT_TRUE(reference.is_open());
  int raw_ispec = 0;
  double radius = 0, theta = 0, phi = 0;
  int count = 0;
  const double r_planet = raw_mesh.globe.planet_constants->r_planet();
  while (reference >> raw_ispec >> radius >> theta >> phi) {
    const int ispec = mesh.h_mesh_to_compute(raw_ispec);
    EXPECT_NEAR(cache.h_coord(ispec, 0, 0, 0, 0) / r_planet, radius, 1e-6);
    EXPECT_NEAR(cache.h_coord(ispec, 0, 0, 0, 1), theta, 1e-6);
    EXPECT_NEAR(cache.h_coord(ispec, 0, 0, 0, 2), phi, 1e-6);
    ++count;
  }
  EXPECT_EQ(count, 6);
  EXPECT_EQ(cache.h_coord.use_count(), 1);
  cache.release();
  EXPECT_EQ(cache.h_coord.data(), nullptr);
  EXPECT_EQ(cache.h_coord.size(), 0);
  for (int dim = 0; dim < 5; ++dim)
    EXPECT_EQ(cache.h_coord.extent(dim), 0);
  cache.release();
  EXPECT_EQ(cache.h_coord.data(), nullptr);
}

namespace spherical_coordinates_test_impl {

// Observe the cache at the deferred-property boundary without retaining a view.
class CacheCheckingReader : public specfem::io::reader {
public:
  const specfem::mesh::cartesian3d_mesh &raw_mesh;
  bool expect_cache;
  bool called = false;

  CacheCheckingReader(const specfem::mesh::cartesian3d_mesh &raw_mesh,
                      bool expect_cache)
      : raw_mesh(raw_mesh), expect_cache(expect_cache) {}

  void read(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                &) override {
    FAIL() << "Unexpected 2D assembly";
  }

  void read(specfem::assembly::assembly<dimension> &assembly) override {
    called = true;
    const auto &cache = assembly.mesh.spherical_coordinates;
    EXPECT_EQ(cache.h_coord.size(),
              expect_cache ? assembly.mesh.h_coord.size() : 0);
    if (expect_cache) {
      EXPECT_EQ(cache.h_coord.use_count(), 1);
    }
    assembly.properties = { assembly.element_types, assembly.mesh,
                            raw_mesh.materials, false };
  }
};

} // namespace spherical_coordinates_test_impl

TEST(SphericalCoordinates, AssemblyReleasesAfterPropertySetup) {
  namespace test_impl = spherical_coordinates_test_impl;
  const auto raw_mesh = specfem::io::read_3d_mesh(
      "data/dim3/EightNodeElastic/database.bin", specfem::attenuation::Setup{});
  // A small synthetic globe payload isolates the constructor's cache lifetime
  // from the Fortran model evaluator. Real globe geometry is covered above.
  specfem::mesh::globe3d_mesh globe_mesh;
  static_cast<specfem::mesh::mesh_dim3_base &>(globe_mesh) = raw_mesh;
  globe_mesh.globe.element_context.resize(raw_mesh.nspec);
  const specfem::quadrature::quadratures quadrature(
      specfem::quadrature::gll::gll{});
  std::vector<std::shared_ptr<specfem::sources::source<test_impl::dimension>>>
      sources;
  const auto construct = [&](const auto &mesh, bool expect_cache) {
    auto reader = std::make_shared<test_impl::CacheCheckingReader>(
        raw_mesh, expect_cache);
    const specfem::assembly::assembly<test_impl::dimension> assembly{
      mesh,
      quadrature,
      sources,
      {},
      {},
      0,
      0.01,
      1,
      1,
      1,
      specfem::simulation::type::forward,
      false,
      reader
    };
    EXPECT_TRUE(reader->called);
    EXPECT_EQ(assembly.mesh.spherical_coordinates.h_coord.size(), 0);
    EXPECT_EQ(assembly.mesh.spherical_coordinates.h_coord.data(), nullptr);
  };
  construct(globe_mesh, true);
  construct(raw_mesh, false);
}
