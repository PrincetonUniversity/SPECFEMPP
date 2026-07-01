#include "SPECFEM_Environment.hpp"
#include "specfem/algorithms/locate_point.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/resolve_coordinates.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/utm.hpp"
#include "specfem/mesh_entity.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <optional>

namespace surface_elevation_test {

constexpr auto dimension = specfem::element::dimension_tag::dim3;

// Single top (Z_MAX) face for the single-element test meshes below.
specfem::mesh::acoustic_free_surface<dimension> make_top_surface() {
  specfem::mesh::acoustic_free_surface<dimension> surface(1);
  surface.index_mapping(0) = 0;
  surface.type(0) = specfem::mesh_entity::dim3::type::top;
  return surface;
}

// Build a single-element (ngll=3) mesh whose top face is a plane gently sloped
// in x: z = 100 + 0.1*x, with node spacing x = 10*ix, y = 10*iy. The
// surface elevation at horizontal (x, y) is therefore 100 + 0.1*x.
//
// Control nodes are needed because the elevation is interpolated with the
// spectral element machinery (compute_locations/compute_jacobian). With
// ngnod = 8 the control nodes are the element corners.
specfem::assembly::mesh<dimension> make_sloped_mesh() {
  specfem::assembly::mesh<dimension> mesh;
  mesh.ngnod = 8;
  mesh.element_grid = specfem::mesh_entity::element<dimension>(3, 3, 3);

  mesh.h_mesh_to_compute = decltype(mesh.h_mesh_to_compute)("m2c", 1);
  mesh.h_mesh_to_compute(0) = 0;

  // GLL-point coordinates (used by the nearest-node search).
  mesh.h_coord = decltype(mesh.h_coord)("coord", 1, 3, 3, 3, 3);
  for (int iz = 0; iz < 3; ++iz)
    for (int iy = 0; iy < 3; ++iy)
      for (int ix = 0; ix < 3; ++ix) {
        mesh.h_coord(0, iz, iy, ix, 0) = ix * 10.0;
        mesh.h_coord(0, iz, iy, ix, 1) = iy * 10.0;
        mesh.h_coord(0, iz, iy, ix, 2) = 100.0 + 0.1 * (ix * 10.0);
      }

  // Control nodes (8 corners) defining the element geometry: x,y in [0,20],
  // z = 100 + 0.1*x (so the top face is the same gently sloped plane).
  mesh.h_control_node_coordinates =
      decltype(mesh.h_control_node_coordinates)("cn", 3, 1, 8);
  const double cx[8] = { 0, 20, 20, 0, 0, 20, 20, 0 };
  const double cy[8] = { 0, 0, 20, 20, 0, 0, 20, 20 };
  const double cz_bottom = 0.0;
  for (int i = 0; i < 8; ++i) {
    const bool top = (i >= 4);
    mesh.h_control_node_coordinates(0, 0, i) = cx[i];
    mesh.h_control_node_coordinates(1, 0, i) = cy[i];
    mesh.h_control_node_coordinates(2, 0, i) =
        top ? (100.0 + 0.1 * cx[i]) : cz_bottom;
  }
  return mesh;
}

// Build a single-element (ngll=3) mesh with a flat top face at z = surface_z,
// centered horizontally on (cx, cy) and spanning +/- half in x and y. Used to
// resolve geographic coordinates whose UTM projection lands at (cx, cy).
specfem::assembly::mesh<dimension>
make_flat_surface_mesh(double cx, double cy, double surface_z, double half) {
  specfem::assembly::mesh<dimension> mesh;
  mesh.ngnod = 8;
  mesh.element_grid = specfem::mesh_entity::element<dimension>(3, 3, 3);

  mesh.h_mesh_to_compute = decltype(mesh.h_mesh_to_compute)("m2c", 1);
  mesh.h_mesh_to_compute(0) = 0;

  mesh.h_coord = decltype(mesh.h_coord)("coord", 1, 3, 3, 3, 3);
  for (int iz = 0; iz < 3; ++iz)
    for (int iy = 0; iy < 3; ++iy)
      for (int ix = 0; ix < 3; ++ix) {
        mesh.h_coord(0, iz, iy, ix, 0) = cx - half + ix * half;
        mesh.h_coord(0, iz, iy, ix, 1) = cy - half + iy * half;
        mesh.h_coord(0, iz, iy, ix, 2) = surface_z;
      }

  mesh.h_control_node_coordinates =
      decltype(mesh.h_control_node_coordinates)("cn", 3, 1, 8);
  const double dx[8] = { -1, 1, 1, -1, -1, 1, 1, -1 };
  const double dy[8] = { -1, -1, 1, 1, -1, -1, 1, 1 };
  for (int i = 0; i < 8; ++i) {
    const bool top = (i >= 4);
    mesh.h_control_node_coordinates(0, 0, i) = cx + dx[i] * half;
    mesh.h_control_node_coordinates(1, 0, i) = cy + dy[i] * half;
    mesh.h_control_node_coordinates(2, 0, i) =
        top ? surface_z : surface_z - 2000.0;
  }
  return mesh;
}

} // namespace surface_elevation_test

TEST(SurfaceElevation, InterpolatesBetweenNodes) {
  const auto mesh = surface_elevation_test::make_sloped_mesh();
  const auto surface = surface_elevation_test::make_top_surface();
  const type_real elevation = specfem::algorithms::project_onto_surface(
                                  mesh, surface, { 5.0, 10.0, 0.0 })
                                  .z;
  // Surface plane value at x=5 is 100.5.
  EXPECT_NEAR(elevation, 100.5, 0.1);
}

TEST(SurfaceElevation, ExactOnNode) {
  const auto mesh = surface_elevation_test::make_sloped_mesh();
  const auto surface = surface_elevation_test::make_top_surface();
  // (x, y) = (10, 10) lies on the surface (z = 101).
  const type_real elevation = specfem::algorithms::project_onto_surface(
                                  mesh, surface, { 10.0, 10.0, 0.0 })
                                  .z;
  EXPECT_NEAR(elevation, 101.0, 1e-3);
}

TEST(SurfaceElevation, NoFreeSurfaceReturnsFlat) {
  const specfem::assembly::mesh<surface_elevation_test::dimension> mesh{};
  const specfem::mesh::acoustic_free_surface<surface_elevation_test::dimension>
      surface{}; // no free-surface faces
  const type_real elevation = specfem::algorithms::project_onto_surface(
                                  mesh, surface, { 1.0, 2.0, 0.0 })
                                  .z;
  EXPECT_FLOAT_EQ(elevation, 0.0);
}

TEST(SurfaceElevation, DepthResolvedAgainstTopography) {
  const auto mesh = surface_elevation_test::make_sloped_mesh();
  const auto surface = surface_elevation_test::make_top_surface();
  // Depth-based cartesian: z = -depth, origin unset -> resolve against topo.
  specfem::coordinate_systems::cartesian_coordinates<
      surface_elevation_test::dimension>
      coords(5.0, 10.0, -1000.0, std::nullopt);

  const auto gc = specfem::assembly::resolve_coordinates(coords, mesh, surface);

  EXPECT_FLOAT_EQ(gc.x, 5.0);
  EXPECT_FLOAT_EQ(gc.y, 10.0);
  // z = elevation(5,10) - depth = ~100.5 - 1000.
  EXPECT_NEAR(gc.z, 100.5 - 1000.0, 0.1);
}

// Geographic coordinates with no free surface: UTM-project the lon/lat and
// apply the flat (z = 0) fallback, so the resolved z is just -depth.
TEST(SurfaceElevation, GeographicResolvesViaUtmFlatFallback) {
  const double lon = 2.674, lat = 51.561, depth = 2000.0;
  const specfem::coordinate_systems::utm_projection_config cfg{ 31, false };

  // Expected easting/northing from the same forward projection.
  const auto cart = specfem::coordinate_systems::transform<
      specfem::coordinate_systems::cartesian_coordinates<
          surface_elevation_test::dimension>>(
      specfem::coordinate_systems::geographic_coordinates{ lon, lat, depth },
      cfg);

  const specfem::assembly::mesh<surface_elevation_test::dimension>
      mesh{}; // no free surface -> flat
  const specfem::mesh::acoustic_free_surface<surface_elevation_test::dimension>
      surface{};
  specfem::coordinate_systems::geographic_coordinates geo(lon, lat, depth);
  const auto gc =
      specfem::assembly::resolve_coordinates(geo, mesh, surface, cfg);

  EXPECT_NEAR(gc.x, static_cast<type_real>(cart.x), 1.0);
  EXPECT_NEAR(gc.y, static_cast<type_real>(cart.y), 1.0);
  EXPECT_NEAR(gc.z, static_cast<type_real>(-depth), 1.0);
}

// Geographic coordinates resolved against a flat topographic surface placed at
// the projected easting/northing: z = surface_elevation - depth.
TEST(SurfaceElevation, GeographicResolvesAgainstTopography) {
  const double lon = 2.674, lat = 51.561, depth = 2000.0;
  const double surface_z = 200.0;
  const specfem::coordinate_systems::utm_projection_config cfg{ 31, false };

  const auto cart = specfem::coordinate_systems::transform<
      specfem::coordinate_systems::cartesian_coordinates<
          surface_elevation_test::dimension>>(
      specfem::coordinate_systems::geographic_coordinates{ lon, lat, depth },
      cfg);

  const auto mesh = surface_elevation_test::make_flat_surface_mesh(
      cart.x, cart.y, surface_z, 5000.0);
  const auto surface = surface_elevation_test::make_top_surface();
  specfem::coordinate_systems::geographic_coordinates geo(lon, lat, depth);
  const auto gc =
      specfem::assembly::resolve_coordinates(geo, mesh, surface, cfg);

  EXPECT_NEAR(gc.x, static_cast<type_real>(cart.x), 1.0);
  EXPECT_NEAR(gc.y, static_cast<type_real>(cart.y), 1.0);
  EXPECT_NEAR(gc.z, static_cast<type_real>(surface_z - depth), 1.0);
}

// Geographic coordinates without a UTM config cannot be projected.
TEST(SurfaceElevation, GeographicWithoutUtmConfigThrows) {
  const specfem::assembly::mesh<surface_elevation_test::dimension> mesh{};
  const specfem::mesh::acoustic_free_surface<surface_elevation_test::dimension>
      surface{};
  specfem::coordinate_systems::geographic_coordinates geo(2.674, 51.561,
                                                          2000.0);
  EXPECT_THROW(specfem::assembly::resolve_coordinates(geo, mesh, surface),
               std::runtime_error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
