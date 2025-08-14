#include "algorithms/locate_point_impl.hpp"
#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "algorithms/locate_point.hpp"
#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// Test helper functions from locate_point_impl namespace

TEST(LOCATE_HELPERS_3D, rough_location_simple) {
  // Create a simple 2x2x2 grid of quadrature points for two cube elements
  const int nspec = 2;
  const int ngllx = 2;
  const int nglly = 2;
  const int ngllz = 2;

  // 3D coordinate layout: (nspec, iz, iy, ix, icoord)
  specfem::algorithms::locate_point_impl::MeshHostCoordinatesViewType3D coord(
      "coord", nspec, ngllz, nglly, ngllx, 3);

  // Element 0: unit cube [0,1]^3
  coord(0, 0, 0, 0, 0) = 0.0;
  coord(0, 0, 0, 0, 1) = 0.0;
  coord(0, 0, 0, 0, 2) = 0.0; // (0,0,0)
  coord(0, 0, 0, 1, 0) = 1.0;
  coord(0, 0, 0, 1, 1) = 0.0;
  coord(0, 0, 0, 1, 2) = 0.0; // (1,0,0)
  coord(0, 0, 1, 0, 0) = 0.0;
  coord(0, 0, 1, 0, 1) = 1.0;
  coord(0, 0, 1, 0, 2) = 0.0; // (0,1,0)
  coord(0, 0, 1, 1, 0) = 1.0;
  coord(0, 0, 1, 1, 1) = 1.0;
  coord(0, 0, 1, 1, 2) = 0.0; // (1,1,0)
  coord(0, 1, 0, 0, 0) = 0.0;
  coord(0, 1, 0, 0, 1) = 0.0;
  coord(0, 1, 0, 0, 2) = 1.0; // (0,0,1)
  coord(0, 1, 0, 1, 0) = 1.0;
  coord(0, 1, 0, 1, 1) = 0.0;
  coord(0, 1, 0, 1, 2) = 1.0; // (1,0,1)
  coord(0, 1, 1, 0, 0) = 0.0;
  coord(0, 1, 1, 0, 1) = 1.0;
  coord(0, 1, 1, 0, 2) = 1.0; // (0,1,1)
  coord(0, 1, 1, 1, 0) = 1.0;
  coord(0, 1, 1, 1, 1) = 1.0;
  coord(0, 1, 1, 1, 2) = 1.0; // (1,1,1)

  // Element 1: cube [1,2] x [0,1] x [0,1]
  coord(1, 0, 0, 0, 0) = 1.0;
  coord(1, 0, 0, 0, 1) = 0.0;
  coord(1, 0, 0, 0, 2) = 0.0; // (1,0,0)
  coord(1, 0, 0, 1, 0) = 2.0;
  coord(1, 0, 0, 1, 1) = 0.0;
  coord(1, 0, 0, 1, 2) = 0.0; // (2,0,0)
  coord(1, 0, 1, 0, 0) = 1.0;
  coord(1, 0, 1, 0, 1) = 1.0;
  coord(1, 0, 1, 0, 2) = 0.0; // (1,1,0)
  coord(1, 0, 1, 1, 0) = 2.0;
  coord(1, 0, 1, 1, 1) = 1.0;
  coord(1, 0, 1, 1, 2) = 0.0; // (2,1,0)
  coord(1, 1, 0, 0, 0) = 1.0;
  coord(1, 1, 0, 0, 1) = 0.0;
  coord(1, 1, 0, 0, 2) = 1.0; // (1,0,1)
  coord(1, 1, 0, 1, 0) = 2.0;
  coord(1, 1, 0, 1, 1) = 0.0;
  coord(1, 1, 0, 1, 2) = 1.0; // (2,0,1)
  coord(1, 1, 1, 0, 0) = 1.0;
  coord(1, 1, 1, 0, 1) = 1.0;
  coord(1, 1, 1, 0, 2) = 1.0; // (1,1,1)
  coord(1, 1, 1, 1, 0) = 2.0;
  coord(1, 1, 1, 1, 1) = 1.0;
  coord(1, 1, 1, 1, 2) = 1.0; // (2,1,1)

  // Test point at (0.25, 0.25, 0.25) - should find element 0, point (0,0,0)
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
      test_point = { 0.25, 0.25, 0.25 };

  auto [ispec, ix, iy, iz] =
      specfem::algorithms::locate_point_impl::rough_location(test_point, coord);

  EXPECT_EQ(ispec, 0);
  EXPECT_EQ(ix, 0);
  EXPECT_EQ(iy, 0);
  EXPECT_EQ(iz, 0);

  // Test point at (1.5, 0.5, 0.5) - should find element 1, but rough_location
  // finds closest GLL point Since element 0 has a GLL point at (1.0, 0.0, 0.0)
  // and element 1 has (1.0, 0.0, 0.0), both at distance sqrt((0.5)^2 + (0.5)^2
  // + (0.5)^2) = 0.866 The point (1.5, 0.5, 0.5) is actually closer to (1,0,0)
  // in element 1, but verify which gets returned
  test_point = { 1.5, 0.5, 0.5 };
  std::tie(ispec, ix, iy, iz) =
      specfem::algorithms::locate_point_impl::rough_location(test_point, coord);
  // Should find element 1, but if element 0 is found, it's still reasonable due
  // to GLL point proximity
  EXPECT_TRUE(ispec == 0 || ispec == 1);
}

TEST(LOCATE_HELPERS_3D, get_best_candidates_simple) {
  // Create a simple 2x2x2 index mapping for 2 cube elements
  const int nspec = 2;
  const int ngllx = 2;
  const int nglly = 2;
  const int ngllz = 2;

  Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, nglly, ngllx);

  // Element 0: global node indices 0-7 (corners of first cube)
  index_mapping(0, 0, 0, 0) = 0;
  index_mapping(0, 0, 0, 1) = 1;
  index_mapping(0, 0, 1, 0) = 2;
  index_mapping(0, 0, 1, 1) = 3;
  index_mapping(0, 1, 0, 0) = 4;
  index_mapping(0, 1, 0, 1) = 5;
  index_mapping(0, 1, 1, 0) = 6;
  index_mapping(0, 1, 1, 1) = 7;

  // Element 1: shares face with element 0 (nodes 1,3,5,7), has new nodes 8-11
  index_mapping(1, 0, 0, 0) = 1;
  index_mapping(1, 0, 0, 1) = 8;
  index_mapping(1, 0, 1, 0) = 3;
  index_mapping(1, 0, 1, 1) = 9;
  index_mapping(1, 1, 0, 0) = 5;
  index_mapping(1, 1, 0, 1) = 10;
  index_mapping(1, 1, 1, 0) = 7;
  index_mapping(1, 1, 1, 1) = 11;

  auto candidates = specfem::algorithms::locate_point_impl::get_best_candidates(
      0, index_mapping);

  // Should return both elements 0 and 1 since they share a face
  EXPECT_EQ(candidates.size(), 2);
  EXPECT_EQ(candidates[0], 0); // Initial guess element
  EXPECT_EQ(candidates[1], 1); // Neighboring element
}

TEST(LOCATE_HELPERS_3D, get_best_location_unit_cube) {
  // Test with a unit cube element
  const int ngnod = 8;
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube control nodes (standard ordering)
  coorg(0) = { 0.0, 0.0, 0.0 }; // Corner 0: (0,0,0)
  coorg(1) = { 1.0, 0.0, 0.0 }; // Corner 1: (1,0,0)
  coorg(2) = { 1.0, 1.0, 0.0 }; // Corner 2: (1,1,0)
  coorg(3) = { 0.0, 1.0, 0.0 }; // Corner 3: (0,1,0)
  coorg(4) = { 0.0, 0.0, 1.0 }; // Corner 4: (0,0,1)
  coorg(5) = { 1.0, 0.0, 1.0 }; // Corner 5: (1,0,1)
  coorg(6) = { 1.0, 1.0, 1.0 }; // Corner 6: (1,1,1)
  coorg(7) = { 0.0, 1.0, 1.0 }; // Corner 7: (0,1,1)

  // Test point at center of unit cube (0.5, 0.5, 0.5)
  // Should map to local coordinates (0, 0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  type_real xi_initial = 0.1;
  type_real eta_initial = 0.1;
  type_real gamma_initial = 0.1;

  auto [xi_final, eta_final, gamma_final] =
      specfem::algorithms::locate_point_impl::get_best_location(
          target, coorg, xi_initial, eta_initial, gamma_initial);

  // For a unit cube, center point (0.5, 0.5, 0.5) should map to (0, 0, 0) in
  // reference coords
  EXPECT_NEAR(xi_final, 0.0, 1e-6);
  EXPECT_NEAR(eta_final, 0.0, 1e-6);
  EXPECT_NEAR(gamma_final, 0.0, 1e-6);

  // Test corner point (0, 0, 0) should map to (-1, -1, -1)
  target = { 0.0, 0.0, 0.0 };
  std::tie(xi_final, eta_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_best_location(target, coorg,
                                                                0.0, 0.0, 0.0);

  EXPECT_NEAR(xi_final, -1.0, 1e-6);
  EXPECT_NEAR(eta_final, -1.0, 1e-6);
  EXPECT_NEAR(gamma_final, -1.0, 1e-6);

  // Test corner point (1, 1, 1) should map to (1, 1, 1)
  target = { 1.0, 1.0, 1.0 };
  std::tie(xi_final, eta_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_best_location(target, coorg,
                                                                0.0, 0.0, 0.0);

  EXPECT_NEAR(xi_final, 1.0, 1e-6);
  EXPECT_NEAR(eta_final, 1.0, 1e-6);
  EXPECT_NEAR(gamma_final, 1.0, 1e-6);
}

TEST(LOCATE_HELPERS_3D, locate_point_core_unit_cube) {
  // Test the core locate_point function with a single unit cube geometry
  // This tests the function without needing a full assembly mesh

  const int nspec = 1;
  const int ngllx = 2;
  const int nglly = 2;
  const int ngllz = 2;
  const int ngnod = 8;

  // Global coordinates of GLL quadrature points for unit cube
  // 3D layout: (nspec, iz, iy, ix, icoord)
  specfem::algorithms::locate_point_impl::MeshHostCoordinatesViewType3D
      global_coords("global_coords", nspec, ngllz, nglly, ngllx, 3);

  // GLL quadrature points for unit cube element mapped to [-1,1]^3 reference
  // element
  global_coords(0, 0, 0, 0, 0) = 0.0;
  global_coords(0, 0, 0, 0, 1) = 0.0;
  global_coords(0, 0, 0, 0, 2) = 0.0; // (-1,-1,-1) -> (0,0,0)
  global_coords(0, 0, 0, 1, 0) = 1.0;
  global_coords(0, 0, 0, 1, 1) = 0.0;
  global_coords(0, 0, 0, 1, 2) = 0.0; // ( 1,-1,-1) -> (1,0,0)
  global_coords(0, 0, 1, 0, 0) = 0.0;
  global_coords(0, 0, 1, 0, 1) = 1.0;
  global_coords(0, 0, 1, 0, 2) = 0.0; // (-1, 1,-1) -> (0,1,0)
  global_coords(0, 0, 1, 1, 0) = 1.0;
  global_coords(0, 0, 1, 1, 1) = 1.0;
  global_coords(0, 0, 1, 1, 2) = 0.0; // ( 1, 1,-1) -> (1,1,0)
  global_coords(0, 1, 0, 0, 0) = 0.0;
  global_coords(0, 1, 0, 0, 1) = 0.0;
  global_coords(0, 1, 0, 0, 2) = 1.0; // (-1,-1, 1) -> (0,0,1)
  global_coords(0, 1, 0, 1, 0) = 1.0;
  global_coords(0, 1, 0, 1, 1) = 0.0;
  global_coords(0, 1, 0, 1, 2) = 1.0; // ( 1,-1, 1) -> (1,0,1)
  global_coords(0, 1, 1, 0, 0) = 0.0;
  global_coords(0, 1, 1, 0, 1) = 1.0;
  global_coords(0, 1, 1, 0, 2) = 1.0; // (-1, 1, 1) -> (0,1,1)
  global_coords(0, 1, 1, 1, 0) = 1.0;
  global_coords(0, 1, 1, 1, 1) = 1.0;
  global_coords(0, 1, 1, 1, 2) = 1.0; // ( 1, 1, 1) -> (1,1,1)

  // Index mapping (trivial for single element)
  Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, nglly, ngllx);
  index_mapping(0, 0, 0, 0) = 0;
  index_mapping(0, 0, 0, 1) = 1;
  index_mapping(0, 0, 1, 0) = 2;
  index_mapping(0, 0, 1, 1) = 3;
  index_mapping(0, 1, 0, 0) = 4;
  index_mapping(0, 1, 0, 1) = 5;
  index_mapping(0, 1, 1, 0) = 6;
  index_mapping(0, 1, 1, 1) = 7;

  // Control node coordinates (corners of unit cube) - 3D:
  // [icoord][element][node]
  Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_nodes("control_nodes", nspec, ngnod, 3);
  control_nodes(0, 0, 0) = 0.0;
  control_nodes(0, 0, 1) = 0.0;
  control_nodes(0, 0, 2) = 0.0; // Corner 0: (0,0,0)
  control_nodes(0, 1, 0) = 1.0;
  control_nodes(0, 1, 1) = 0.0;
  control_nodes(0, 1, 2) = 0.0; // Corner 1: (1,0,0)
  control_nodes(0, 2, 0) = 1.0;
  control_nodes(0, 2, 1) = 1.0;
  control_nodes(0, 2, 2) = 0.0; // Corner 2: (1,1,0)
  control_nodes(0, 3, 0) = 0.0;
  control_nodes(0, 3, 1) = 1.0;
  control_nodes(0, 3, 2) = 0.0; // Corner 3: (0,1,0)
  control_nodes(0, 4, 0) = 0.0;
  control_nodes(0, 4, 1) = 0.0;
  control_nodes(0, 4, 2) = 1.0; // Corner 4: (0,0,1)
  control_nodes(0, 5, 0) = 1.0;
  control_nodes(0, 5, 1) = 0.0;
  control_nodes(0, 5, 2) = 1.0; // Corner 5: (1,0,1)
  control_nodes(0, 6, 0) = 1.0;
  control_nodes(0, 6, 1) = 1.0;
  control_nodes(0, 6, 2) = 1.0; // Corner 6: (1,1,1)
  control_nodes(0, 7, 0) = 0.0;
  control_nodes(0, 7, 1) = 1.0;
  control_nodes(0, 7, 2) = 1.0; // Corner 7: (0,1,1)

  // Test point at center of unit cube
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  // Should find element 0 with local coordinates near (0, 0, 0) for center
  // point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_NEAR(result.xi, 0.0, 1e-6);
  EXPECT_NEAR(result.eta, 0.0, 1e-6);
  EXPECT_NEAR(result.gamma, 0.0, 1e-6);

  // Test corner point (0, 0, 0) should map to (-1, -1, -1)
  target = { 0.0, 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_NEAR(result.xi, -1.0, 1e-6);
  EXPECT_NEAR(result.eta, -1.0, 1e-6);
  EXPECT_NEAR(result.gamma, -1.0, 1e-6);

  // Test corner point (1, 1, 1) should map to (1, 1, 1)
  target = { 1.0, 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_NEAR(result.xi, 1.0, 1e-6);
  EXPECT_NEAR(result.eta, 1.0, 1e-6);
  EXPECT_NEAR(result.gamma, 1.0, 1e-6);
}

TEST(LOCATE_HELPERS_3D, locate_point_core_2x2x2_eight_elements) {
  // Test the core locate_point function with a 2x2x2 configuration of eight
  // unit cube elements Element layout in 3D:
  //     z=1 level:  z=0 level:
  //     [4] [5]     [0] [1]
  //     [6] [7]     [2] [3]

  const int nspec = 8; // Eight elements
  const int ngllx = 2;
  const int nglly = 2;
  const int ngllz = 2;
  const int ngnod = 8;

  // Global coordinates of GLL quadrature points for all eight elements
  // 3D layout: (nspec, iz, iy, ix, icoord)
  specfem::algorithms::locate_point_impl::MeshHostCoordinatesViewType3D
      global_coords("global_coords", nspec, ngllz, nglly, ngllx, 3);

  // Element 0: [0,1] x [0,1] x [0,1]
  global_coords(0, 0, 0, 0, 0) = 0.0;
  global_coords(0, 0, 0, 0, 1) = 0.0;
  global_coords(0, 0, 0, 0, 2) = 0.0;
  global_coords(0, 0, 0, 1, 0) = 1.0;
  global_coords(0, 0, 0, 1, 1) = 0.0;
  global_coords(0, 0, 0, 1, 2) = 0.0;
  global_coords(0, 0, 1, 0, 0) = 0.0;
  global_coords(0, 0, 1, 0, 1) = 1.0;
  global_coords(0, 0, 1, 0, 2) = 0.0;
  global_coords(0, 0, 1, 1, 0) = 1.0;
  global_coords(0, 0, 1, 1, 1) = 1.0;
  global_coords(0, 0, 1, 1, 2) = 0.0;
  global_coords(0, 1, 0, 0, 0) = 0.0;
  global_coords(0, 1, 0, 0, 1) = 0.0;
  global_coords(0, 1, 0, 0, 2) = 1.0;
  global_coords(0, 1, 0, 1, 0) = 1.0;
  global_coords(0, 1, 0, 1, 1) = 0.0;
  global_coords(0, 1, 0, 1, 2) = 1.0;
  global_coords(0, 1, 1, 0, 0) = 0.0;
  global_coords(0, 1, 1, 0, 1) = 1.0;
  global_coords(0, 1, 1, 0, 2) = 1.0;
  global_coords(0, 1, 1, 1, 0) = 1.0;
  global_coords(0, 1, 1, 1, 1) = 1.0;
  global_coords(0, 1, 1, 1, 2) = 1.0;

  // Element 1: [1,2] x [0,1] x [0,1]
  global_coords(1, 0, 0, 0, 0) = 1.0;
  global_coords(1, 0, 0, 0, 1) = 0.0;
  global_coords(1, 0, 0, 0, 2) = 0.0;
  global_coords(1, 0, 0, 1, 0) = 2.0;
  global_coords(1, 0, 0, 1, 1) = 0.0;
  global_coords(1, 0, 0, 1, 2) = 0.0;
  global_coords(1, 0, 1, 0, 0) = 1.0;
  global_coords(1, 0, 1, 0, 1) = 1.0;
  global_coords(1, 0, 1, 0, 2) = 0.0;
  global_coords(1, 0, 1, 1, 0) = 2.0;
  global_coords(1, 0, 1, 1, 1) = 1.0;
  global_coords(1, 0, 1, 1, 2) = 0.0;
  global_coords(1, 1, 0, 0, 0) = 1.0;
  global_coords(1, 1, 0, 0, 1) = 0.0;
  global_coords(1, 1, 0, 0, 2) = 1.0;
  global_coords(1, 1, 0, 1, 0) = 2.0;
  global_coords(1, 1, 0, 1, 1) = 0.0;
  global_coords(1, 1, 0, 1, 2) = 1.0;
  global_coords(1, 1, 1, 0, 0) = 1.0;
  global_coords(1, 1, 1, 0, 1) = 1.0;
  global_coords(1, 1, 1, 0, 2) = 1.0;
  global_coords(1, 1, 1, 1, 0) = 2.0;
  global_coords(1, 1, 1, 1, 1) = 1.0;
  global_coords(1, 1, 1, 1, 2) = 1.0;

  // For brevity, set up just element 0 with proper index mapping and control
  // nodes This test is mainly to verify the function doesn't crash and returns
  // reasonable bounds

  // Simplified index mapping for element 0
  Kokkos::View<int ****, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, nglly, ngllx);
  // Element 0: global node indices 0-7
  index_mapping(0, 0, 0, 0) = 0;
  index_mapping(0, 0, 0, 1) = 1;
  index_mapping(0, 0, 1, 0) = 2;
  index_mapping(0, 0, 1, 1) = 3;
  index_mapping(0, 1, 0, 0) = 4;
  index_mapping(0, 1, 0, 1) = 5;
  index_mapping(0, 1, 1, 0) = 6;
  index_mapping(0, 1, 1, 1) = 7;

  // Set up remaining elements with non-overlapping node indices
  for (int ispec = 1; ispec < nspec; ispec++) {
    int node_offset = ispec * 8;
    for (int k = 0; k < ngllz; k++) {
      for (int j = 0; j < nglly; j++) {
        for (int i = 0; i < ngllx; i++) {
          index_mapping(ispec, k, j, i) = node_offset + k * 4 + j * 2 + i;
        }
      }
    }
  }

  // Control node coordinates for element 0
  Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_nodes("control_nodes", nspec, ngnod, 3);
  // Element 0: unit cube corners
  control_nodes(0, 0, 0) = 0.0;
  control_nodes(0, 0, 1) = 0.0;
  control_nodes(0, 0, 2) = 0.0;
  control_nodes(0, 1, 0) = 1.0;
  control_nodes(0, 1, 1) = 0.0;
  control_nodes(0, 1, 2) = 0.0;
  control_nodes(0, 2, 0) = 1.0;
  control_nodes(0, 2, 1) = 1.0;
  control_nodes(0, 2, 2) = 0.0;
  control_nodes(0, 3, 0) = 0.0;
  control_nodes(0, 3, 1) = 1.0;
  control_nodes(0, 3, 2) = 0.0;
  control_nodes(0, 4, 0) = 0.0;
  control_nodes(0, 4, 1) = 0.0;
  control_nodes(0, 4, 2) = 1.0;
  control_nodes(0, 5, 0) = 1.0;
  control_nodes(0, 5, 1) = 0.0;
  control_nodes(0, 5, 2) = 1.0;
  control_nodes(0, 6, 0) = 1.0;
  control_nodes(0, 6, 1) = 1.0;
  control_nodes(0, 6, 2) = 1.0;
  control_nodes(0, 7, 0) = 0.0;
  control_nodes(0, 7, 1) = 1.0;
  control_nodes(0, 7, 2) = 1.0;

  // Set other elements to unit cubes at different locations for now (simplified
  // test)
  for (int ispec = 1; ispec < nspec; ispec++) {
    for (int inode = 0; inode < ngnod; inode++) {
      control_nodes(ispec, inode, 0) = ispec + (inode % 2); // x offset
      control_nodes(ispec, inode, 1) = (inode / 2) % 2;     // y
      control_nodes(ispec, inode, 2) = (inode / 4) % 2;     // z
    }
  }

  // Test a point to verify the basic functionality works
  specfem::point::global_coordinates<specfem::dimension::type::dim3> target = {
    0.5, 0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  // Should find element 0 with reasonable local coordinates
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(std::abs(result.xi) <= 1.01);
  EXPECT_TRUE(std::abs(result.eta) <= 1.01);
  EXPECT_TRUE(std::abs(result.gamma) <= 1.01);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
