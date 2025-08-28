#include "algorithms/locate_point_impl.hpp"
#include "../../test_macros.hpp"
#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "algorithms/locate_point.hpp"
#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "utilities/utilities.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using specfem::utilities::is_close;

// Test helper functions from impl namespa
TEST(LOCATE_HELPERS, rough_location_simple) {
  // Create a simple 2x2 grid of quadrature points
  // Element 0: points at (0,0), (0.5,0), (0,0.5), (0.5,0.5)
  // Element 1: points at (0.5,0), (1,0), (0.5,0.5), (1,0.5)
  const int nspec = 2;
  const int ngllx = 2;
  const int ngllz = 2;

  specfem::kokkos::HostView4d<type_real> coord("coord", 2, nspec, ngllz, ngllx);

  // Element 0
  coord(0, 0, 0, 0) = 0.0;
  coord(1, 0, 0, 0) = 0.0; // (0,0)
  coord(0, 0, 0, 1) = 0.5;
  coord(1, 0, 0, 1) = 0.0; // (0.5,0)
  coord(0, 0, 1, 0) = 0.0;
  coord(1, 0, 1, 0) = 0.5; // (0,0.5)
  coord(0, 0, 1, 1) = 0.5;
  coord(1, 0, 1, 1) = 0.5; // (0.5,0.5)

  // Element 1
  coord(0, 1, 0, 0) = 0.5;
  coord(1, 1, 0, 0) = 0.0; // (0.5,0)
  coord(0, 1, 0, 1) = 1.0;
  coord(1, 1, 0, 1) = 0.0; // (1,0)
  coord(0, 1, 1, 0) = 0.5;
  coord(1, 1, 1, 0) = 0.5; // (0.5,0.5)
  coord(0, 1, 1, 1) = 1.0;
  coord(1, 1, 1, 1) = 0.5; // (1,0.5)

  // Test point close to (0.25, 0.25) - should find element 0, point (0,0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      test_point = { 0.1, 0.1 };

  auto [ix, iz, ispec] =
      specfem::algorithms::locate_point_impl::rough_location(test_point, coord);

  EXPECT_EQ(ispec, 0);
  EXPECT_EQ(ix, 0);
  EXPECT_EQ(iz, 0);

  // Test point close to (0.75, 0.25) - multiple equidistant points exist,
  // rough_location returns the first one found (element 0)
  test_point = { 0.75, 0.25 };
  std::tie(ix, iz, ispec) =
      specfem::algorithms::locate_point_impl::rough_location(test_point, coord);
  EXPECT_EQ(ispec, 0);
  EXPECT_EQ(ix, 1);
  EXPECT_EQ(iz, 0);
}

TEST(LOCATE_HELPERS, get_best_candidates_simple) {
  // Create a simple 2x2 index mapping for 2 elements
  const int nspec = 2;
  const int ngllx = 2;
  const int ngllz = 2;

  Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, ngllx);

  // Element 0: global node indices 0,1,2,3
  index_mapping(0, 0, 0) = 0;
  index_mapping(0, 0, 1) = 1;
  index_mapping(0, 1, 0) = 2;
  index_mapping(0, 1, 1) = 3;

  // Element 1: shares nodes 1,3 with element 0
  index_mapping(1, 0, 0) = 1;
  index_mapping(1, 0, 1) = 4;
  index_mapping(1, 1, 0) = 3;
  index_mapping(1, 1, 1) = 5;

  auto candidates = specfem::algorithms::locate_point_impl::get_best_candidates(
      0, index_mapping);

  // Should return both elements 0 and 1 since they share nodes
  EXPECT_EQ(candidates.size(), 2);
  EXPECT_EQ(candidates[0], 0); // Initial guess element
  EXPECT_EQ(candidates[1], 1); // Neighboring element
}

TEST(LOCATE_HELPERS, get_local_coordinates_unit_square) {
  // Test with a unit square element
  const int ngnod = 4;
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit square control nodes
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 1.0, 0.0 }; // Bottom-right
  coorg(2) = { 1.0, 1.0 }; // Top-right
  coorg(3) = { 0.0, 1.0 }; // Top-left

  // Test point at center of unit square (0.5, 0.5)
  // Should map to local coordinates (0, 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  type_real xi_initial = 0.1; // Start with slight offset
  type_real gamma_initial = 0.1;

  auto [xi_final, gamma_final] =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, xi_initial, gamma_initial);

  // For a unit square, center point (0.5, 0.5) should map to (0, 0) in
  // reference coords
  EXPECT_TRUE(is_close(xi_final, type_real{ 0.0 }))
      << expected_got(0.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 0.0 }))
      << expected_got(0.0, gamma_final);

  // Test corner point (0, 0) should map to (-1, -1)
  target = { 0.0, 0.0 };
  std::tie(xi_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ -1.0 }))
      << expected_got(-1.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ -1.0 }))
      << expected_got(-1.0, gamma_final);

  // Test corner point (1, 1) should map to (1, 1)
  target = { 1.0, 1.0 };
  std::tie(xi_final, gamma_final) =
      specfem::algorithms::locate_point_impl::get_local_coordinates(
          target, coorg, 0.0, 0.0);

  EXPECT_TRUE(is_close(xi_final, type_real{ 1.0 }))
      << expected_got(1.0, xi_final);
  EXPECT_TRUE(is_close(gamma_final, type_real{ 1.0 }))
      << expected_got(1.0, gamma_final);
}

TEST(LOCATE_HELPERS, locate_point_core_unit_square) {
  // Test the core locate_point function with a simple unit square geometry
  // This tests the function without needing a full assembly mesh

  // Create a single unit square element
  const int nspec = 1;
  const int ngllx = 2;
  const int ngllz = 2;
  const int ngnod = 4;

  // Global coordinates of GLL quadrature points for unit square
  specfem::kokkos::HostView4d<type_real> global_coords("global_coords", 2,
                                                       nspec, ngllz, ngllx);

  // GLL quadrature points for unit square element mapped to [-1,1]x[-1,1]
  // reference element Assuming standard GLL points at -1 and +1 map to physical
  // coordinates
  global_coords(0, 0, 0, 0) = 0.0;
  global_coords(1, 0, 0, 0) = 0.0; // (-1,-1) -> (0,0)
  global_coords(0, 0, 0, 1) = 1.0;
  global_coords(1, 0, 0, 1) = 0.0; // ( 1,-1) -> (1,0)
  global_coords(0, 0, 1, 0) = 0.0;
  global_coords(1, 0, 1, 0) = 1.0; // (-1, 1) -> (0,1)
  global_coords(0, 0, 1, 1) = 1.0;
  global_coords(1, 0, 1, 1) = 1.0; // ( 1, 1) -> (1,1)

  // Index mapping (trivial for single element)
  Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, ngllx);
  index_mapping(0, 0, 0) = 0;
  index_mapping(0, 0, 1) = 1;
  index_mapping(0, 1, 0) = 2;
  index_mapping(0, 1, 1) = 3;

  // Control node coordinates (corners of unit square) - 3D view:
  // [coord_dim][element][node]
  Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_nodes("control_nodes", 2, nspec, ngnod);
  control_nodes(0, 0, 0) = 0.0;
  control_nodes(1, 0, 0) = 0.0; // Corner 0: (0,0)
  control_nodes(0, 0, 1) = 1.0;
  control_nodes(1, 0, 1) = 0.0; // Corner 1: (1,0)
  control_nodes(0, 0, 2) = 1.0;
  control_nodes(1, 0, 2) = 1.0; // Corner 2: (1,1)
  control_nodes(0, 0, 3) = 0.0;
  control_nodes(1, 0, 3) = 1.0; // Corner 3: (0,1)

  // Test point at center of unit square
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  using GraphType = boos

      auto result = specfem::algorithms::locate_point_impl::locate_point_core(
          target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  // Should find element 0 with local coordinates near (0, 0) for center point
  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test corner point (0, 0) should map to (-1, -1)
  target = { 0.0, 0.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ -1.0 }))
      << expected_got(-1.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ -1.0 }))
      << expected_got(-1.0, result.gamma);

  // Test corner point (1, 1) should map to (1, 1)
  target = { 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 1.0 }))
      << expected_got(1.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 1.0 }))
      << expected_got(1.0, result.gamma);
}

TEST(LOCATE_HELPERS, locate_point_core_2x2_four_elements) {
  // Test the core locate_point function with a 2x2 configuration of four unit
  // square elements Element layout:
  //   [2] [3]
  //   [0] [1]
  // Each element is a unit square spanning (0,1) x (0,1) in their local
  // coordinate

  const int nspec = 4; // Four elements
  const int ngllx = 2;
  const int ngllz = 2;
  const int ngnod = 4;

  // Global coordinates of GLL quadrature points for all four elements
  specfem::kokkos::HostView4d<type_real> global_coords("global_coords", 2,
                                                       nspec, ngllz, ngllx);

  // Element 0: bottom-left square [0,1] x [0,1]
  global_coords(0, 0, 0, 0) = 0.0;
  global_coords(1, 0, 0, 0) = 0.0; // (0,0)
  global_coords(0, 0, 0, 1) = 1.0;
  global_coords(1, 0, 0, 1) = 0.0; // (1,0)
  global_coords(0, 0, 1, 0) = 0.0;
  global_coords(1, 0, 1, 0) = 1.0; // (0,1)
  global_coords(0, 0, 1, 1) = 1.0;
  global_coords(1, 0, 1, 1) = 1.0; // (1,1)

  // Element 1: bottom-right square [1,2] x [0,1]
  global_coords(0, 1, 0, 0) = 1.0;
  global_coords(1, 1, 0, 0) = 0.0; // (1,0)
  global_coords(0, 1, 0, 1) = 2.0;
  global_coords(1, 1, 0, 1) = 0.0; // (2,0)
  global_coords(0, 1, 1, 0) = 1.0;
  global_coords(1, 1, 1, 0) = 1.0; // (1,1)
  global_coords(0, 1, 1, 1) = 2.0;
  global_coords(1, 1, 1, 1) = 1.0; // (2,1)

  // Element 2: top-left square [0,1] x [1,2]
  global_coords(0, 2, 0, 0) = 0.0;
  global_coords(1, 2, 0, 0) = 1.0; // (0,1)
  global_coords(0, 2, 0, 1) = 1.0;
  global_coords(1, 2, 0, 1) = 1.0; // (1,1)
  global_coords(0, 2, 1, 0) = 0.0;
  global_coords(1, 2, 1, 0) = 2.0; // (0,2)
  global_coords(0, 2, 1, 1) = 1.0;
  global_coords(1, 2, 1, 1) = 2.0; // (1,2)

  // Element 3: top-right square [1,2] x [1,2]
  global_coords(0, 3, 0, 0) = 1.0;
  global_coords(1, 3, 0, 0) = 1.0; // (1,1)
  global_coords(0, 3, 0, 1) = 2.0;
  global_coords(1, 3, 0, 1) = 1.0; // (2,1)
  global_coords(0, 3, 1, 0) = 1.0;
  global_coords(1, 3, 1, 0) = 2.0; // (1,2)
  global_coords(0, 3, 1, 1) = 2.0;
  global_coords(1, 3, 1, 1) = 2.0; // (2,2)

  // Index mapping - global node indices for shared nodes
  Kokkos::View<int ***, Kokkos::LayoutLeft, Kokkos::HostSpace> index_mapping(
      "index_mapping", nspec, ngllz, ngllx);

  // Element 0: nodes 0,1,2,3
  index_mapping(0, 0, 0) = 0;
  index_mapping(0, 0, 1) = 1;
  index_mapping(0, 1, 0) = 2;
  index_mapping(0, 1, 1) = 3;

  // Element 1: shares right edge with element 0 (nodes 1,3), has new nodes 4,5
  index_mapping(1, 0, 0) = 1;
  index_mapping(1, 0, 1) = 4;
  index_mapping(1, 1, 0) = 3;
  index_mapping(1, 1, 1) = 5;

  // Element 2: shares top edge with element 0 (nodes 2,3), has new nodes 6,7
  index_mapping(2, 0, 0) = 2;
  index_mapping(2, 0, 1) = 3;
  index_mapping(2, 1, 0) = 6;
  index_mapping(2, 1, 1) = 7;

  // Element 3: shares corner with elements 0,1,2 (node 3), shares edges, has
  // new node 8
  index_mapping(3, 0, 0) = 3;
  index_mapping(3, 0, 1) = 5;
  index_mapping(3, 1, 0) = 7;
  index_mapping(3, 1, 1) = 8;

  // Control node coordinates for all four elements
  Kokkos::View<type_real ***, Kokkos::LayoutLeft, Kokkos::HostSpace>
      control_nodes("control_nodes", 2, nspec, ngnod);

  // Element 0: [0,1] x [0,1]
  control_nodes(0, 0, 0) = 0.0;
  control_nodes(1, 0, 0) = 0.0; // (0,0)
  control_nodes(0, 0, 1) = 1.0;
  control_nodes(1, 0, 1) = 0.0; // (1,0)
  control_nodes(0, 0, 2) = 1.0;
  control_nodes(1, 0, 2) = 1.0; // (1,1)
  control_nodes(0, 0, 3) = 0.0;
  control_nodes(1, 0, 3) = 1.0; // (0,1)

  // Element 1: [1,2] x [0,1]
  control_nodes(0, 1, 0) = 1.0;
  control_nodes(1, 1, 0) = 0.0; // (1,0)
  control_nodes(0, 1, 1) = 2.0;
  control_nodes(1, 1, 1) = 0.0; // (2,0)
  control_nodes(0, 1, 2) = 2.0;
  control_nodes(1, 1, 2) = 1.0; // (2,1)
  control_nodes(0, 1, 3) = 1.0;
  control_nodes(1, 1, 3) = 1.0; // (1,1)

  // Element 2: [0,1] x [1,2]
  control_nodes(0, 2, 0) = 0.0;
  control_nodes(1, 2, 0) = 1.0; // (0,1)
  control_nodes(0, 2, 1) = 1.0;
  control_nodes(1, 2, 1) = 1.0; // (1,1)
  control_nodes(0, 2, 2) = 1.0;
  control_nodes(1, 2, 2) = 2.0; // (1,2)
  control_nodes(0, 2, 3) = 0.0;
  control_nodes(1, 2, 3) = 2.0; // (0,2)

  // Element 3: [1,2] x [1,2]
  control_nodes(0, 3, 0) = 1.0;
  control_nodes(1, 3, 0) = 1.0; // (1,1)
  control_nodes(0, 3, 1) = 2.0;
  control_nodes(1, 3, 1) = 1.0; // (2,1)
  control_nodes(0, 3, 2) = 2.0;
  control_nodes(1, 3, 2) = 2.0; // (2,2)
  control_nodes(0, 3, 3) = 1.0;
  control_nodes(1, 3, 3) = 2.0; // (1,2)

  // Test points in different elements

  // Test 1: Point (0.5, 0.5) should be in element 0 with local coordinates (0,
  // 0)
  specfem::point::global_coordinates<specfem::dimension::type::dim2> target = {
    0.5, 0.5
  };

  auto result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 0);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test 2: Point (1.5, 0.5) should be in element 1 with local coordinates (0,
  // 0)
  target = { 1.5, 0.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 1);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test 3: Point (0.5, 1.5) should be in element 2 with local coordinates (0,
  // 0)
  target = { 0.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 2);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test 4: Point (1.5, 1.5) should be in element 3 with local coordinates (0,
  // 0)
  target = { 1.5, 1.5 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 3);
  EXPECT_TRUE(is_close(result.xi, type_real{ 0.0 }))
      << expected_got(0.0, result.xi);
  EXPECT_TRUE(is_close(result.gamma, type_real{ 0.0 }))
      << expected_got(0.0, result.gamma);

  // Test 5: Corner point (1, 1) - shared by all four elements, should find one
  // of them
  target = { 1.0, 1.0 };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  // Should find one of the elements (0, 1, 2, or 3) - all have this corner
  EXPECT_TRUE(result.ispec >= 0 && result.ispec <= 3);
  // At corner, should have local coordinates (±1, ±1) depending on element
  EXPECT_TRUE(std::abs(std::abs(result.xi) - 1.0) < 1e-6);
  EXPECT_TRUE(std::abs(std::abs(result.gamma) - 1.0) < 1e-6);

  // One off test: point (1.33, 1.33) should be in element 3 with local
  // coordinates (0.66, 0.66)
  type_real one_third = type_real{ 1.0 } / type_real{ 3.0 };
  target = { 1 + one_third, 1 + one_third };
  result = specfem::algorithms::locate_point_impl::locate_point_core(
      target, global_coords, index_mapping, control_nodes, ngnod, ngllx);

  EXPECT_EQ(result.ispec, 3);
  EXPECT_TRUE(is_close(result.xi, -one_third))
      << expected_got(-one_third, result.xi);
  EXPECT_TRUE(is_close(result.gamma, -one_third))
      << expected_got(-one_third, result.gamma);

  // Add one test for a point that is not in any element
  // Point (2.5, 2.5) should not be in any element
  target = { 2.5, 2.5 };
  EXPECT_THROW(
      specfem::algorithms::locate_point_impl::locate_point_core(
          target, global_coords, index_mapping, control_nodes, ngnod, ngllx),
      std::runtime_error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
