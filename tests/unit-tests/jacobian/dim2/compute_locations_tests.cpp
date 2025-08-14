#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "kokkos_abstractions.h"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/simd.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::jacobian::compute_locations;

class ComputeLocationsDim2Test : public ::testing::Test {
protected:
  void SetUp() override {
    // Common test setup can go here
  }

  void TearDown() override {
    // Common test cleanup can go here
  }
};

TEST_F(ComputeLocationsDim2Test, UnitSquareCornerNodes) {
  // Test with a unit square element using 4 corner nodes
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit square control nodes (corners)
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 1.0, 0.0 }; // Bottom-right
  coorg(2) = { 1.0, 1.0 }; // Top-right
  coorg(3) = { 0.0, 1.0 }; // Top-left

  // Test center point (0, 0) in reference coordinates -> (0.5, 0.5) in physical
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.z);

  // Test corner at (-1, -1) in reference -> (0, 0) in physical
  result = compute_locations(coorg, ngnod, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test corner at (1, -1) in reference -> (1, 0) in physical
  result = compute_locations(coorg, ngnod, 1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test corner at (1, 1) in reference -> (1, 1) in physical
  result = compute_locations(coorg, ngnod, 1.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.z);

  // Test corner at (-1, 1) in reference -> (0, 1) in physical
  result = compute_locations(coorg, ngnod, -1.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.z);
}

TEST_F(ComputeLocationsDim2Test, ScaledAndTranslatedSquare) {
  // Test with a scaled and translated square element
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Square with corner at (2,3) and side length 4
  coorg(0) = { 2.0, 3.0 }; // Bottom-left
  coorg(1) = { 6.0, 3.0 }; // Bottom-right
  coorg(2) = { 6.0, 7.0 }; // Top-right
  coorg(3) = { 2.0, 7.0 }; // Top-left

  // Test center point (0, 0) in reference -> (4, 5) in physical
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(4.0)))
      << expected_got(4.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(5.0)))
      << expected_got(5.0, result.z);

  // Test corner at (-1, -1) in reference -> (2, 3) in physical
  result = compute_locations(coorg, ngnod, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(3.0)))
      << expected_got(3.0, result.z);

  // Test arbitrary point (0.5, -0.5) in reference
  result = compute_locations(coorg, ngnod, 0.5, -0.5);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(5.0)))
      << expected_got(5.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(4.0)))
      << expected_got(4.0, result.z);
}

TEST_F(ComputeLocationsDim2Test, QuadrilateralElement) {
  // Test with a non-square quadrilateral element
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Arbitrary quadrilateral
  coorg(0) = { 0.0, 0.0 };  // Bottom-left
  coorg(1) = { 2.0, 0.5 };  // Bottom-right
  coorg(2) = { 1.8, 2.2 };  // Top-right
  coorg(3) = { -0.2, 1.8 }; // Top-left

  // Test center point (0, 0) in reference coordinates
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0);

  // Center should be average of corner coordinates for bilinear mapping
  type_real expected_x = (0.0 + 2.0 + 1.8 + (-0.2)) / 4.0;
  type_real expected_z = (0.0 + 0.5 + 2.2 + 1.8) / 4.0;
  EXPECT_TRUE(specfem::utilities::is_close(result.x, expected_x))
      << expected_got(expected_x, result.x);
  EXPECT_TRUE(specfem::utilities::is_close(result.z, expected_z))
      << expected_got(expected_z, result.z);

  // Test corner at (-1, -1) in reference -> should map to coorg(0)
  result = compute_locations(coorg, ngnod, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);
}

TEST_F(ComputeLocationsDim2Test, NineNodeElement) {
  // Test with a 9-node quadrilateral element
  const int ngnod = 9;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // 3x3 unit square grid: 9 nodes at regular positions
  // Node ordering follows standard spectral element convention
  // Bottom row: nodes 0, 4, 1
  // Middle row: nodes 7, 8, 5
  // Top row:    nodes 3, 6, 2
  coorg(0) = { 0.0, 0.0 }; // Bottom-left corner
  coorg(1) = { 1.0, 0.0 }; // Bottom-right corner
  coorg(2) = { 1.0, 1.0 }; // Top-right corner
  coorg(3) = { 0.0, 1.0 }; // Top-left corner
  coorg(4) = { 0.5, 0.0 }; // Bottom-middle
  coorg(5) = { 1.0, 0.5 }; // Right-middle
  coorg(6) = { 0.5, 1.0 }; // Top-middle
  coorg(7) = { 0.0, 0.5 }; // Left-middle
  coorg(8) = { 0.5, 0.5 }; // Center

  // Test center point (0, 0) in reference -> should map to center node
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.z);

  // Test corner at (-1, -1) in reference -> should map to node 0
  result = compute_locations(coorg, ngnod, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test at (1, 1) in reference -> should map to node 2
  result = compute_locations(coorg, ngnod, 1.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.z);
}

TEST_F(ComputeLocationsDim2Test, BilinearMapping) {
  // Test that the mapping is indeed bilinear for 4-node elements
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit square
  coorg(0) = { 0.0, 0.0 };
  coorg(1) = { 1.0, 0.0 };
  coorg(2) = { 1.0, 1.0 };
  coorg(3) = { 0.0, 1.0 };

  // Test linearity in xi direction (gamma = 0)
  auto result1 = compute_locations(coorg, ngnod, -0.5, 0.0);
  auto result2 = compute_locations(coorg, ngnod, 0.0, 0.0);
  auto result3 = compute_locations(coorg, ngnod, 0.5, 0.0);

  // Should have equal spacing along line
  type_real diff1_x = result2.x - result1.x;
  type_real diff2_x = result3.x - result2.x;
  type_real diff1_z = result2.z - result1.z;
  type_real diff2_z = result3.z - result2.z;
  EXPECT_TRUE(specfem::utilities::is_close(diff1_x, diff2_x))
      << expected_got(diff2_x, diff1_x);
  EXPECT_TRUE(specfem::utilities::is_close(diff1_z, diff2_z))
      << expected_got(diff2_z, diff1_z);

  // Test linearity in gamma direction (xi = 0)
  result1 = compute_locations(coorg, ngnod, 0.0, -0.5);
  result2 = compute_locations(coorg, ngnod, 0.0, 0.0);
  result3 = compute_locations(coorg, ngnod, 0.0, 0.5);

  // Should have equal spacing along line
  diff1_x = result2.x - result1.x;
  diff2_x = result3.x - result2.x;
  diff1_z = result2.z - result1.z;
  diff2_z = result3.z - result2.z;
  EXPECT_TRUE(specfem::utilities::is_close(diff1_x, diff2_x))
      << expected_got(diff2_x, diff1_x);
  EXPECT_TRUE(specfem::utilities::is_close(diff1_z, diff2_z))
      << expected_got(diff2_z, diff1_z);
}

TEST_F(ComputeLocationsDim2Test, EdgeMidpoints) {
  // Test that edge midpoints map correctly for 4-node elements
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  coorg(0) = { 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0 };
  coorg(2) = { 2.0, 3.0 };
  coorg(3) = { 0.0, 3.0 };

  // Bottom edge midpoint (0, -1) in reference
  auto result = compute_locations(coorg, ngnod, 0.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Midpoint between (0,0) and (2,0)
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Right edge midpoint (1, 0) in reference
  result = compute_locations(coorg, ngnod, 1.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.z); // Midpoint between (2,0) and (2,3)

  // Top edge midpoint (0, 1) in reference
  result = compute_locations(coorg, ngnod, 0.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Midpoint between (0,3) and (2,3)
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(3.0)))
      << expected_got(3.0, result.z);

  // Left edge midpoint (-1, 0) in reference
  result = compute_locations(coorg, ngnod, -1.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.z); // Midpoint between (0,0) and (0,3)
}
