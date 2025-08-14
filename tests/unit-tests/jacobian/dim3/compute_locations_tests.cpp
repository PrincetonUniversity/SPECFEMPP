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

class ComputeLocationsDim3Test : public ::testing::Test {
protected:
  void SetUp() override {
    // Common test setup can go here
  }

  void TearDown() override {
    // Common test cleanup can go here
  }
};

TEST_F(ComputeLocationsDim3Test, UnitCubeCornerNodes) {
  // Test with a unit cube element using 8 corner nodes
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube control nodes (corners) - standard spectral element ordering
  coorg(0) = { 0.0, 0.0, 0.0 }; // Corner 0: (0,0,0)
  coorg(1) = { 1.0, 0.0, 0.0 }; // Corner 1: (1,0,0)
  coorg(2) = { 1.0, 1.0, 0.0 }; // Corner 2: (1,1,0)
  coorg(3) = { 0.0, 1.0, 0.0 }; // Corner 3: (0,1,0)
  coorg(4) = { 0.0, 0.0, 1.0 }; // Corner 4: (0,0,1)
  coorg(5) = { 1.0, 0.0, 1.0 }; // Corner 5: (1,0,1)
  coorg(6) = { 1.0, 1.0, 1.0 }; // Corner 6: (1,1,1)
  coorg(7) = { 0.0, 1.0, 1.0 }; // Corner 7: (0,1,1)

  // Test center point (0, 0, 0) in reference coordinates -> (0.5, 0.5, 0.5) in
  // physical
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.z);

  // Test corner at (-1, -1, -1) in reference -> (0, 0, 0) in physical
  result = compute_locations(coorg, ngnod, -1.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test corner at (1, -1, -1) in reference -> (1, 0, 0) in physical
  result = compute_locations(coorg, ngnod, 1.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test corner at (1, 1, 1) in reference -> (1, 1, 1) in physical
  result = compute_locations(coorg, ngnod, 1.0, 1.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.z);

  // Test corner at (-1, -1, 1) in reference -> (0, 0, 1) in physical
  result = compute_locations(coorg, ngnod, -1.0, -1.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.z);
}

TEST_F(ComputeLocationsDim3Test, ScaledAndTranslatedCube) {
  // Test with a scaled and translated cube element
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Cube with corner at (2,3,4) and side length 2
  coorg(0) = { 2.0, 3.0, 4.0 }; // Corner 0: (2,3,4)
  coorg(1) = { 4.0, 3.0, 4.0 }; // Corner 1: (4,3,4)
  coorg(2) = { 4.0, 5.0, 4.0 }; // Corner 2: (4,5,4)
  coorg(3) = { 2.0, 5.0, 4.0 }; // Corner 3: (2,5,4)
  coorg(4) = { 2.0, 3.0, 6.0 }; // Corner 4: (2,3,6)
  coorg(5) = { 4.0, 3.0, 6.0 }; // Corner 5: (4,3,6)
  coorg(6) = { 4.0, 5.0, 6.0 }; // Corner 6: (4,5,6)
  coorg(7) = { 2.0, 5.0, 6.0 }; // Corner 7: (2,5,6)

  // Test center point (0, 0, 0) in reference -> (3, 4, 5) in physical
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(3.0)))
      << expected_got(3.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(4.0)))
      << expected_got(4.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(5.0)))
      << expected_got(5.0, result.z);

  // Test corner at (-1, -1, -1) in reference -> (2, 3, 4) in physical
  result = compute_locations(coorg, ngnod, -1.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(3.0)))
      << expected_got(3.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(4.0)))
      << expected_got(4.0, result.z);

  // Test arbitrary point (0.5, -0.5, 0.25) in reference
  result = compute_locations(coorg, ngnod, 0.5, -0.5, 0.25);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(3.5)))
      << expected_got(3.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(3.5)))
      << expected_got(3.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(5.25)))
      << expected_got(5.25, result.z);
}

TEST_F(ComputeLocationsDim3Test, HexahedralElement) {
  // Test with a non-cube hexahedral element
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Arbitrary hexahedron
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 2.0, 0.5, 0.0 };
  coorg(2) = { 1.8, 2.2, 0.2 };
  coorg(3) = { -0.2, 1.8, 0.1 };
  coorg(4) = { 0.1, 0.1, 3.0 };
  coorg(5) = { 2.1, 0.6, 2.8 };
  coorg(6) = { 1.9, 2.3, 3.2 };
  coorg(7) = { -0.1, 1.9, 3.1 };

  // Test center point (0, 0, 0) in reference coordinates
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);

  // Center should be average of corner coordinates for trilinear mapping
  type_real expected_x =
      (0.0 + 2.0 + 1.8 + (-0.2) + 0.1 + 2.1 + 1.9 + (-0.1)) / 8.0;
  type_real expected_y = (0.0 + 0.5 + 2.2 + 1.8 + 0.1 + 0.6 + 2.3 + 1.9) / 8.0;
  type_real expected_z = (0.0 + 0.0 + 0.2 + 0.1 + 3.0 + 2.8 + 3.2 + 3.1) / 8.0;
  EXPECT_TRUE(specfem::utilities::is_close(result.x,
                                           static_cast<type_real>(expected_x)))
      << expected_got(expected_x, result.x);
  EXPECT_TRUE(specfem::utilities::is_close(result.y,
                                           static_cast<type_real>(expected_y)))
      << expected_got(expected_y, result.y);
  EXPECT_TRUE(specfem::utilities::is_close(result.z,
                                           static_cast<type_real>(expected_z)))
      << expected_got(expected_z, result.z);

  // Test corner at (-1, -1, -1) in reference -> should map to coorg(0)
  result = compute_locations(coorg, ngnod, -1.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);
}

TEST_F(ComputeLocationsDim3Test, TrilinearMapping) {
  // Test that the mapping is indeed trilinear for 8-node elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 1.0, 0.0, 0.0 };
  coorg(2) = { 1.0, 1.0, 0.0 };
  coorg(3) = { 0.0, 1.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 1.0 };
  coorg(5) = { 1.0, 0.0, 1.0 };
  coorg(6) = { 1.0, 1.0, 1.0 };
  coorg(7) = { 0.0, 1.0, 1.0 };

  // Test linearity in xi direction (eta = 0, gamma = 0)
  auto result1 = compute_locations(coorg, ngnod, -0.5, 0.0, 0.0);
  auto result2 = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);
  auto result3 = compute_locations(coorg, ngnod, 0.5, 0.0, 0.0);

  // Should have equal spacing along line
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.x - result1.x, static_cast<type_real>(result3.x - result2.x)))
      << expected_got(result3.x - result2.x, result2.x - result1.x);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.y - result1.y, static_cast<type_real>(result3.y - result2.y)))
      << expected_got(result3.y - result2.y, result2.y - result1.y);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.z - result1.z, static_cast<type_real>(result3.z - result2.z)))
      << expected_got(result3.z - result2.z, result2.z - result1.z);

  // Test linearity in eta direction (xi = 0, gamma = 0)
  result1 = compute_locations(coorg, ngnod, 0.0, -0.5, 0.0);
  result2 = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);
  result3 = compute_locations(coorg, ngnod, 0.0, 0.5, 0.0);

  // Should have equal spacing along line
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.x - result1.x, static_cast<type_real>(result3.x - result2.x)))
      << expected_got(result3.x - result2.x, result2.x - result1.x);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.y - result1.y, static_cast<type_real>(result3.y - result2.y)))
      << expected_got(result3.y - result2.y, result2.y - result1.y);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.z - result1.z, static_cast<type_real>(result3.z - result2.z)))
      << expected_got(result3.z - result2.z, result2.z - result1.z);

  // Test linearity in gamma direction (xi = 0, eta = 0)
  result1 = compute_locations(coorg, ngnod, 0.0, 0.0, -0.5);
  result2 = compute_locations(coorg, ngnod, 0.0, 0.0, 0.0);
  result3 = compute_locations(coorg, ngnod, 0.0, 0.0, 0.5);

  // Should have equal spacing along line
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.x - result1.x, static_cast<type_real>(result3.x - result2.x)))
      << expected_got(result3.x - result2.x, result2.x - result1.x);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.y - result1.y, static_cast<type_real>(result3.y - result2.y)))
      << expected_got(result3.y - result2.y, result2.y - result1.y);
  EXPECT_TRUE(specfem::utilities::is_close(
      result2.z - result1.z, static_cast<type_real>(result3.z - result2.z)))
      << expected_got(result3.z - result2.z, result2.z - result1.z);
}

TEST_F(ComputeLocationsDim3Test, FaceCenters) {
  // Test that face centers map correctly for 8-node elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0, 0.0 };
  coorg(2) = { 2.0, 3.0, 0.0 };
  coorg(3) = { 0.0, 3.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 4.0 };
  coorg(5) = { 2.0, 0.0, 4.0 };
  coorg(6) = { 2.0, 3.0, 4.0 };
  coorg(7) = { 0.0, 3.0, 4.0 };

  // Bottom face center (0, 0, -1) in reference
  auto result = compute_locations(coorg, ngnod, 0.0, 0.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Center of bottom face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Top face center (0, 0, 1) in reference
  result = compute_locations(coorg, ngnod, 0.0, 0.0, 1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Center of top face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(4.0)))
      << expected_got(4.0, result.z);

  // Front face center (-1, 0, 0) in reference
  result = compute_locations(coorg, ngnod, -1.0, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x); // Center of front face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.z);

  // Back face center (1, 0, 0) in reference
  result = compute_locations(coorg, ngnod, 1.0, 0.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.x); // Center of back face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(1.5)))
      << expected_got(1.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.z);

  // Left face center (0, -1, 0) in reference
  result = compute_locations(coorg, ngnod, 0.0, -1.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Center of left face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.z);

  // Right face center (0, 1, 0) in reference
  result = compute_locations(coorg, ngnod, 0.0, 1.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Center of right face
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(3.0)))
      << expected_got(3.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.z);
}

TEST_F(ComputeLocationsDim3Test, EdgeMidpoints) {
  // Test that edge midpoints map correctly for 8-node elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0, 0.0 };
  coorg(2) = { 2.0, 3.0, 0.0 };
  coorg(3) = { 0.0, 3.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 4.0 };
  coorg(5) = { 2.0, 0.0, 4.0 };
  coorg(6) = { 2.0, 3.0, 4.0 };
  coorg(7) = { 0.0, 3.0, 4.0 };

  // Bottom edge midpoint (0, -1, -1) in reference - edge between nodes 0 and 1
  auto result = compute_locations(coorg, ngnod, 0.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.x); // Midpoint between (0,0,0) and (2,0,0)
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Vertical edge midpoint (-1, -1, 0) in reference - edge between nodes 0 and
  // 4
  result = compute_locations(coorg, ngnod, -1.0, -1.0, 0.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.z); // Midpoint between (0,0,0) and (0,0,4)
}
