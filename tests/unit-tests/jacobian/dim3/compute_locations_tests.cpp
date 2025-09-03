#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "kokkos_abstractions.h"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
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

  // Helper to create unit cube element [0,1] x [0,1] x [0,1]
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  create_unit_cube_8node() {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coorg("coorg", 8);
    coorg(0) = { 0.0, 0.0, 0.0 }; // Corner 0: (0,0,0)
    coorg(1) = { 1.0, 0.0, 0.0 }; // Corner 1: (1,0,0)
    coorg(2) = { 1.0, 1.0, 0.0 }; // Corner 2: (1,1,0)
    coorg(3) = { 0.0, 1.0, 0.0 }; // Corner 3: (0,1,0)
    coorg(4) = { 0.0, 0.0, 1.0 }; // Corner 4: (0,0,1)
    coorg(5) = { 1.0, 0.0, 1.0 }; // Corner 5: (1,0,1)
    coorg(6) = { 1.0, 1.0, 1.0 }; // Corner 6: (1,1,1)
    coorg(7) = { 0.0, 1.0, 1.0 }; // Corner 7: (0,1,1)
    return coorg;
  }

  // Helper to create scaled and translated cube element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  create_scaled_translated_cube_8node(type_real xmin, type_real xmax,
                                      type_real ymin, type_real ymax,
                                      type_real zmin, type_real zmax) {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coorg("coorg", 8);
    coorg(0) = { xmin, ymin, zmin }; // Corner 0: (xmin,ymin,zmin)
    coorg(1) = { xmax, ymin, zmin }; // Corner 1: (xmax,ymin,zmin)
    coorg(2) = { xmax, ymax, zmin }; // Corner 2: (xmax,ymax,zmin)
    coorg(3) = { xmin, ymax, zmin }; // Corner 3: (xmin,ymax,zmin)
    coorg(4) = { xmin, ymin, zmax }; // Corner 4: (xmin,ymin,zmax)
    coorg(5) = { xmax, ymin, zmax }; // Corner 5: (xmax,ymin,zmax)
    coorg(6) = { xmax, ymax, zmax }; // Corner 6: (xmax,ymax,zmax)
    coorg(7) = { xmin, ymax, zmax }; // Corner 7: (xmin,ymax,zmax)
    return coorg;
  }

  // Helper to create arbitrary hexahedral element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  create_hexahedron_8node(
      const std::array<std::array<type_real, 3>, 8> &corners) {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coorg("coorg", 8);
    for (int i = 0; i < 8; ++i) {
      coorg(i) = { corners[i][0], corners[i][1], corners[i][2] };
    }
    return coorg;
  }

  // Helper to create 27-node unit cube element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  create_unit_cube_27node() {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coorg("coorg", 27);

    // Corner nodes (0-7)
    coorg(0) = { 0.0, 0.0, 0.0 };
    coorg(1) = { 1.0, 0.0, 0.0 };
    coorg(2) = { 1.0, 1.0, 0.0 };
    coorg(3) = { 0.0, 1.0, 0.0 };
    coorg(4) = { 0.0, 0.0, 1.0 };
    coorg(5) = { 1.0, 0.0, 1.0 };
    coorg(6) = { 1.0, 1.0, 1.0 };
    coorg(7) = { 0.0, 1.0, 1.0 };

    // Mid-edge nodes (8-19)
    coorg(8) = { 0.5, 0.0, 0.0 };
    coorg(9) = { 1.0, 0.5, 0.0 };
    coorg(10) = { 0.5, 1.0, 0.0 };
    coorg(11) = { 0.0, 0.5, 0.0 };
    coorg(12) = { 0.0, 0.0, 0.5 };
    coorg(13) = { 1.0, 0.0, 0.5 };
    coorg(14) = { 1.0, 1.0, 0.5 };
    coorg(15) = { 0.0, 1.0, 0.5 };
    coorg(16) = { 0.5, 0.0, 1.0 };
    coorg(17) = { 1.0, 0.5, 1.0 };
    coorg(18) = { 0.5, 1.0, 1.0 };
    coorg(19) = { 0.0, 0.5, 1.0 };

    // Face center nodes (20-25)
    coorg(20) = { 0.5, 0.5, 0.0 };
    coorg(21) = { 0.5, 0.0, 0.5 };
    coorg(22) = { 1.0, 0.5, 0.5 };
    coorg(23) = { 0.5, 1.0, 0.5 };
    coorg(24) = { 0.0, 0.5, 0.5 };
    coorg(25) = { 0.5, 0.5, 1.0 };

    // Center node (26)
    coorg(26) = { 0.5, 0.5, 0.5 };

    return coorg;
  }
};

TEST_F(ComputeLocationsDim3Test, UnitCubeCornerNodes) {
  // Test with a unit cube element using 8 corner nodes
  const int ngnod = 8;
  auto coorg = create_unit_cube_8node();

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
  // Cube with corner at (2,3,4) and side length 2: from (2,3,4) to (4,5,6)
  auto coorg =
      create_scaled_translated_cube_8node(2.0, 4.0, 3.0, 5.0, 4.0, 6.0);

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
  // Arbitrary hexahedron
  auto coorg = create_hexahedron_8node({ { { { 0.0, 0.0, 0.0 } },
                                           { { 2.0, 0.5, 0.0 } },
                                           { { 1.8, 2.2, 0.2 } },
                                           { { -0.2, 1.8, 0.1 } },
                                           { { 0.1, 0.1, 3.0 } },
                                           { { 2.1, 0.6, 2.8 } },
                                           { { 1.9, 2.3, 3.2 } },
                                           { { -0.1, 1.9, 3.1 } } } });

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

TEST_F(ComputeLocationsDim3Test, TwentySevenNodeElement) {
  // Test with a 27-node hexahedral element
  const int ngnod = 27;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // 27-node unit cube control nodes (corners + mid-edge + face-center + center)
  // Corner nodes (0-7)
  coorg(0) = { 0.0, 0.0, 0.0 }; // Corner 0: (0,0,0)
  coorg(1) = { 1.0, 0.0, 0.0 }; // Corner 1: (1,0,0)
  coorg(2) = { 1.0, 1.0, 0.0 }; // Corner 2: (1,1,0)
  coorg(3) = { 0.0, 1.0, 0.0 }; // Corner 3: (0,1,0)
  coorg(4) = { 0.0, 0.0, 1.0 }; // Corner 4: (0,0,1)
  coorg(5) = { 1.0, 0.0, 1.0 }; // Corner 5: (1,0,1)
  coorg(6) = { 1.0, 1.0, 1.0 }; // Corner 6: (1,1,1)
  coorg(7) = { 0.0, 1.0, 1.0 }; // Corner 7: (0,1,1)

  // Mid-edge nodes (8-19)
  coorg(8) = { 0.5, 0.0, 0.0 };  // Edge 0-1
  coorg(9) = { 1.0, 0.5, 0.0 };  // Edge 1-2
  coorg(10) = { 0.5, 1.0, 0.0 }; // Edge 2-3
  coorg(11) = { 0.0, 0.5, 0.0 }; // Edge 3-0
  coorg(12) = { 0.0, 0.0, 0.5 }; // Edge 0-4
  coorg(13) = { 1.0, 0.0, 0.5 }; // Edge 1-5
  coorg(14) = { 1.0, 1.0, 0.5 }; // Edge 2-6
  coorg(15) = { 0.0, 1.0, 0.5 }; // Edge 3-7
  coorg(16) = { 0.5, 0.0, 1.0 }; // Edge 4-5
  coorg(17) = { 1.0, 0.5, 1.0 }; // Edge 5-6
  coorg(18) = { 0.5, 1.0, 1.0 }; // Edge 6-7
  coorg(19) = { 0.0, 0.5, 1.0 }; // Edge 7-4

  // Face center nodes (20-25)
  coorg(20) = { 0.5, 0.5, 0.0 }; // Face z=0
  coorg(21) = { 0.5, 0.0, 0.5 }; // Face y=0
  coorg(22) = { 1.0, 0.5, 0.5 }; // Face x=1
  coorg(23) = { 0.5, 1.0, 0.5 }; // Face y=1
  coorg(24) = { 0.0, 0.5, 0.5 }; // Face x=0
  coorg(25) = { 0.5, 0.5, 1.0 }; // Face z=1

  // Center node (26)
  coorg(26) = { 0.5, 0.5, 0.5 }; // Center

  // Test center point (0, 0, 0) in reference -> should map to center node (0.5,
  // 0.5, 0.5)
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

  // Test corner at (-1, -1, -1) in reference -> should map to node 0 (0, 0, 0)
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

  // Test corner at (1, 1, 1) in reference -> should map to node 6 (1, 1, 1)
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

  // Test edge midpoint (0, -1, -1) in reference -> should map to edge node 8
  // (0.5, 0, 0)
  result = compute_locations(coorg, ngnod, 0.0, -1.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);

  // Test face center (0, 0, -1) in reference -> should map to face node 20
  // (0.5, 0.5, 0)
  result = compute_locations(coorg, ngnod, 0.0, 0.0, -1.0);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.x, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.x);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.y, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.y);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.z, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.z);
}
