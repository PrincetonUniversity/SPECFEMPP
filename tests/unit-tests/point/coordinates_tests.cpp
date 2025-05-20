#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <gtest/gtest.h>

// Base test fixture for Kokkos initialization
class CoordinatesTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize Kokkos
    Kokkos::initialize();
  }

  void TearDown() override {
    // Finalize Kokkos
    Kokkos::finalize();
  }
};

// Test fixture for 2D coordinates tests
class PointCoordinatesTest : public CoordinatesTest {};

// Tests for 2D coordinates

// Test 2D local coordinates
TEST_F(PointCoordinatesTest, LocalCoordinates2D) {
  // Default constructor
  specfem::point::local_coordinates<specfem::dimension::type::dim2>
      local_default;

  // Constructor with parameters
  const int ispec = 5;
  const type_real xi = 0.5;
  const type_real gamma = -0.3;
  specfem::point::local_coordinates<specfem::dimension::type::dim2> local(
      ispec, xi, gamma);

  // Check values
  EXPECT_EQ(local.ispec, ispec);
  EXPECT_DOUBLE_EQ(local.xi, xi);
  EXPECT_DOUBLE_EQ(local.gamma, gamma);
}

// Test 2D global coordinates
TEST_F(PointCoordinatesTest, GlobalCoordinates2D) {
  // Default constructor
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      global_default;

  // Constructor with parameters
  const type_real x = 10.5;
  const type_real z = -3.2;
  specfem::point::global_coordinates<specfem::dimension::type::dim2> global(x,
                                                                            z);

  // Check values
  EXPECT_DOUBLE_EQ(global.x, x);
  EXPECT_DOUBLE_EQ(global.z, z);
}

// Test 2D distance function
TEST_F(PointCoordinatesTest, Distance2D) {
  // Create two points
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p1(0.0,
                                                                        0.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p2(3.0,
                                                                        4.0);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim2>(p1, p2);

  // Expected result is 5.0 (Pythagorean triangle 3-4-5)
  EXPECT_DOUBLE_EQ(dist, 5.0);
}

// Test 2D distance symmetry
TEST_F(PointCoordinatesTest, DistanceSymmetry2D) {
  // Create two points
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p1(1.0,
                                                                        2.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p2(3.0,
                                                                        5.0);

  // Calculate distances in both directions
  type_real dist_forward =
      specfem::point::distance<specfem::dimension::type::dim2>(p1, p2);
  type_real dist_backward =
      specfem::point::distance<specfem::dimension::type::dim2>(p2, p1);

  // Check symmetry
  EXPECT_DOUBLE_EQ(dist_forward, dist_backward);
}

// Test 2D zero distance
TEST_F(PointCoordinatesTest, ZeroDistance2D) {
  // Create a point
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p(2.5,
                                                                       3.5);

  // Calculate distance to itself
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim2>(p, p);

  // Expected result is 0.0
  EXPECT_DOUBLE_EQ(dist, 0.0);
}

// Test 2D non-integer coordinates
TEST_F(PointCoordinatesTest, NonIntegerCoordinates2D) {
  // Create two points with non-integer coordinates
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p1(1.5,
                                                                        2.5);
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p2(4.5,
                                                                        6.5);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim2>(p1, p2);

  // Expected result
  type_real expected = std::sqrt(3.0 * 3.0 + 4.0 * 4.0);
  EXPECT_DOUBLE_EQ(dist, expected);
}

// Test 2D negative coordinates
TEST_F(PointCoordinatesTest, NegativeCoordinates2D) {
  // Create two points with negative coordinates
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p1(-1.0,
                                                                        -2.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim2> p2(2.0,
                                                                        1.0);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim2>(p1, p2);

  // Expected result
  type_real expected = std::sqrt(3.0 * 3.0 + 3.0 * 3.0);
  EXPECT_DOUBLE_EQ(dist, expected);
}

// Tests for 3D coordinates

// Test 3D local coordinates
TEST_F(PointCoordinatesTest, LocalCoordinates3D) {
  // Default constructor
  specfem::point::local_coordinates<specfem::dimension::type::dim3>
      local_default;

  // Constructor with parameters
  const int ispec = 8;
  const type_real xi = 0.7;
  const type_real eta = 0.2;
  const type_real gamma = -0.4;
  specfem::point::local_coordinates<specfem::dimension::type::dim3> local(
      ispec, xi, eta, gamma);

  // Check values
  EXPECT_EQ(local.ispec, ispec);
  EXPECT_DOUBLE_EQ(local.xi, xi);
  EXPECT_DOUBLE_EQ(local.eta, eta);
  EXPECT_DOUBLE_EQ(local.gamma, gamma);
}

// Test 3D global coordinates
TEST_F(PointCoordinatesTest, GlobalCoordinates3D) {
  // Default constructor
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
      global_default;

  // Constructor with parameters
  const type_real x = 11.5;
  const type_real y = 22.7;
  const type_real z = -4.3;
  specfem::point::global_coordinates<specfem::dimension::type::dim3> global(
      x, y, z);

  // Check values
  EXPECT_DOUBLE_EQ(global.x, x);
  EXPECT_DOUBLE_EQ(global.y, y);
  EXPECT_DOUBLE_EQ(global.z, z);
}

// Test 3D distance function
TEST_F(PointCoordinatesTest, Distance3D) {
  // Create two points
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p1(
      0.0, 0.0, 0.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p2(
      3.0, 4.0, 12.0);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim3>(p1, p2);

  // Expected result is 13.0 (3D extension of the Pythagorean theorem)
  EXPECT_DOUBLE_EQ(dist, 13.0);
}

// Test 3D distance symmetry
TEST_F(PointCoordinatesTest, DistanceSymmetry3D) {
  // Create two points
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p1(
      1.0, 2.0, 3.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p2(
      4.0, 6.0, 8.0);

  // Calculate distances in both directions
  type_real dist_forward =
      specfem::point::distance<specfem::dimension::type::dim3>(p1, p2);
  type_real dist_backward =
      specfem::point::distance<specfem::dimension::type::dim3>(p2, p1);

  // Check symmetry
  EXPECT_DOUBLE_EQ(dist_forward, dist_backward);
}

// Test 3D zero distance
TEST_F(PointCoordinatesTest, ZeroDistance3D) {
  // Create a point
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p(2.5, 3.5,
                                                                       4.5);

  // Calculate distance to itself
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim3>(p, p);

  // Expected result is 0.0
  EXPECT_DOUBLE_EQ(dist, 0.0);
}

// Test 3D non-integer coordinates
TEST_F(PointCoordinatesTest, NonIntegerCoordinates3D) {
  // Create two points with non-integer coordinates
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p1(
      1.5, 2.5, 3.5);
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p2(
      4.5, 6.5, 7.5);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim3>(p1, p2);

  // Expected result
  type_real expected = std::sqrt(3.0 * 3.0 + 4.0 * 4.0 + 4.0 * 4.0);
  EXPECT_DOUBLE_EQ(dist, expected);
}

// Test 3D negative coordinates
TEST_F(PointCoordinatesTest, NegativeCoordinates3D) {
  // Create two points with negative coordinates
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p1(
      -1.0, -2.0, -3.0);
  specfem::point::global_coordinates<specfem::dimension::type::dim3> p2(
      2.0, 1.0, 0.0);

  // Calculate distance
  type_real dist =
      specfem::point::distance<specfem::dimension::type::dim3>(p1, p2);

  // Expected result
  type_real expected = std::sqrt(3.0 * 3.0 + 3.0 * 3.0 + 3.0 * 3.0);
  EXPECT_DOUBLE_EQ(dist, expected);
}

// Main function
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
