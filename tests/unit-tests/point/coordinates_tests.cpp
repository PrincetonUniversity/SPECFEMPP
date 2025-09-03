#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
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

  // Constructor with parameters
  const int ispec = 5;
  const type_real xi = 0.5;
  const type_real gamma = -0.3;
  specfem::point::local_coordinates<specfem::dimension::type::dim2> local(
      ispec, xi, gamma);

  // Check values
  EXPECT_EQ(local.ispec, ispec);
  EXPECT_REAL_EQ(local.xi, xi);
  EXPECT_REAL_EQ(local.gamma, gamma);
}

// Test 2D global coordinates
TEST_F(PointCoordinatesTest, GlobalCoordinates2D) {

  // Constructor with parameters
  const type_real x = 10.5;
  const type_real z = -3.2;
  specfem::point::global_coordinates<specfem::dimension::type::dim2> global(x,
                                                                            z);

  // Check values
  EXPECT_REAL_EQ(global.x, x);
  EXPECT_REAL_EQ(global.z, z);
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
  EXPECT_REAL_EQ(dist, 5.0);
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
  EXPECT_REAL_EQ(dist_forward, dist_backward);
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
  EXPECT_REAL_EQ(dist, 0.0);
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
  EXPECT_REAL_EQ(dist, expected);
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
  EXPECT_REAL_EQ(dist, expected);
}

// Tests for 3D coordinates

// Test 3D local coordinates
TEST_F(PointCoordinatesTest, LocalCoordinates3D) {

  // Constructor with parameters
  const int ispec = 8;
  const type_real xi = 0.7;
  const type_real eta = 0.2;
  const type_real gamma = -0.4;
  specfem::point::local_coordinates<specfem::dimension::type::dim3> local(
      ispec, xi, eta, gamma);

  // Check values
  EXPECT_EQ(local.ispec, ispec);
  EXPECT_REAL_EQ(local.xi, xi);
  EXPECT_REAL_EQ(local.eta, eta);
  EXPECT_REAL_EQ(local.gamma, gamma);
}

// Test 3D global coordinates
TEST_F(PointCoordinatesTest, GlobalCoordinates3D) {

  // Constructor with parameters
  const type_real x = 11.5;
  const type_real y = 22.7;
  const type_real z = -4.3;
  specfem::point::global_coordinates<specfem::dimension::type::dim3> global(
      x, y, z);

  // Check values
  EXPECT_REAL_EQ(global.x, x);
  EXPECT_REAL_EQ(global.y, y);
  EXPECT_REAL_EQ(global.z, z);
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
  EXPECT_REAL_EQ(dist, 13.0);
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
  EXPECT_REAL_EQ(dist_forward, dist_backward);
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
  EXPECT_REAL_EQ(dist, 0.0);
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
  EXPECT_REAL_EQ(dist, expected);
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
  EXPECT_REAL_EQ(dist, expected);
}

// ======================= Array Constructor Tests =======================

// Test 2D global coordinates array constructor
TEST_F(PointCoordinatesTest, GlobalCoordinates2D_ArrayConstructor) {
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = 3.14;
  coord_array[1] = 2.71;

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coords(
      coord_array);

  EXPECT_REAL_EQ(coords.x, 3.14);
  EXPECT_REAL_EQ(coords.z, 2.71);
}

// Test 3D global coordinates array constructor
TEST_F(PointCoordinatesTest, GlobalCoordinates3D_ArrayConstructor) {
  Kokkos::View<type_real[3], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = 3.14;
  coord_array[1] = 1.41;
  coord_array[2] = 2.71;

  specfem::point::global_coordinates<specfem::dimension::type::dim3> coords(
      coord_array);

  EXPECT_REAL_EQ(coords.x, 3.14);
  EXPECT_REAL_EQ(coords.y, 1.41);
  EXPECT_REAL_EQ(coords.z, 2.71);
}

// Test 2D local coordinates array constructor
TEST_F(PointCoordinatesTest, LocalCoordinates2D_ArrayConstructor) {
  int ispec = 123;
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = -0.3;
  coord_array[1] = 0.7;

  specfem::point::local_coordinates<specfem::dimension::type::dim2> coords(
      ispec, coord_array);

  EXPECT_EQ(coords.ispec, ispec);
  EXPECT_REAL_EQ(coords.xi, -0.3);
  EXPECT_REAL_EQ(coords.gamma, 0.7);
}

// Test 3D local coordinates array constructor
TEST_F(PointCoordinatesTest, LocalCoordinates3D_ArrayConstructor) {
  int ispec = 456;
  Kokkos::View<type_real[3], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = -0.3;
  coord_array[1] = 0.1;
  coord_array[2] = 0.7;

  specfem::point::local_coordinates<specfem::dimension::type::dim3> coords(
      ispec, coord_array);

  EXPECT_EQ(coords.ispec, ispec);
  EXPECT_REAL_EQ(coords.xi, -0.3);
  EXPECT_REAL_EQ(coords.eta, 0.1);
  EXPECT_REAL_EQ(coords.gamma, 0.7);
}

// Test equivalence between array and value constructors for 2D global
TEST_F(PointCoordinatesTest, GlobalCoordinates2D_ArrayValueEquivalence) {
  type_real x = 5.5;
  type_real z = -2.3;

  // Value constructor
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      coords_value(x, z);

  // Array constructor
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = x;
  coord_array[1] = z;
  specfem::point::global_coordinates<specfem::dimension::type::dim2>
      coords_array(coord_array);

  // Check equivalence
  EXPECT_REAL_EQ(coords_value.x, coords_array.x);
  EXPECT_REAL_EQ(coords_value.z, coords_array.z);
}

// Test equivalence between array and value constructors for 3D global
TEST_F(PointCoordinatesTest, GlobalCoordinates3D_ArrayValueEquivalence) {
  type_real x = 5.5;
  type_real y = 1.2;
  type_real z = -2.3;

  // Value constructor
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
      coords_value(x, y, z);

  // Array constructor
  Kokkos::View<type_real[3], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = x;
  coord_array[1] = y;
  coord_array[2] = z;
  specfem::point::global_coordinates<specfem::dimension::type::dim3>
      coords_array(coord_array);

  // Check equivalence
  EXPECT_REAL_EQ(coords_value.x, coords_array.x);
  EXPECT_REAL_EQ(coords_value.y, coords_array.y);
  EXPECT_REAL_EQ(coords_value.z, coords_array.z);
}

// Test equivalence between array and value constructors for 2D local
TEST_F(PointCoordinatesTest, LocalCoordinates2D_ArrayValueEquivalence) {
  int ispec = 789;
  type_real xi = 0.8;
  type_real gamma = -0.6;

  // Value constructor
  specfem::point::local_coordinates<specfem::dimension::type::dim2>
      coords_value(ispec, xi, gamma);

  // Array constructor
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = xi;
  coord_array[1] = gamma;
  specfem::point::local_coordinates<specfem::dimension::type::dim2>
      coords_array(ispec, coord_array);

  // Check equivalence
  EXPECT_EQ(coords_value.ispec, coords_array.ispec);
  EXPECT_REAL_EQ(coords_value.xi, coords_array.xi);
  EXPECT_REAL_EQ(coords_value.gamma, coords_array.gamma);
}

// Test equivalence between array and value constructors for 3D local
TEST_F(PointCoordinatesTest, LocalCoordinates3D_ArrayValueEquivalence) {
  int ispec = 999;
  type_real xi = 0.8;
  type_real eta = -0.2;
  type_real gamma = -0.6;

  // Value constructor
  specfem::point::local_coordinates<specfem::dimension::type::dim3>
      coords_value(ispec, xi, eta, gamma);

  // Array constructor
  Kokkos::View<type_real[3], Kokkos::HostSpace> coord_array("coords");
  coord_array[0] = xi;
  coord_array[1] = eta;
  coord_array[2] = gamma;
  specfem::point::local_coordinates<specfem::dimension::type::dim3>
      coords_array(ispec, coord_array);

  // Check equivalence
  EXPECT_EQ(coords_value.ispec, coords_array.ispec);
  EXPECT_REAL_EQ(coords_value.xi, coords_array.xi);
  EXPECT_REAL_EQ(coords_value.eta, coords_array.eta);
  EXPECT_REAL_EQ(coords_value.gamma, coords_array.gamma);
}

// Test with zero values using array constructor
TEST_F(PointCoordinatesTest, ArrayConstructor_ZeroValues) {
  type_real zero = 0.0;

  // 2D global with zeros
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array_2d("coords_2d");
  coord_array_2d[0] = zero;
  coord_array_2d[1] = zero;
  specfem::point::global_coordinates<specfem::dimension::type::dim2> global_2d(
      coord_array_2d);

  // 3D global with zeros
  Kokkos::View<type_real[3], Kokkos::HostSpace> coord_array_3d("coords_3d");
  coord_array_3d[0] = zero;
  coord_array_3d[1] = zero;
  coord_array_3d[2] = zero;
  specfem::point::global_coordinates<specfem::dimension::type::dim3> global_3d(
      coord_array_3d);

  EXPECT_REAL_EQ(global_2d.x, zero);
  EXPECT_REAL_EQ(global_2d.z, zero);
  EXPECT_REAL_EQ(global_3d.x, zero);
  EXPECT_REAL_EQ(global_3d.y, zero);
  EXPECT_REAL_EQ(global_3d.z, zero);
}

// Test with extreme values using array constructor
TEST_F(PointCoordinatesTest, ArrayConstructor_ExtremeValues) {
  type_real large = 1e10;
  type_real small = -1e10;
  int large_int = 999999;

  // 2D global with extreme values
  Kokkos::View<type_real[2], Kokkos::HostSpace> coord_array_2d("coords_2d");
  coord_array_2d[0] = large;
  coord_array_2d[1] = small;
  specfem::point::global_coordinates<specfem::dimension::type::dim2> global_2d(
      coord_array_2d);

  // 2D local with extreme values
  Kokkos::View<type_real[2], Kokkos::HostSpace> local_array_2d("local_2d");
  local_array_2d[0] = small;
  local_array_2d[1] = large;
  specfem::point::local_coordinates<specfem::dimension::type::dim2> local_2d(
      large_int, local_array_2d);

  EXPECT_REAL_EQ(global_2d.x, large);
  EXPECT_REAL_EQ(global_2d.z, small);
  EXPECT_EQ(local_2d.ispec, large_int);
  EXPECT_REAL_EQ(local_2d.xi, small);
  EXPECT_REAL_EQ(local_2d.gamma, large);
}
