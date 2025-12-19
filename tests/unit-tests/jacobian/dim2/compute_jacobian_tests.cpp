#include "SPECFEM_Environment.hpp"
#include "kokkos_abstractions.h"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "test_macros.hpp"
#include "utilities/interface.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::jacobian::compute_jacobian;

class ComputeJacobianDim2Test : public ::testing::Test {
protected:
  void SetUp() override {
    // Common test setup can go here
  }

  void TearDown() override {
    // Common test cleanup can go here
  }

  // Helper to create unit square element [0,1] x [0,1]
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
  create_unit_square_4node() {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>
        coorg("coorg", 4);
    coorg(0) = { 0.0, 0.0 }; // Bottom-left
    coorg(1) = { 1.0, 0.0 }; // Bottom-right
    coorg(2) = { 1.0, 1.0 }; // Top-right
    coorg(3) = { 0.0, 1.0 }; // Top-left
    return coorg;
  }

  // Helper to create scaled square element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
  create_scaled_square_4node(type_real scale) {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>
        coorg("coorg", 4);
    coorg(0) = { 0.0, 0.0 };
    coorg(1) = { scale, 0.0 };
    coorg(2) = { scale, scale };
    coorg(3) = { 0.0, scale };
    return coorg;
  }

  // Helper to create translated square element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
  create_translated_unit_square_4node(type_real dx, type_real dz) {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>
        coorg("coorg", 4);
    coorg(0) = { dx, dz };
    coorg(1) = { static_cast<type_real>(1.0) + dx, dz };
    coorg(2) = { static_cast<type_real>(1.0) + dx,
                 static_cast<type_real>(1.0) + dz };
    coorg(3) = { dx, static_cast<type_real>(1.0) + dz };
    return coorg;
  }

  // Helper to create 9-node unit square element
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
  create_unit_square_9node() {
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace>
        coorg("coorg", 9);
    coorg(0) = { 0.0, 0.0 }; // Corner 0: (0,0)
    coorg(1) = { 1.0, 0.0 }; // Corner 1: (1,0)
    coorg(2) = { 1.0, 1.0 }; // Corner 2: (1,1)
    coorg(3) = { 0.0, 1.0 }; // Corner 3: (0,1)
    coorg(4) = { 0.5, 0.0 }; // Mid-edge 4: (0.5,0)
    coorg(5) = { 1.0, 0.5 }; // Mid-edge 5: (1,0.5)
    coorg(6) = { 0.5, 1.0 }; // Mid-edge 6: (0.5,1)
    coorg(7) = { 0.0, 0.5 }; // Mid-edge 7: (0,0.5)
    coorg(8) = { 0.5, 0.5 }; // Center 8: (0.5,0.5)
    return coorg;
  }
};

TEST_F(ComputeJacobianDim2Test, UnitSquareIdentityMapping) {
  // Test with a unit square element using 4 corner nodes
  // This should give identity-like jacobian mapping
  const int ngnod = 4;
  auto coorg = create_unit_square_4node();

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // For unit square, jacobian should be 0.25 (area = 1, reference area = 4)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.25)))
      << expected_got(0.25, result.jacobian);

  // For unit square mapping, the inverse jacobian matrix elements should be:
  // xi_x = 2, gamma_x = 0, xi_z = 0, gamma_z = 2
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, ScaledSquareMapping) {
  // Test with a scaled square (side length 2)
  const int ngnod = 4;
  auto coorg = create_scaled_square_4node(2.0);

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // For 2x2 square, jacobian should be 1.0 (area = 4, reference area = 4)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(1.0)))
      << expected_got(1.0, result.jacobian);

  // For 2x2 square mapping, the inverse jacobian matrix elements should be:
  // xi_x = 1, gamma_x = 0, xi_z = 0, gamma_z = 1
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, TranslatedSquareMapping) {
  // Test with a translated square (translation shouldn't affect jacobian)
  const int ngnod = 4;
  auto coorg = create_translated_unit_square_4node(5.0, 3.0);

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // Translation shouldn't change jacobian values
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.25)))
      << expected_got(0.25, result.jacobian);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, ShearMapping) {
  // Test with a sheared quadrilateral
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Sheared quadrilateral (unit square sheared in x-direction)
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 1.0, 0.0 }; // Bottom-right
  coorg(2) = { 1.5, 1.0 }; // Top-right (sheared)
  coorg(3) = { 0.5, 1.0 }; // Top-left (sheared)

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // Jacobian should still be 0.25 (area preserved under shear)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.25)))
      << expected_got(0.25, result.jacobian);

  // Check that jacobian matrix elements account for shear
  // For this shear mapping: dx/dxi = 0.5, dx/dgamma = 0.25, dz/dxi = 0,
  // dz/dgamma = 0.5 Inverse jacobian: xix = 2, gammax = 0, xiz = -1, gammaz = 2
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(-1.0)))
      << expected_got(-1.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, JacobianConsistencyAcrossElement) {
  // Test that jacobian is constant across a bilinear element (for 4-node
  // elements)
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Simple parallelogram
  coorg(0) = { 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0 };
  coorg(2) = { 3.0, 1.0 };
  coorg(3) = { 1.0, 1.0 };

  // Test jacobian at multiple points - should be constant for bilinear mapping
  auto result_center = compute_jacobian(coorg, ngnod, 0.0, 0.0);
  auto result_corner1 = compute_jacobian(coorg, ngnod, -0.5, -0.5);
  auto result_corner2 = compute_jacobian(coorg, ngnod, 0.5, 0.5);

  // Jacobian should be constant across element
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.jacobian, static_cast<type_real>(result_corner1.jacobian)))
      << expected_got(result_corner1.jacobian, result_center.jacobian);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.jacobian, static_cast<type_real>(result_corner2.jacobian)))
      << expected_got(result_corner2.jacobian, result_center.jacobian);

  // Inverse jacobian matrix elements should also be constant
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.xix, static_cast<type_real>(result_corner1.xix)))
      << expected_got(result_corner1.xix, result_center.xix);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.xix, static_cast<type_real>(result_corner2.xix)))
      << expected_got(result_corner2.xix, result_center.xix);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammax, static_cast<type_real>(result_corner1.gammax)))
      << expected_got(result_corner1.gammax, result_center.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammax, static_cast<type_real>(result_corner2.gammax)))
      << expected_got(result_corner2.gammax, result_center.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.xiz, static_cast<type_real>(result_corner1.xiz)))
      << expected_got(result_corner1.xiz, result_center.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.xiz, static_cast<type_real>(result_corner2.xiz)))
      << expected_got(result_corner2.xiz, result_center.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammaz, static_cast<type_real>(result_corner1.gammaz)))
      << expected_got(result_corner1.gammaz, result_center.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammaz, static_cast<type_real>(result_corner2.gammaz)))
      << expected_got(result_corner2.gammaz, result_center.gammaz);
}

TEST_F(ComputeJacobianDim2Test, RectangleMapping) {
  // Test with a rectangle (different scaling in each direction)
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Rectangle: width = 4, height = 2
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 4.0, 0.0 }; // Bottom-right
  coorg(2) = { 4.0, 2.0 }; // Top-right
  coorg(3) = { 0.0, 2.0 }; // Top-left

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // For 4x2 rectangle, jacobian should be 2.0 (area = 8, reference area = 4)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(2.0)))
      << expected_got(2.0, result.jacobian);

  // For 4x2 rectangle mapping:
  // dx/dxi = 2, dx/dgamma = 0, dz/dxi = 0, dz/dgamma = 1
  // Inverse jacobian: xi_x = 0.5, gamma_x = 0, xi_z = 0, gamma_z = 1
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, PositiveJacobianCheck) {
  // Test that jacobian is positive for properly oriented elements
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Various properly oriented quadrilaterals
  std::vector<std::array<std::array<type_real, 2>, 4> > test_elements = {
    // Unit square
    { { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 } } },
    // Trapezoid
    { { { 0.0, 0.0 }, { 2.0, 0.0 }, { 1.5, 1.0 }, { 0.5, 1.0 } } },
    // Rotated square
    { { { 1.0, 0.0 }, { 0.0, 1.0 }, { -1.0, 0.0 }, { 0.0, -1.0 } } }
  };

  for (const auto &element : test_elements) {
    for (int i = 0; i < 4; ++i) {
      coorg(i) = { element[i][0], element[i][1] };
    }

    auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);
    EXPECT_GT(result.jacobian, 0.0)
        << "Jacobian should be positive for properly oriented element";
  }
}

TEST_F(ComputeJacobianDim2Test, NegativeJacobianCheck) {
  // Test that jacobian is negative for improperly oriented elements
  const int ngnod = 4;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Clockwise oriented square (improper orientation)
  coorg(0) = { 0.0, 0.0 }; // Bottom-left
  coorg(1) = { 0.0, 1.0 }; // Top-left (swapped)
  coorg(2) = { 1.0, 1.0 }; // Top-right
  coorg(3) = { 1.0, 0.0 }; // Bottom-right (swapped)

  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);
  EXPECT_LT(result.jacobian, 0.0)
      << "Jacobian should be negative for improperly oriented element";
}

TEST_F(ComputeJacobianDim2Test, OverloadedFunction) {
  // Test the overloaded function that takes explicit derivatives
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

  // Create dummy derivative matrix for 4-node element at center
  std::vector<std::vector<type_real> > dershape2D(2, std::vector<type_real>(4));
  // At center (0,0), derivatives for bilinear element:
  dershape2D[0][0] = -0.25;
  dershape2D[1][0] = -0.25; // node 0
  dershape2D[0][1] = 0.25;
  dershape2D[1][1] = -0.25; // node 1
  dershape2D[0][2] = 0.25;
  dershape2D[1][2] = 0.25; // node 2
  dershape2D[0][3] = -0.25;
  dershape2D[1][3] = 0.25; // node 3

  auto result_overloaded = compute_jacobian(coorg, ngnod, dershape2D);
  auto result_direct = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // Results should match
  EXPECT_TRUE(specfem::utilities::is_close(
      result_overloaded.jacobian,
      static_cast<type_real>(result_direct.jacobian)))
      << expected_got(result_direct.jacobian, result_overloaded.jacobian);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_overloaded.xix, static_cast<type_real>(result_direct.xix)))
      << expected_got(result_direct.xix, result_overloaded.xix);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_overloaded.gammax, static_cast<type_real>(result_direct.gammax)))
      << expected_got(result_direct.gammax, result_overloaded.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_overloaded.xiz, static_cast<type_real>(result_direct.xiz)))
      << expected_got(result_direct.xiz, result_overloaded.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_overloaded.gammaz, static_cast<type_real>(result_direct.gammaz)))
      << expected_got(result_direct.gammaz, result_overloaded.gammaz);
}

TEST_F(ComputeJacobianDim2Test, NineNodeUnitSquareMapping) {
  // Test with a 9-node unit square element
  const int ngnod = 9;
  auto coorg = create_unit_square_9node();

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // For unit square, jacobian should be 0.25 (area = 1, reference area = 4)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.25)))
      << expected_got(0.25, result.jacobian);

  // For unit square mapping, the inverse jacobian matrix elements should be:
  // Each direction scaled by factor of 2
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim2Test, NineNodeScaledSquareMapping) {
  // Test with a 9-node scaled square element
  const int ngnod = 9;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // 9-node square with side length 2
  coorg(0) = { 0.0, 0.0 }; // Corner 0
  coorg(1) = { 2.0, 0.0 }; // Corner 1
  coorg(2) = { 2.0, 2.0 }; // Corner 2
  coorg(3) = { 0.0, 2.0 }; // Corner 3
  coorg(4) = { 1.0, 0.0 }; // Mid-edge 4
  coorg(5) = { 2.0, 1.0 }; // Mid-edge 5
  coorg(6) = { 1.0, 2.0 }; // Mid-edge 6
  coorg(7) = { 0.0, 1.0 }; // Mid-edge 7
  coorg(8) = { 1.0, 1.0 }; // Center 8

  // Test at center point (0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0);

  // For 2x2 square, jacobian should be 1.0 (area = 4, reference area = 4)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(1.0)))
      << expected_got(1.0, result.jacobian);

  // For 2x2 square mapping, each direction scaled by factor of 1
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.gammaz);
}
