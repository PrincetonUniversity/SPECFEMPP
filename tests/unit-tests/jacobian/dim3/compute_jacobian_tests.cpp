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

using specfem::jacobian::compute_jacobian;

class ComputeJacobianDim3Test : public ::testing::Test {
protected:
  void SetUp() override {
    // Common test setup can go here
  }

  void TearDown() override {
    // Common test cleanup can go here
  }
};

TEST_F(ComputeJacobianDim3Test, UnitCubeIdentityMapping) {
  // Test with a unit cube element using 8 corner nodes
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube control nodes (corners)
  coorg(0) = { 0.0, 0.0, 0.0 }; // Corner 0: (0,0,0)
  coorg(1) = { 1.0, 0.0, 0.0 }; // Corner 1: (1,0,0)
  coorg(2) = { 1.0, 1.0, 0.0 }; // Corner 2: (1,1,0)
  coorg(3) = { 0.0, 1.0, 0.0 }; // Corner 3: (0,1,0)
  coorg(4) = { 0.0, 0.0, 1.0 }; // Corner 4: (0,0,1)
  coorg(5) = { 1.0, 0.0, 1.0 }; // Corner 5: (1,0,1)
  coorg(6) = { 1.0, 1.0, 1.0 }; // Corner 6: (1,1,1)
  coorg(7) = { 0.0, 1.0, 1.0 }; // Corner 7: (0,1,1)

  // Test at center point (0, 0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // For unit cube, jacobian should be 0.125 (volume = 1, reference volume = 8)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.125)))
      << expected_got(0.125, result.jacobian);

  // For unit cube mapping, the inverse jacobian matrix elements should be:
  // Each direction scaled by factor of 2
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiy, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiy);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etay, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.etay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammay, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etaz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etaz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim3Test, ScaledCubeMapping) {
  // Test with a scaled cube (side length 2)
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Cube with side length 2
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0, 0.0 };
  coorg(2) = { 2.0, 2.0, 0.0 };
  coorg(3) = { 0.0, 2.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 2.0 };
  coorg(5) = { 2.0, 0.0, 2.0 };
  coorg(6) = { 2.0, 2.0, 2.0 };
  coorg(7) = { 0.0, 2.0, 2.0 };

  // Test at center point (0, 0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // For 2x2x2 cube, jacobian should be 1.0 (volume = 8, reference volume = 8)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(1.0)))
      << expected_got(1.0, result.jacobian);

  // For 2x2x2 cube mapping, each direction scaled by factor of 1
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiy, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiy);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etay, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.etay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammay, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etaz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etaz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.gammaz);
}

TEST_F(ComputeJacobianDim3Test, TranslatedCubeMapping) {
  // Test with a translated cube (translation shouldn't affect jacobian)
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Unit cube translated by (5, 3, 7)
  coorg(0) = { 5.0, 3.0, 7.0 };
  coorg(1) = { 6.0, 3.0, 7.0 };
  coorg(2) = { 6.0, 4.0, 7.0 };
  coorg(3) = { 5.0, 4.0, 7.0 };
  coorg(4) = { 5.0, 3.0, 8.0 };
  coorg(5) = { 6.0, 3.0, 8.0 };
  coorg(6) = { 6.0, 4.0, 8.0 };
  coorg(7) = { 5.0, 4.0, 8.0 };

  // Test at center point (0, 0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // Translation shouldn't change jacobian values
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.125)))
      << expected_got(0.125, result.jacobian);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etay, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.etay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
  // Off-diagonal terms should be zero
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiy, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiy);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammay, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etaz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etaz);
}

TEST_F(ComputeJacobianDim3Test, RectangularPrismMapping) {
  // Test with a rectangular prism (different scaling in each direction)
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Rectangular prism: width = 4, height = 2, depth = 3
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 4.0, 0.0, 0.0 };
  coorg(2) = { 4.0, 2.0, 0.0 };
  coorg(3) = { 0.0, 2.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 3.0 };
  coorg(5) = { 4.0, 0.0, 3.0 };
  coorg(6) = { 4.0, 2.0, 3.0 };
  coorg(7) = { 0.0, 2.0, 3.0 };

  // Test at center point (0, 0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // For 4x2x3 prism, jacobian should be 3.0 (volume = 24, reference volume = 8)
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(3.0)))
      << expected_got(3.0, result.jacobian);

  // For 4x2x3 prism mapping:
  // dx/dxi = 2, dy/deta = 1, dz/dgamma = 1.5
  // Inverse jacobian: xi_x = 0.5, eta_y = 1.0, gamma_z = 2/3
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(0.5)))
      << expected_got(0.5, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etay, static_cast<type_real>(1.0)))
      << expected_got(1.0, result.etay);
  EXPECT_TRUE(specfem::utilities::is_close(result.gammaz,
                                           static_cast<type_real>(2.0 / 3.0)))
      << expected_got(2.0 / 3.0, result.gammaz);
  // Off-diagonal terms should be zero
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiy, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiy);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammay, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etaz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etaz);
}

TEST_F(ComputeJacobianDim3Test, ShearMapping) {
  // Test with a sheared hexahedral element
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Sheared hexahedron (unit cube sheared in x-direction based on z)
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 1.0, 0.0, 0.0 };
  coorg(2) = { 1.0, 1.0, 0.0 };
  coorg(3) = { 0.0, 1.0, 0.0 };
  coorg(4) = { 0.5, 0.0, 1.0 }; // Sheared
  coorg(5) = { 1.5, 0.0, 1.0 }; // Sheared
  coorg(6) = { 1.5, 1.0, 1.0 }; // Sheared
  coorg(7) = { 0.5, 1.0, 1.0 }; // Sheared

  // Test at center point (0, 0, 0) in reference coordinates
  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // Volume should be preserved under shear, so jacobian should be 0.125
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(0.125)))
      << expected_got(0.125, result.jacobian);

  // Check that jacobian matrix elements account for shear
  // For this shear mapping: dx/dxi = 0.5, dx/deta = 0, dx/dgamma = 0.25
  //                        dy/dxi = 0,   dy/deta = 0.5, dy/dgamma = 0
  //                        dz/dxi = 0,   dz/deta = 0,   dz/dgamma = 0.5
  // The shear introduces coupling in the xiz term (∂ξ/∂z)
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xix, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammax, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammax);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiy, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.xiy);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etay, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.etay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammay, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.gammay);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.xiz, static_cast<type_real>(-1.0)))
      << expected_got(-1.0, result.xiz); // Shear coupling
  EXPECT_TRUE(
      specfem::utilities::is_close(result.etaz, static_cast<type_real>(0.0)))
      << expected_got(0.0, result.etaz);
  EXPECT_TRUE(
      specfem::utilities::is_close(result.gammaz, static_cast<type_real>(2.0)))
      << expected_got(2.0, result.gammaz);
}

TEST_F(ComputeJacobianDim3Test, JacobianConsistencyAcrossElement) {
  // Test that jacobian is constant across a trilinear element (for 8-node
  // elements)
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Axis-aligned rectangular prism (trilinear mapping has constant jacobian)
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 2.0, 0.0, 0.0 };
  coorg(2) = { 2.0, 1.0, 0.0 };
  coorg(3) = { 0.0, 1.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 3.0 };
  coorg(5) = { 2.0, 0.0, 3.0 };
  coorg(6) = { 2.0, 1.0, 3.0 };
  coorg(7) = { 0.0, 1.0, 3.0 };

  // Test jacobian at multiple points - should be constant for trilinear mapping
  auto result_center = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);
  auto result_corner1 = compute_jacobian(coorg, ngnod, -0.5, -0.5, -0.5);
  auto result_corner2 = compute_jacobian(coorg, ngnod, 0.5, 0.5, 0.5);

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
      result_center.etax, static_cast<type_real>(result_corner1.etax)))
      << expected_got(result_corner1.etax, result_center.etax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.etax, static_cast<type_real>(result_corner2.etax)))
      << expected_got(result_corner2.etax, result_center.etax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammax, static_cast<type_real>(result_corner1.gammax)))
      << expected_got(result_corner1.gammax, result_center.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(
      result_center.gammax, static_cast<type_real>(result_corner2.gammax)))
      << expected_got(result_corner2.gammax, result_center.gammax);
}

TEST_F(ComputeJacobianDim3Test, PositiveJacobianCheck) {
  // Test that jacobian is positive for properly oriented elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Various properly oriented hexahedra
  std::vector<std::array<std::array<type_real, 3>, 8> > test_elements = {
    // Unit cube
    { { { 0.0, 0.0, 0.0 },
        { 1.0, 0.0, 0.0 },
        { 1.0, 1.0, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.0, 1.0 },
        { 1.0, 0.0, 1.0 },
        { 1.0, 1.0, 1.0 },
        { 0.0, 1.0, 1.0 } } },
    // Scaled cube
    { { { 0.0, 0.0, 0.0 },
        { 2.0, 0.0, 0.0 },
        { 2.0, 2.0, 0.0 },
        { 0.0, 2.0, 0.0 },
        { 0.0, 0.0, 2.0 },
        { 2.0, 0.0, 2.0 },
        { 2.0, 2.0, 2.0 },
        { 0.0, 2.0, 2.0 } } }
  };

  for (const auto &element : test_elements) {
    for (int i = 0; i < 8; ++i) {
      coorg(i) = { element[i][0], element[i][1], element[i][2] };
    }

    auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);
    EXPECT_GT(result.jacobian, 0.0)
        << "Jacobian should be positive for properly oriented element";
  }
}

TEST_F(ComputeJacobianDim3Test, NegativeJacobianCheck) {
  // Test that jacobian is negative for improperly oriented elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Improperly oriented cube (swapping nodes to reverse orientation)
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 0.0, 1.0, 0.0 }; // Swapped with node 3
  coorg(2) = { 1.0, 1.0, 0.0 };
  coorg(3) = { 1.0, 0.0, 0.0 }; // Swapped with node 1
  coorg(4) = { 0.0, 0.0, 1.0 };
  coorg(5) = { 0.0, 1.0, 1.0 }; // Swapped with node 7
  coorg(6) = { 1.0, 1.0, 1.0 };
  coorg(7) = { 1.0, 0.0, 1.0 }; // Swapped with node 5

  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);
  EXPECT_LT(result.jacobian, 0.0)
      << "Jacobian should be negative for improperly oriented element";
}

TEST_F(ComputeJacobianDim3Test, JacobianDeterminantFormula) {
  // Test that the jacobian determinant calculation is correct
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Simple rectangular prism where we can verify the determinant manually
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { 3.0, 0.0, 0.0 }; // dx/dxi = 1.5
  coorg(2) = { 3.0, 2.0, 0.0 }; // dy/deta = 1.0
  coorg(3) = { 0.0, 2.0, 0.0 };
  coorg(4) = { 0.0, 0.0, 4.0 }; // dz/dgamma = 2.0
  coorg(5) = { 3.0, 0.0, 4.0 };
  coorg(6) = { 3.0, 2.0, 4.0 };
  coorg(7) = { 0.0, 2.0, 4.0 };

  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // For this prism: jacobian determinant = 1.5 * 1.0 * 2.0 = 3.0
  EXPECT_TRUE(specfem::utilities::is_close(result.jacobian,
                                           static_cast<type_real>(3.0)))
      << expected_got(3.0, result.jacobian);

  // Check specific inverse jacobian elements
  EXPECT_TRUE(specfem::utilities::is_close(result.xix,
                                           static_cast<type_real>(1.0 / 1.5)))
      << expected_got(1.0 / 1.5, result.xix); // 1/(dx/dxi)
  EXPECT_TRUE(specfem::utilities::is_close(result.etay,
                                           static_cast<type_real>(1.0 / 1.0)))
      << expected_got(1.0 / 1.0, result.etay); // 1/(dy/deta)
  EXPECT_TRUE(specfem::utilities::is_close(result.gammaz,
                                           static_cast<type_real>(1.0 / 2.0)))
      << expected_got(1.0 / 2.0, result.gammaz); // 1/(dz/dgamma)
}

TEST_F(ComputeJacobianDim3Test, SmallElementStability) {
  // Test numerical stability for very small elements
  const int ngnod = 8;
  Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
      coorg("coorg", ngnod);

  // Very small cube (side length 1e-6)
  const type_real scale = 1e-6;
  coorg(0) = { 0.0, 0.0, 0.0 };
  coorg(1) = { scale, 0.0, 0.0 };
  coorg(2) = { scale, scale, 0.0 };
  coorg(3) = { 0.0, scale, 0.0 };
  coorg(4) = { 0.0, 0.0, scale };
  coorg(5) = { scale, 0.0, scale };
  coorg(6) = { scale, scale, scale };
  coorg(7) = { 0.0, scale, scale };

  auto result = compute_jacobian(coorg, ngnod, 0.0, 0.0, 0.0);

  // Expected jacobian for small cube
  type_real expected_jacobian = (scale * scale * scale) / 8.0;
  EXPECT_TRUE(specfem::utilities::is_close(
      result.jacobian, static_cast<type_real>(expected_jacobian)))
      << expected_got(expected_jacobian, result.jacobian);

  // Inverse jacobian elements should scale appropriately
  EXPECT_TRUE(specfem::utilities::is_close(result.xix,
                                           static_cast<type_real>(2.0 / scale)))
      << expected_got(2.0 / scale, result.xix);
  EXPECT_TRUE(specfem::utilities::is_close(result.etay,
                                           static_cast<type_real>(2.0 / scale)))
      << expected_got(2.0 / scale, result.etay);
  EXPECT_TRUE(specfem::utilities::is_close(result.gammaz,
                                           static_cast<type_real>(2.0 / scale)))
      << expected_got(2.0 / scale, result.gammaz);
}
