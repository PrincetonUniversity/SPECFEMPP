// core/specfem/point/test_partial_derivatives.hpp

#include "enumerations/interface.hpp"
#include "specfem/point/partial_derivatives.hpp"
#include "specfem_setup.hpp"
#include "test_setup.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Test fixture for Kokkos
class PointPartialDerivativesTest : public ::testing::Test {
protected:
  void SetUp() override { Kokkos::initialize(); }
  void TearDown() override { Kokkos::finalize(); }
};

// ===============================
// 2D, no Jacobian
// ===============================
TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_DefaultConstructor) {
  point::partial_derivatives<dimension::type::dim2, false, false> pd;
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_ValueConstructor) {
  point::partial_derivatives<dimension::type::dim2, false, false> pd(1.1, 2.2,
                                                                     3.3, 4.4);
  EXPECT_REAL_EQ(pd.xix, 1.1);
  EXPECT_REAL_EQ(pd.gammax, 2.2);
  EXPECT_REAL_EQ(pd.xiz, 3.3);
  EXPECT_REAL_EQ(pd.gammaz, 4.4);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_ConstantConstructor) {
  point::partial_derivatives<dimension::type::dim2, false, false> pd(7.7);
  EXPECT_REAL_EQ(pd.xix, 7.7);
  EXPECT_REAL_EQ(pd.gammax, 7.7);
  EXPECT_REAL_EQ(pd.xiz, 7.7);
  EXPECT_REAL_EQ(pd.gammaz, 7.7);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_Init) {
  point::partial_derivatives<dimension::type::dim2, false, false> pd(1, 2, 3,
                                                                     4);
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_Arithmetic) {
  using PD = point::partial_derivatives<dimension::type::dim2, false, false>;
  PD a(1, 2, 3, 4), b(10, 20, 30, 40);
  PD c = a + b;
  EXPECT_REAL_EQ(c.xix, 11);
  EXPECT_REAL_EQ(c.gammax, 22);
  EXPECT_REAL_EQ(c.xiz, 33);
  EXPECT_REAL_EQ(c.gammaz, 44);

  a += b;
  EXPECT_REAL_EQ(a.xix, 11);
  EXPECT_REAL_EQ(a.gammax, 22);
  EXPECT_REAL_EQ(a.xiz, 33);
  EXPECT_REAL_EQ(a.gammaz, 44);

  PD d = b * 2.0;
  EXPECT_REAL_EQ(d.xix, 20);
  EXPECT_REAL_EQ(d.gammax, 40);
  EXPECT_REAL_EQ(d.xiz, 60);
  EXPECT_REAL_EQ(d.gammaz, 80);

  PD e = 3.0 * b;
  EXPECT_REAL_EQ(e.xix, 30);
  EXPECT_REAL_EQ(e.gammax, 60);
  EXPECT_REAL_EQ(e.xiz, 90);
  EXPECT_REAL_EQ(e.gammaz, 120);
}

// ===============================
// 3D, no Jacobian
// ===============================
TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_DefaultConstructor) {
  point::partial_derivatives<dimension::type::dim3, false, false> pd;
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiy, 0.0);
  EXPECT_REAL_EQ(pd.gammay, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_ValueConstructor) {
  point::partial_derivatives<dimension::type::dim3, false, false> pd(1, 2, 3, 4,
                                                                     5, 6);
  EXPECT_REAL_EQ(pd.xix, 1);
  EXPECT_REAL_EQ(pd.gammax, 2);
  EXPECT_REAL_EQ(pd.xiy, 3);
  EXPECT_REAL_EQ(pd.gammay, 4);
  EXPECT_REAL_EQ(pd.xiz, 5);
  EXPECT_REAL_EQ(pd.gammaz, 6);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_ConstantConstructor) {
  point::partial_derivatives<dimension::type::dim3, false, false> pd(9.9);
  EXPECT_REAL_EQ(pd.xix, 9.9);
  EXPECT_REAL_EQ(pd.gammax, 9.9);
  EXPECT_REAL_EQ(pd.xiy, 9.9);
  EXPECT_REAL_EQ(pd.gammay, 9.9);
  EXPECT_REAL_EQ(pd.xiz, 9.9);
  EXPECT_REAL_EQ(pd.gammaz, 9.9);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_Init) {
  point::partial_derivatives<dimension::type::dim3, false, false> pd(1, 2, 3, 4,
                                                                     5, 6);
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiy, 0.0);
  EXPECT_REAL_EQ(pd.gammay, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_Arithmetic) {
  using PD = point::partial_derivatives<dimension::type::dim3, false, false>;
  PD a(1, 2, 3, 4, 5, 6), b(10, 20, 30, 40, 50, 60);
  PD c = a + b;
  EXPECT_REAL_EQ(c.xix, 11);
  EXPECT_REAL_EQ(c.gammax, 22);
  EXPECT_REAL_EQ(c.xiy, 33);
  EXPECT_REAL_EQ(c.gammay, 44);
  EXPECT_REAL_EQ(c.xiz, 55);
  EXPECT_REAL_EQ(c.gammaz, 66);

  a += b;
  EXPECT_REAL_EQ(a.xix, 11);
  EXPECT_REAL_EQ(a.gammax, 22);
  EXPECT_REAL_EQ(a.xiy, 33);
  EXPECT_REAL_EQ(a.gammay, 44);
  EXPECT_REAL_EQ(a.xiz, 55);
  EXPECT_REAL_EQ(a.gammaz, 66);

  PD d = b * 2.0;
  EXPECT_REAL_EQ(d.xix, 20);
  EXPECT_REAL_EQ(d.gammax, 40);
  EXPECT_REAL_EQ(d.xiy, 60);
  EXPECT_REAL_EQ(d.gammay, 80);
  EXPECT_REAL_EQ(d.xiz, 100);
  EXPECT_REAL_EQ(d.gammaz, 120);

  PD e = 3.0 * b;
  EXPECT_REAL_EQ(e.xix, 30);
  EXPECT_REAL_EQ(e.gammax, 60);
  EXPECT_REAL_EQ(e.xiy, 90);
  EXPECT_REAL_EQ(e.gammay, 120);
  EXPECT_REAL_EQ(e.xiz, 150);
  EXPECT_REAL_EQ(e.gammaz, 180);
}

// ===============================
// 2D, with Jacobian
// ===============================
TEST_F(PointPartialDerivativesTest,
       PartialDerivatives2D_WithJacobian_Constructors) {
  using PD = point::partial_derivatives<dimension::type::dim2, true, false>;
  PD pd1;
  pd1.init();
  EXPECT_REAL_EQ(pd1.xix, 0.0);
  EXPECT_REAL_EQ(pd1.gammax, 0.0);
  EXPECT_REAL_EQ(pd1.xiz, 0.0);
  EXPECT_REAL_EQ(pd1.gammaz, 0.0);
  EXPECT_REAL_EQ(pd1.jacobian, 0.0);

  PD pd2(1, 2, 3, 4, 5);
  EXPECT_REAL_EQ(pd2.xix, 1);
  EXPECT_REAL_EQ(pd2.gammax, 2);
  EXPECT_REAL_EQ(pd2.xiz, 3);
  EXPECT_REAL_EQ(pd2.gammaz, 4);
  EXPECT_REAL_EQ(pd2.jacobian, 5);

  PD pd3(7.7);
  EXPECT_REAL_EQ(pd3.xix, 7.7);
  EXPECT_REAL_EQ(pd3.gammax, 7.7);
  EXPECT_REAL_EQ(pd3.xiz, 7.7);
  EXPECT_REAL_EQ(pd3.gammaz, 7.7);
  EXPECT_REAL_EQ(pd3.jacobian, 7.7);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_WithJacobian_Init) {
  using PD = point::partial_derivatives<dimension::type::dim2, true, false>;
  PD pd(1, 2, 3, 4, 5);
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
  EXPECT_REAL_EQ(pd.jacobian, 0.0);
}

TEST_F(PointPartialDerivativesTest,
       PartialDerivatives2D_WithJacobian_ComputeNormal) {
  using PD = point::partial_derivatives<dimension::type::dim2, true, false>;
  PD pd(1, 2, 3, 4, 5);
  auto n_bottom = pd.compute_normal(enums::edge::type::BOTTOM);
  EXPECT_REAL_EQ(n_bottom(0), -2.0 * 5.0);
  EXPECT_REAL_EQ(n_bottom(1), -4.0 * 5.0);

  auto n_top = pd.compute_normal(enums::edge::type::TOP);
  EXPECT_REAL_EQ(n_top(0), 2.0 * 5.0);
  EXPECT_REAL_EQ(n_top(1), 4.0 * 5.0);

  auto n_left = pd.compute_normal(enums::edge::type::LEFT);
  EXPECT_REAL_EQ(n_left(0), -1.0 * 5.0);
  EXPECT_REAL_EQ(n_left(1), -3.0 * 5.0);

  auto n_right = pd.compute_normal(enums::edge::type::RIGHT);
  EXPECT_REAL_EQ(n_right(0), 1.0 * 5.0);
  EXPECT_REAL_EQ(n_right(1), 3.0 * 5.0);
}

// ===============================
// 3D, with Jacobian
// ===============================
TEST_F(PointPartialDerivativesTest,
       PartialDerivatives3D_WithJacobian_Constructors) {
  using PD = point::partial_derivatives<dimension::type::dim3, true, false>;
  PD pd1;
  pd1.init();
  EXPECT_REAL_EQ(pd1.xix, 0.0);
  EXPECT_REAL_EQ(pd1.gammax, 0.0);
  EXPECT_REAL_EQ(pd1.xiy, 0.0);
  EXPECT_REAL_EQ(pd1.gammay, 0.0);
  EXPECT_REAL_EQ(pd1.xiz, 0.0);
  EXPECT_REAL_EQ(pd1.gammaz, 0.0);
  EXPECT_REAL_EQ(pd1.jacobian, 0.0);

  PD pd2(1, 2, 3, 4, 5, 6, 7);
  EXPECT_REAL_EQ(pd2.xix, 1);
  EXPECT_REAL_EQ(pd2.gammax, 2);
  EXPECT_REAL_EQ(pd2.xiy, 3);
  EXPECT_REAL_EQ(pd2.gammay, 4);
  EXPECT_REAL_EQ(pd2.xiz, 5);
  EXPECT_REAL_EQ(pd2.gammaz, 6);
  EXPECT_REAL_EQ(pd2.jacobian, 7);

  PD pd3(8.8);
  EXPECT_REAL_EQ(pd3.xix, 8.8);
  EXPECT_REAL_EQ(pd3.gammax, 8.8);
  EXPECT_REAL_EQ(pd3.xiy, 8.8);
  EXPECT_REAL_EQ(pd3.gammay, 8.8);
  EXPECT_REAL_EQ(pd3.xiz, 8.8);
  EXPECT_REAL_EQ(pd3.gammaz, 8.8);
  EXPECT_REAL_EQ(pd3.jacobian, 8.8);
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_WithJacobian_Init) {
  using PD = point::partial_derivatives<dimension::type::dim3, true, false>;
  PD pd(1, 2, 3, 4, 5, 6, 7);
  pd.init();
  EXPECT_REAL_EQ(pd.xix, 0.0);
  EXPECT_REAL_EQ(pd.gammax, 0.0);
  EXPECT_REAL_EQ(pd.xiy, 0.0);
  EXPECT_REAL_EQ(pd.gammay, 0.0);
  EXPECT_REAL_EQ(pd.xiz, 0.0);
  EXPECT_REAL_EQ(pd.gammaz, 0.0);
  EXPECT_REAL_EQ(pd.jacobian, 0.0);
}

// ===============================
// SIMD instantiation (compile only)
// ===============================
TEST_F(PointPartialDerivativesTest, PartialDerivatives2D_SIMD_Compile) {
  using PD = point::partial_derivatives<dimension::type::dim2, false, true>;
  PD pd;
  (void)pd;
}

TEST_F(PointPartialDerivativesTest, PartialDerivatives3D_SIMD_Compile) {
  using PD = point::partial_derivatives<dimension::type::dim3, false, true>;
  PD pd;
  (void)pd;
}

TEST_F(PointPartialDerivativesTest,
       PartialDerivatives2D_WithJacobian_SIMD_Compile) {
  using PD = point::partial_derivatives<dimension::type::dim2, true, true>;
  PD pd;
  (void)pd;
}

TEST_F(PointPartialDerivativesTest,
       PartialDerivatives3D_WithJacobian_SIMD_Compile) {
  using PD = point::partial_derivatives<dimension::type::dim3, true, true>;
  PD pd;
  (void)pd;
}

// ===============================
// Main
// ===============================
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
