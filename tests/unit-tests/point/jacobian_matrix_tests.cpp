// core/specfem/point/test_jacobian_matrix.hpp

#include "enumerations/interface.hpp"
#include "specfem/datatype.hpp"
#include "specfem/point/jacobian_matrix.hpp"
#include "specfem/utilities.hpp"
#include "specfem_setup.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

// Base test fixture for Jacobian matrix tests with template parameter for
// SIMD
template <bool UseSIMD>
class PointJacobianMatrixTestUntyped : public ::testing::Test {
protected:
  // Define SIMD-related types for convenience
  using simd_type = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd_type::datatype;
  using mask_type = typename simd_type::mask_type;

  void SetUp() override {
    if (!Kokkos::is_initialized())
      Kokkos::initialize();
  }

  void TearDown() override {
    if (Kokkos::is_initialized())
      Kokkos::finalize();
  }
};

template <typename T>
class PointJacobianMatrixTest
    : public PointJacobianMatrixTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointJacobianMatrixTest, TestTypes);

// ===============================
// 2D, no Jacobian
// ===============================
TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_DefaultConstructor) {
  constexpr bool using_simd = TypeParam::value;

  using pd_type =
      point::jacobian_matrix<dimension::type::dim2, false, using_simd>;
  pd_type pd;
  typename pd_type::value_type zero_val{ 0.0 };
  pd.init();

  // Use is_close to check values for both SIMD and non-SIMD cases
  EXPECT_TRUE(specfem::utilities::is_close(pd.xix, zero_val))
      << ExpectedGot(0.0, pd.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammax, zero_val))
      << ExpectedGot(0.0, pd.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd.xiz, zero_val))
      << ExpectedGot(0.0, pd.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammaz, zero_val))
      << ExpectedGot(0.0, pd.gammaz);
}

TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_ValueConstructor) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype xix_val{
    1.1
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype gammax_val{
    2.2
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype xiz_val{
    3.3
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype gammaz_val{
    4.4
  };

  point::jacobian_matrix<dimension::type::dim2, false, using_simd> pd(
      xix_val, gammax_val, xiz_val, gammaz_val);

  // Check values using is_close
  EXPECT_TRUE(specfem::utilities::is_close(pd.xix, xix_val))
      << ExpectedGot(xix_val, pd.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammax, gammax_val))
      << ExpectedGot(gammax_val, pd.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd.xiz, xiz_val))
      << ExpectedGot(xiz_val, pd.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammaz, gammaz_val))
      << ExpectedGot(gammaz_val, pd.gammaz);
}

TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_ConstantConstructor) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Value to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype const_val{
    7.7
  };

  point::jacobian_matrix<dimension::type::dim2, false, using_simd> pd(
      const_val);

  // Check values using is_close
  EXPECT_TRUE(specfem::utilities::is_close(pd.xix, const_val))
      << ExpectedGot(const_val, pd.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammax, const_val))
      << ExpectedGot(const_val, pd.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd.xiz, const_val))
      << ExpectedGot(const_val, pd.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammaz, const_val))
      << ExpectedGot(const_val, pd.gammaz);
}

TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_Init) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype xix_val{
    1.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype gammax_val{
    2.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype xiz_val{
    3.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype gammaz_val{
    4.0
  };

  using pd_type =
      point::jacobian_matrix<dimension::type::dim2, false, using_simd>;
  pd_type pd(xix_val, gammax_val, xiz_val, gammaz_val);
  typename pd_type::value_type zero_val{ 0.0 };
  pd.init();

  // Check values after init
  EXPECT_TRUE(specfem::utilities::is_close(pd.xix, zero_val))
      << ExpectedGot(zero_val, pd.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammax, zero_val))
      << ExpectedGot(zero_val, pd.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd.xiz, zero_val))
      << ExpectedGot(zero_val, pd.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammaz, zero_val))
      << ExpectedGot(0.0, pd.gammaz);
}

TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_Arithmetic) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use for constructors
  typename specfem::datatype::simd<type_real, using_simd>::datatype a_xix_val{
    1.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype
      a_gammax_val{ 2.0 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype a_xiz_val{
    3.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype
      a_gammaz_val{ 4.0 };

  typename specfem::datatype::simd<type_real, using_simd>::datatype b_xix_val{
    10.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype
      b_gammax_val{ 20.0 };
  typename specfem::datatype::simd<type_real, using_simd>::datatype b_xiz_val{
    30.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype
      b_gammaz_val{ 40.0 };

  using PD = point::jacobian_matrix<dimension::type::dim2, false, using_simd>;
  PD a(a_xix_val, a_gammax_val, a_xiz_val, a_gammaz_val);
  PD b(b_xix_val, b_gammax_val, b_xiz_val, b_gammaz_val);

  // Addition
  PD c = a + b;
  EXPECT_TRUE(specfem::utilities::is_close(c.xix, a_xix_val + b_xix_val))
      << ExpectedGot(a_xix_val + b_xix_val, c.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(c.gammax, a_gammax_val + b_gammax_val))
      << ExpectedGot(a_gammax_val + b_gammax_val, c.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(c.xiz, a_xiz_val + b_xiz_val))
      << ExpectedGot(a_xiz_val + b_xiz_val, c.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(c.gammaz, a_gammaz_val + b_gammaz_val))
      << ExpectedGot(a_gammaz_val + b_gammaz_val, c.gammaz);

  // Addition assignment
  a += b;
  EXPECT_TRUE(specfem::utilities::is_close(a.xix, a_xix_val + b_xix_val))
      << ExpectedGot(a_xix_val + b_xix_val, a.xix);
  EXPECT_TRUE(
      specfem::utilities::is_close(a.gammax, a_gammax_val + b_gammax_val))
      << ExpectedGot(a_gammax_val + b_gammax_val, a.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(a.xiz, a_xiz_val + b_xiz_val))
      << ExpectedGot(a_xiz_val + b_xiz_val, a.xiz);
  EXPECT_TRUE(
      specfem::utilities::is_close(a.gammaz, a_gammaz_val + b_gammaz_val))
      << ExpectedGot(a_gammaz_val + b_gammaz_val, a.gammaz);

  // Scalar multiplication (object * scalar)
  if constexpr (!using_simd) {
    type_real scalar_2 = 2.0;

    PD d = b * scalar_2;
    EXPECT_TRUE(specfem::utilities::is_close(d.xix, b_xix_val * scalar_2))
        << ExpectedGot(b_xix_val * scalar_2, d.xix);
    EXPECT_TRUE(specfem::utilities::is_close(d.gammax, b_gammax_val * scalar_2))
        << ExpectedGot(b_gammax_val * scalar_2, d.gammax);
    EXPECT_TRUE(specfem::utilities::is_close(d.xiz, b_xiz_val * scalar_2))
        << ExpectedGot(b_xiz_val * scalar_2, d.xiz);
    EXPECT_TRUE(specfem::utilities::is_close(d.gammaz, b_gammaz_val * scalar_2))
        << ExpectedGot(b_gammaz_val * scalar_2, d.gammaz);

    // Scalar multiplication (scalar * object)
    typename specfem::datatype::simd<type_real, using_simd>::datatype scalar_3{
      3.0
    };

    PD e = scalar_3 * b;
    EXPECT_TRUE(specfem::utilities::is_close(e.xix, scalar_3 * b_xix_val))
        << ExpectedGot(scalar_3 * b_xix_val, e.xix);
    EXPECT_TRUE(specfem::utilities::is_close(e.gammax, scalar_3 * b_gammax_val))
        << ExpectedGot(scalar_3 * b_gammax_val, e.gammax);
    EXPECT_TRUE(specfem::utilities::is_close(e.xiz, scalar_3 * b_xiz_val))
        << ExpectedGot(scalar_3 * b_xiz_val, e.xiz);
    EXPECT_TRUE(specfem::utilities::is_close(e.gammaz, scalar_3 * b_gammaz_val))
        << ExpectedGot(scalar_3 * b_gammaz_val, e.gammaz);
  }
}

// ===============================
// 2D, with Jacobian
// ===============================
TYPED_TEST(PointJacobianMatrixTest,
           JacobianMatrix2D_WithJacobian_Constructors) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype zero_val{
    0.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype one_val{
    1.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype two_val{
    2.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype three_val{
    3.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype four_val{
    4.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype five_val{
    5.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype const_val{
    7.7
  };

  using PD = point::jacobian_matrix<dimension::type::dim2, true, using_simd>;

  // Default constructor and init
  PD pd1;
  pd1.init();

  EXPECT_TRUE(specfem::utilities::is_close(pd1.xix, zero_val))
      << ExpectedGot(zero_val, pd1.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.gammax, zero_val))
      << ExpectedGot(zero_val, pd1.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.xiz, zero_val))
      << ExpectedGot(zero_val, pd1.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.gammaz, zero_val))
      << ExpectedGot(zero_val, pd1.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.jacobian, zero_val))
      << ExpectedGot(zero_val, pd1.jacobian);

  // Value constructor
  PD pd2(one_val, two_val, three_val, four_val, five_val);

  EXPECT_TRUE(specfem::utilities::is_close(pd2.xix, one_val))
      << ExpectedGot(one_val, pd2.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammax, two_val))
      << ExpectedGot(two_val, pd2.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.xiz, three_val))
      << ExpectedGot(three_val, pd2.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammaz, four_val))
      << ExpectedGot(four_val, pd2.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.jacobian, five_val))
      << ExpectedGot(five_val, pd2.jacobian);

  // Constant constructor
  PD pd3(const_val);

  EXPECT_TRUE(specfem::utilities::is_close(pd3.xix, const_val))
      << ExpectedGot(const_val, pd3.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.gammax, const_val))
      << ExpectedGot(const_val, pd3.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.xiz, const_val))
      << ExpectedGot(const_val, pd3.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.gammaz, const_val))
      << ExpectedGot(const_val, pd3.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.jacobian, const_val))
      << ExpectedGot(const_val, pd3.jacobian);
}

TYPED_TEST(PointJacobianMatrixTest, JacobianMatrix2D_WithJacobian_Init) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype zero_val{
    0.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype one_val{
    1.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype two_val{
    2.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype three_val{
    3.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype four_val{
    4.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype five_val{
    5.0
  };

  using PD = point::jacobian_matrix<dimension::type::dim2, true, using_simd>;
  PD pd(one_val, two_val, three_val, four_val, five_val);
  pd.init();

  EXPECT_TRUE(specfem::utilities::is_close(pd.xix, zero_val))
      << ExpectedGot(zero_val, pd.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammax, zero_val))
      << ExpectedGot(zero_val, pd.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd.xiz, zero_val))
      << ExpectedGot(zero_val, pd.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.gammaz, zero_val))
      << ExpectedGot(zero_val, pd.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd.jacobian, zero_val))
      << ExpectedGot(zero_val, pd.jacobian);
}

// ===============================
// 3D, with Jacobian
// ===============================
TYPED_TEST(PointJacobianMatrixTest,
           JacobianMatrix3D_WithJacobian_Constructors) {
  constexpr bool using_simd = TypeParam::value;

  // Get the SIMD size
  constexpr int simd_size =
      specfem::datatype::simd<type_real, using_simd>::size();

  // Values to use in constructor
  typename specfem::datatype::simd<type_real, using_simd>::datatype zero_val{
    0.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype one_val{
    1.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype two_val{
    2.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype three_val{
    3.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype four_val{
    4.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype five_val{
    5.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype six_val{
    6.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype seven_val{
    7.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype eight_val{
    8.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype nine_val{
    9.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype ten_val{
    10.0
  };
  typename specfem::datatype::simd<type_real, using_simd>::datatype const_val{
    8.8
  };

  using PD = point::jacobian_matrix<dimension::type::dim3, true, using_simd>;

  // Default constructor and init
  PD pd1;
  pd1.init();

  EXPECT_TRUE(specfem::utilities::is_close(pd1.xix, zero_val))
      << ExpectedGot(zero_val, pd1.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.etax, zero_val))
      << ExpectedGot(zero_val, pd1.etax);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.gammax, zero_val))
      << ExpectedGot(zero_val, pd1.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.xiy, zero_val))
      << ExpectedGot(zero_val, pd1.xiy);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.etay, zero_val))
      << ExpectedGot(zero_val, pd1.etay);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.gammay, zero_val))
      << ExpectedGot(zero_val, pd1.gammay);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.xiz, zero_val))
      << ExpectedGot(zero_val, pd1.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.etaz, zero_val))
      << ExpectedGot(zero_val, pd1.etaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.gammaz, zero_val))
      << ExpectedGot(zero_val, pd1.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd1.jacobian, zero_val))
      << ExpectedGot(zero_val, pd1.jacobian);

  // Value constructor
  PD pd2(one_val, two_val, three_val, four_val, five_val, six_val, seven_val,
         eight_val, nine_val, ten_val);

  EXPECT_TRUE(specfem::utilities::is_close(pd2.xix, one_val))
      << ExpectedGot(one_val, pd2.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etax, two_val))
      << ExpectedGot(two_val, pd2.etax);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammax, three_val))
      << ExpectedGot(three_val, pd2.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.xiy, four_val))
      << ExpectedGot(four_val, pd2.xiy);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etay, five_val))
      << ExpectedGot(five_val, pd2.etay);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammay, six_val))
      << ExpectedGot(six_val, pd2.gammay);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.xiz, seven_val))
      << ExpectedGot(seven_val, pd2.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etaz, eight_val))
      << ExpectedGot(eight_val, pd2.etaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammaz, nine_val))
      << ExpectedGot(nine_val, pd2.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.jacobian, ten_val))
      << ExpectedGot(ten_val, pd2.jacobian);

  // Constant constructor
  PD pd3(const_val);

  EXPECT_TRUE(specfem::utilities::is_close(pd3.xix, const_val))
      << ExpectedGot(const_val, pd3.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.etax, const_val))
      << ExpectedGot(const_val, pd3.etax);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.gammax, const_val))
      << ExpectedGot(const_val, pd3.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.xiy, const_val))
      << ExpectedGot(const_val, pd3.xiy);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.etay, const_val))
      << ExpectedGot(const_val, pd3.etay);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.gammay, const_val))
      << ExpectedGot(const_val, pd3.gammay);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.xiz, const_val))
      << ExpectedGot(const_val, pd3.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.etaz, const_val))
      << ExpectedGot(const_val, pd3.etaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.gammaz, const_val))
      << ExpectedGot(const_val, pd3.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd3.jacobian, const_val))
      << ExpectedGot(const_val, pd3.jacobian);

  // re-initialize pd2 to check init functionality
  pd2.init();

  EXPECT_TRUE(specfem::utilities::is_close(pd2.xix, zero_val))
      << ExpectedGot(zero_val, pd2.xix);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etax, zero_val))
      << ExpectedGot(zero_val, pd2.etax);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammax, zero_val))
      << ExpectedGot(zero_val, pd2.gammax);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.xiy, zero_val))
      << ExpectedGot(zero_val, pd2.xiy);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etay, zero_val))
      << ExpectedGot(zero_val, pd2.etay);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammay, zero_val))
      << ExpectedGot(zero_val, pd2.gammay);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.xiz, zero_val))
      << ExpectedGot(zero_val, pd2.xiz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.etaz, zero_val))
      << ExpectedGot(zero_val, pd2.etaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.gammaz, zero_val))
      << ExpectedGot(zero_val, pd2.gammaz);
  EXPECT_TRUE(specfem::utilities::is_close(pd2.jacobian, zero_val))
      << ExpectedGot(zero_val, pd2.jacobian);
}

// ===============================
// SIMD type verification (both cases)
// ===============================
TYPED_TEST(PointJacobianMatrixTest, VerifySIMDTypes) {
  constexpr bool using_simd = TypeParam::value;

  // Verify 2D types
  using PD2D = point::jacobian_matrix<dimension::type::dim2, false, using_simd>;
  using simd_type2D = typename PD2D::simd;
  bool is_expected_simd2D =
      std::is_same<simd_type2D,
                   specfem::datatype::simd<type_real, using_simd> >::value;
  EXPECT_TRUE(is_expected_simd2D);

  // Verify 3D types
  using PD3D = point::jacobian_matrix<dimension::type::dim3, false, using_simd>;
  using simd_type3D = typename PD3D::simd;
  bool is_expected_simd3D =
      std::is_same<simd_type3D,
                   specfem::datatype::simd<type_real, using_simd> >::value;
  EXPECT_TRUE(is_expected_simd3D);

  // Verify 2D with Jacobian
  using PD2DJac =
      point::jacobian_matrix<dimension::type::dim2, true, using_simd>;
  using simd_type2DJac = typename PD2DJac::simd;
  bool is_expected_simd2DJac =
      std::is_same<simd_type2DJac,
                   specfem::datatype::simd<type_real, using_simd> >::value;
  EXPECT_TRUE(is_expected_simd2DJac);

  // Verify 3D with Jacobian
  using PD3DJac =
      point::jacobian_matrix<dimension::type::dim3, true, using_simd>;
  using simd_type3DJac = typename PD3DJac::simd;
  bool is_expected_simd3DJac =
      std::is_same<simd_type3DJac,
                   specfem::datatype::simd<type_real, using_simd> >::value;
  EXPECT_TRUE(is_expected_simd3DJac);
}
