// Tests for specfem::point::memory_variable (dim2 and dim3,
// elastic/constant_isotropic)

#include "specfem/constants.hpp"
#include "specfem/element.hpp"
#include "specfem/point/memory_variable.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

template <bool UseSIMD>
class PointMemoryVariableTestUntyped : public ::testing::Test {
protected:
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
class PointMemoryVariableTest
    : public PointMemoryVariableTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointMemoryVariableTest, TestTypes);

// ============================================================
// dim2 tests
// ============================================================

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_DefaultConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  mv_type mv;
  typename mv_type::value_type zero(0.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), zero(i)))
        << "Rxx(" << i << "): " << ExpectedGot(zero(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), zero(i)))
        << "Rxz(" << i << "): " << ExpectedGot(zero(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), zero(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(zero(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_ValueConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type rxx_val(1.0);
  typename mv_type::value_type rxz_val(2.0);
  typename mv_type::value_type rk_val(3.0);

  mv_type mv(rxx_val, rxz_val, rk_val);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), rxx_val(i)))
        << "Rxx(" << i << "): " << ExpectedGot(rxx_val(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), rxz_val(i)))
        << "Rxz(" << i << "): " << ExpectedGot(rxz_val(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), rk_val(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(rk_val(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_ConstantConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type const_val(5.0);
  mv_type mv(const_val);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), const_val(i)))
        << "Rxx(" << i << "): " << ExpectedGot(const_val(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), const_val(i)))
        << "Rxz(" << i << "): " << ExpectedGot(const_val(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), const_val(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(const_val(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_InitMethod) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type const_val(5.0);
  typename mv_type::value_type zero(0.0);

  mv_type mv(const_val);
  mv.init();

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), zero(i)))
        << "Rxx(" << i << "): " << ExpectedGot(zero(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), zero(i)))
        << "Rxz(" << i << "): " << ExpectedGot(zero(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), zero(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(zero(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_Addition) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type a_rxx(1.0), a_rxz(2.0), a_rk(3.0);
  typename mv_type::value_type b_rxx(10.0), b_rxz(20.0), b_rk(30.0);

  mv_type a(a_rxx, a_rxz, a_rk);
  mv_type b(b_rxx, b_rxz, b_rk);
  mv_type c = a + b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxx(i), a_rxx(i) + b_rxx(i)))
        << "c.Rxx(" << i << "): " << ExpectedGot(a_rxx(i) + b_rxx(i), c.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxz(i), a_rxz(i) + b_rxz(i)))
        << "c.Rxz(" << i << "): " << ExpectedGot(a_rxz(i) + b_rxz(i), c.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rkappa(i), a_rk(i) + b_rk(i)))
        << "c.Rkappa(" << i
        << "): " << ExpectedGot(a_rk(i) + b_rk(i), c.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_AdditionAssignment) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type a_rxx(1.0), a_rxz(2.0), a_rk(3.0);
  typename mv_type::value_type b_rxx(10.0), b_rxz(20.0), b_rk(30.0);

  mv_type a(a_rxx, a_rxz, a_rk);
  mv_type b(b_rxx, b_rxz, b_rk);
  a += b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxx(i), a_rxx(i) + b_rxx(i)))
        << "a.Rxx(" << i << "): " << ExpectedGot(a_rxx(i) + b_rxx(i), a.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxz(i), a_rxz(i) + b_rxz(i)))
        << "a.Rxz(" << i << "): " << ExpectedGot(a_rxz(i) + b_rxz(i), a.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rkappa(i), a_rk(i) + b_rk(i)))
        << "a.Rkappa(" << i
        << "): " << ExpectedGot(a_rk(i) + b_rk(i), a.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_ScalarMultiplication) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename mv_type::value_type rxx_val(1.0);
    typename mv_type::value_type rxz_val(2.0);
    typename mv_type::value_type rk_val(3.0);
    type_real scalar = 2.5;

    mv_type mv(rxx_val, rxz_val, rk_val);
    mv_type result = mv * scalar;

    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rxx(i), rxx_val(i) * scalar))
          << "result.Rxx(" << i
          << "): " << ExpectedGot(rxx_val(i) * scalar, result.Rxx(i));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rxz(i), rxz_val(i) * scalar))
          << "result.Rxz(" << i
          << "): " << ExpectedGot(rxz_val(i) * scalar, result.Rxz(i));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rkappa(i), rk_val(i) * scalar))
          << "result.Rkappa(" << i
          << "): " << ExpectedGot(rk_val(i) * scalar, result.Rkappa(i));
    }
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable2D_EqualityOperator) {
  constexpr bool using_simd = TypeParam::value;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type rxx(1.0), rxz(2.0), rk(3.0);
  typename mv_type::value_type rxx_alt(9.0);

  mv_type mv1(rxx, rxz, rk);
  mv_type mv2(rxx, rxz, rk);
  mv_type mv3(rxx_alt, rxz, rk);

  EXPECT_TRUE(mv1 == mv2);
  EXPECT_FALSE(mv1 == mv3);
}

// ============================================================
// dim3 tests
// ============================================================

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_DefaultConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  mv_type mv;
  typename mv_type::value_type zero(0.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), zero(i)))
        << "Rxx(" << i << "): " << ExpectedGot(zero(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryy(i), zero(i)))
        << "Ryy(" << i << "): " << ExpectedGot(zero(i), mv.Ryy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxy(i), zero(i)))
        << "Rxy(" << i << "): " << ExpectedGot(zero(i), mv.Rxy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), zero(i)))
        << "Rxz(" << i << "): " << ExpectedGot(zero(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryz(i), zero(i)))
        << "Ryz(" << i << "): " << ExpectedGot(zero(i), mv.Ryz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), zero(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(zero(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_ValueConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0), ryz(5.0),
      rk(6.0);

  mv_type mv(rxx, ryy, rxy, rxz, ryz, rk);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), rxx(i)))
        << "Rxx(" << i << "): " << ExpectedGot(rxx(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryy(i), ryy(i)))
        << "Ryy(" << i << "): " << ExpectedGot(ryy(i), mv.Ryy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxy(i), rxy(i)))
        << "Rxy(" << i << "): " << ExpectedGot(rxy(i), mv.Rxy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), rxz(i)))
        << "Rxz(" << i << "): " << ExpectedGot(rxz(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryz(i), ryz(i)))
        << "Ryz(" << i << "): " << ExpectedGot(ryz(i), mv.Ryz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), rk(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(rk(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_ConstantConstructor) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type const_val(7.0);
  mv_type mv(const_val);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), const_val(i)))
        << "Rxx(" << i << "): " << ExpectedGot(const_val(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryy(i), const_val(i)))
        << "Ryy(" << i << "): " << ExpectedGot(const_val(i), mv.Ryy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxy(i), const_val(i)))
        << "Rxy(" << i << "): " << ExpectedGot(const_val(i), mv.Rxy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), const_val(i)))
        << "Rxz(" << i << "): " << ExpectedGot(const_val(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryz(i), const_val(i)))
        << "Ryz(" << i << "): " << ExpectedGot(const_val(i), mv.Ryz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), const_val(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(const_val(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_InitMethod) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type const_val(7.0);
  typename mv_type::value_type zero(0.0);

  mv_type mv(const_val);
  mv.init();

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxx(i), zero(i)))
        << "Rxx(" << i << "): " << ExpectedGot(zero(i), mv.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryy(i), zero(i)))
        << "Ryy(" << i << "): " << ExpectedGot(zero(i), mv.Ryy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxy(i), zero(i)))
        << "Rxy(" << i << "): " << ExpectedGot(zero(i), mv.Rxy(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rxz(i), zero(i)))
        << "Rxz(" << i << "): " << ExpectedGot(zero(i), mv.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Ryz(i), zero(i)))
        << "Ryz(" << i << "): " << ExpectedGot(zero(i), mv.Ryz(i));
    EXPECT_TRUE(specfem::utilities::is_close(mv.Rkappa(i), zero(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(zero(i), mv.Rkappa(i));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_Addition) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type a_rxx(1.0), a_ryy(2.0), a_rxy(3.0), a_rxz(4.0),
      a_ryz(5.0), a_rk(6.0);
  typename mv_type::value_type b_rxx(10.0), b_ryy(20.0), b_rxy(30.0),
      b_rxz(40.0), b_ryz(50.0), b_rk(60.0);

  mv_type a(a_rxx, a_ryy, a_rxy, a_rxz, a_ryz, a_rk);
  mv_type b(b_rxx, b_ryy, b_rxy, b_rxz, b_ryz, b_rk);
  mv_type c = a + b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxx(i), a_rxx(i) + b_rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Ryy(i), a_ryy(i) + b_ryy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxy(i), a_rxy(i) + b_rxy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxz(i), a_rxz(i) + b_rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Ryz(i), a_ryz(i) + b_ryz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rkappa(i), a_rk(i) + b_rk(i)));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_AdditionAssignment) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type a_rxx(1.0), a_ryy(2.0), a_rxy(3.0), a_rxz(4.0),
      a_ryz(5.0), a_rk(6.0);
  typename mv_type::value_type b_rxx(10.0), b_ryy(20.0), b_rxy(30.0),
      b_rxz(40.0), b_ryz(50.0), b_rk(60.0);

  mv_type a(a_rxx, a_ryy, a_rxy, a_rxz, a_ryz, a_rk);
  mv_type b(b_rxx, b_ryy, b_rxy, b_rxz, b_ryz, b_rk);
  a += b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxx(i), a_rxx(i) + b_rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Ryy(i), a_ryy(i) + b_ryy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxy(i), a_rxy(i) + b_rxy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxz(i), a_rxz(i) + b_rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Ryz(i), a_ryz(i) + b_ryz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rkappa(i), a_rk(i) + b_rk(i)));
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_ScalarMultiplication) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename mv_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0),
        ryz(5.0), rk(6.0);
    type_real scalar = 2.5;

    mv_type mv(rxx, ryy, rxy, rxz, ryz, rk);
    mv_type result = mv * scalar;

    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxx(i), rxx(i) * scalar))
          << "result.Rxx(" << i
          << "): " << ExpectedGot(rxx(i) * scalar, result.Rxx(i));
      EXPECT_TRUE(specfem::utilities::is_close(result.Ryy(i), ryy(i) * scalar))
          << "result.Ryy(" << i
          << "): " << ExpectedGot(ryy(i) * scalar, result.Ryy(i));
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxy(i), rxy(i) * scalar))
          << "result.Rxy(" << i
          << "): " << ExpectedGot(rxy(i) * scalar, result.Rxy(i));
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxz(i), rxz(i) * scalar))
          << "result.Rxz(" << i
          << "): " << ExpectedGot(rxz(i) * scalar, result.Rxz(i));
      EXPECT_TRUE(specfem::utilities::is_close(result.Ryz(i), ryz(i) * scalar))
          << "result.Ryz(" << i
          << "): " << ExpectedGot(ryz(i) * scalar, result.Ryz(i));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rkappa(i), rk(i) * scalar))
          << "result.Rkappa(" << i
          << "): " << ExpectedGot(rk(i) * scalar, result.Rkappa(i));
    }
  }
}

TYPED_TEST(PointMemoryVariableTest, MemoryVariable3D_EqualityOperator) {
  constexpr bool using_simd = TypeParam::value;

  using mv_type = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename mv_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0), ryz(5.0),
      rk(6.0);
  typename mv_type::value_type ryy_alt(9.0);

  mv_type mv1(rxx, ryy, rxy, rxz, ryz, rk);
  mv_type mv2(rxx, ryy, rxy, rxz, ryz, rk);
  mv_type mv3(rxx, ryy_alt, rxy, rxz, ryz, rk);

  EXPECT_TRUE(mv1 == mv2);
  EXPECT_FALSE(mv1 == mv3);
}

// ============================================================
// SIMD type verification
// ============================================================

TYPED_TEST(PointMemoryVariableTest, SIMDTypeVerification) {
  constexpr bool using_simd = TypeParam::value;

  using mv2d = point::memory_variable<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;
  using mv3d = point::memory_variable<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  bool simd_match_2d =
      std::is_same_v<typename mv2d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_2d);

  bool simd_match_3d =
      std::is_same_v<typename mv3d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_3d);
}
