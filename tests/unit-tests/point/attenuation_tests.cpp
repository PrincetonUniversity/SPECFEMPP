// Tests for specfem::point::attenuation (combined attenuation_factors +
// memory_variable)

#include "SPECFEM_Environment.hpp"
#include "specfem/constants.hpp"
#include "specfem/element.hpp"
#include "specfem/point/attenuation.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

template <bool UseSIMD>
class PointAttenuationTestUntyped : public ::testing::Test {};

template <typename T>
class PointAttenuationTest : public PointAttenuationTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointAttenuationTest, TestTypes);

// ============================================================
// Static property checks
// ============================================================

TYPED_TEST(PointAttenuationTest, StaticProperties2D) {
  constexpr bool using_simd = TypeParam::value;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  EXPECT_EQ(att_type::dimension_tag, element::dimension_tag::dim2);
  EXPECT_EQ(att_type::attenuation_tag,
            element::attenuation_tag::constant_isotropic);
  EXPECT_EQ(att_type::N_SLS, specfem::constants::N_SLS);
  EXPECT_EQ(att_type::using_simd, using_simd);
}

TYPED_TEST(PointAttenuationTest, StaticProperties3D) {
  constexpr bool using_simd = TypeParam::value;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  EXPECT_EQ(att_type::dimension_tag, element::dimension_tag::dim3);
  EXPECT_EQ(att_type::attenuation_tag,
            element::attenuation_tag::constant_isotropic);
  EXPECT_EQ(att_type::N_SLS, specfem::constants::N_SLS);
  EXPECT_EQ(att_type::using_simd, using_simd);
}

// ============================================================
// Default constructor — R fields zero; factor fields default
// ============================================================

TYPED_TEST(PointAttenuationTest, DefaultConstructor2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  att_type att;
  typename att_type::value_type zero(0.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), zero(i)))
        << "Rxx(" << i << "): " << ExpectedGot(zero(i), att.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), zero(i)))
        << "Rxz(" << i << "): " << ExpectedGot(zero(i), att.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), zero(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(zero(i), att.Rkappa(i));
  }
}

TYPED_TEST(PointAttenuationTest, DefaultConstructor3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  att_type att;
  typename att_type::value_type zero(0.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryy(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxy(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryz(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), zero(i)));
  }
}

// ============================================================
// Full value constructor — 2D
// ============================================================

TYPED_TEST(PointAttenuationTest, ValueConstructor2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(1.5);
  typename att_type::common_factor_type mu_val(2.5);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;
  typename att_type::value_type rxx_val(1.0), rxz_val(2.0), rk_val(3.0);

  att_type att(kappa_val, mu_val, alpha, beta, gamma, rxx_val, rxz_val, rk_val);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(
        specfem::utilities::is_close(att.kappa_common_factor(i), kappa_val(i)))
        << "kappa_common_factor(" << i
        << "): " << ExpectedGot(kappa_val(i), att.kappa_common_factor(i));
    EXPECT_TRUE(
        specfem::utilities::is_close(att.mu_common_factor(i), mu_val(i)))
        << "mu_common_factor(" << i
        << "): " << ExpectedGot(mu_val(i), att.mu_common_factor(i));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), rxx_val(i)))
        << "Rxx(" << i << "): " << ExpectedGot(rxx_val(i), att.Rxx(i));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), rxz_val(i)))
        << "Rxz(" << i << "): " << ExpectedGot(rxz_val(i), att.Rxz(i));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), rk_val(i)))
        << "Rkappa(" << i << "): " << ExpectedGot(rk_val(i), att.Rkappa(i));
  }
  EXPECT_TRUE(specfem::utilities::is_close(att.alpha_rk, alpha))
      << ExpectedGot(alpha, att.alpha_rk);
  EXPECT_TRUE(specfem::utilities::is_close(att.beta_rk, beta))
      << ExpectedGot(beta, att.beta_rk);
  EXPECT_TRUE(specfem::utilities::is_close(att.gamma_rk, gamma))
      << ExpectedGot(gamma, att.gamma_rk);
}

// ============================================================
// Full value constructor — 3D
// ============================================================

TYPED_TEST(PointAttenuationTest, ValueConstructor3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(3.0);
  typename att_type::common_factor_type mu_val(4.0);
  type_real alpha = 0.7, beta = 0.2, gamma = 0.1;
  typename att_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0),
      ryz(5.0), rk(6.0);

  att_type att(kappa_val, mu_val, alpha, beta, gamma, rxx, ryy, rxy, rxz, ryz,
               rk);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(
        specfem::utilities::is_close(att.kappa_common_factor(i), kappa_val(i)));
    EXPECT_TRUE(
        specfem::utilities::is_close(att.mu_common_factor(i), mu_val(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryy(i), ryy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxy(i), rxy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryz(i), ryz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), rk(i)));
  }
  EXPECT_TRUE(specfem::utilities::is_close(att.alpha_rk, alpha));
  EXPECT_TRUE(specfem::utilities::is_close(att.beta_rk, beta));
  EXPECT_TRUE(specfem::utilities::is_close(att.gamma_rk, gamma));
}

// ============================================================
// init() method — resets R fields to zero
// ============================================================

TYPED_TEST(PointAttenuationTest, InitMethod2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(1.0);
  typename att_type::common_factor_type mu_val(2.0);
  typename att_type::value_type const_val(5.0);
  typename att_type::value_type zero(0.0);

  att_type att(kappa_val, mu_val, 0.5, 0.3, 0.1, const_val, const_val,
               const_val);
  att.init();

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), zero(i)));
  }
}

TYPED_TEST(PointAttenuationTest, InitMethod3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(1.0);
  typename att_type::common_factor_type mu_val(2.0);
  typename att_type::value_type const_val(7.0);
  typename att_type::value_type zero(0.0);

  att_type att(kappa_val, mu_val, 0.5, 0.3, 0.1, const_val, const_val,
               const_val, const_val, const_val, const_val);
  att.init();

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxx(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryy(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxy(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rxz(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Ryz(i), zero(i)));
    EXPECT_TRUE(specfem::utilities::is_close(att.Rkappa(i), zero(i)));
  }
}

// ============================================================
// operator+ — 2D
// ============================================================

TYPED_TEST(PointAttenuationTest, Addition2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kf(1.0);
  typename att_type::common_factor_type mf(2.0);
  typename att_type::value_type a_rxx(1.0), a_rxz(2.0), a_rk(3.0);
  typename att_type::value_type b_rxx(10.0), b_rxz(20.0), b_rk(30.0);

  att_type a(kf, mf, 0.5, 0.3, 0.1, a_rxx, a_rxz, a_rk);
  att_type b(kf, mf, 0.5, 0.3, 0.1, b_rxx, b_rxz, b_rk);
  att_type c = a + b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxx(i), a_rxx(i) + b_rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxz(i), a_rxz(i) + b_rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rkappa(i), a_rk(i) + b_rk(i)));
  }
}

// ============================================================
// operator+= — 2D
// ============================================================

TYPED_TEST(PointAttenuationTest, AdditionAssignment2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kf(1.0);
  typename att_type::common_factor_type mf(2.0);
  typename att_type::value_type a_rxx(1.0), a_rxz(2.0), a_rk(3.0);
  typename att_type::value_type b_rxx(10.0), b_rxz(20.0), b_rk(30.0);

  att_type a(kf, mf, 0.5, 0.3, 0.1, a_rxx, a_rxz, a_rk);
  att_type b(kf, mf, 0.5, 0.3, 0.1, b_rxx, b_rxz, b_rk);
  a += b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxx(i), a_rxx(i) + b_rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rxz(i), a_rxz(i) + b_rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(a.Rkappa(i), a_rk(i) + b_rk(i)));
  }
}

// ============================================================
// operator* (scalar) — 2D
// ============================================================

TYPED_TEST(PointAttenuationTest, ScalarMultiplication2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename att_type::common_factor_type kf(1.0);
    typename att_type::common_factor_type mf(2.0);
    typename att_type::value_type rxx_val(1.0), rxz_val(2.0), rk_val(3.0);
    type_real scalar = 2.5;

    att_type att(kf, mf, 0.5, 0.3, 0.1, rxx_val, rxz_val, rk_val);
    att_type result = att * scalar;

    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rxx(i), rxx_val(i) * scalar));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rxz(i), rxz_val(i) * scalar));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rkappa(i), rk_val(i) * scalar));
    }
  }
}

// ============================================================
// operator+ — 3D
// ============================================================

TYPED_TEST(PointAttenuationTest, Addition3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kf(1.0);
  typename att_type::common_factor_type mf(2.0);
  typename att_type::value_type a_rxx(1.0), a_ryy(2.0), a_rxy(3.0), a_rxz(4.0),
      a_ryz(5.0), a_rk(6.0);
  typename att_type::value_type b_rxx(10.0), b_ryy(20.0), b_rxy(30.0),
      b_rxz(40.0), b_ryz(50.0), b_rk(60.0);

  att_type a(kf, mf, 0.5, 0.3, 0.1, a_rxx, a_ryy, a_rxy, a_rxz, a_ryz, a_rk);
  att_type b(kf, mf, 0.5, 0.3, 0.1, b_rxx, b_ryy, b_rxy, b_rxz, b_ryz, b_rk);
  att_type c = a + b;

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxx(i), a_rxx(i) + b_rxx(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Ryy(i), a_ryy(i) + b_ryy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxy(i), a_rxy(i) + b_rxy(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rxz(i), a_rxz(i) + b_rxz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Ryz(i), a_ryz(i) + b_ryz(i)));
    EXPECT_TRUE(specfem::utilities::is_close(c.Rkappa(i), a_rk(i) + b_rk(i)));
  }
}

// ============================================================
// operator+= — 3D
// ============================================================

TYPED_TEST(PointAttenuationTest, AdditionAssignment3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kf(1.0);
  typename att_type::common_factor_type mf(2.0);
  typename att_type::value_type a_rxx(1.0), a_ryy(2.0), a_rxy(3.0), a_rxz(4.0),
      a_ryz(5.0), a_rk(6.0);
  typename att_type::value_type b_rxx(10.0), b_ryy(20.0), b_rxy(30.0),
      b_rxz(40.0), b_ryz(50.0), b_rk(60.0);

  att_type a(kf, mf, 0.5, 0.3, 0.1, a_rxx, a_ryy, a_rxy, a_rxz, a_ryz, a_rk);
  att_type b(kf, mf, 0.5, 0.3, 0.1, b_rxx, b_ryy, b_rxy, b_rxz, b_ryz, b_rk);
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

// ============================================================
// operator* (scalar) — 3D
// ============================================================

TYPED_TEST(PointAttenuationTest, ScalarMultiplication3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename att_type::common_factor_type kf(1.0);
    typename att_type::common_factor_type mf(2.0);
    typename att_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0),
        ryz(5.0), rk(6.0);
    type_real scalar = 2.5;

    att_type att(kf, mf, 0.5, 0.3, 0.1, rxx, ryy, rxy, rxz, ryz, rk);
    att_type result = att * scalar;

    for (int i = 0; i < N; ++i) {
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxx(i), rxx(i) * scalar));
      EXPECT_TRUE(specfem::utilities::is_close(result.Ryy(i), ryy(i) * scalar));
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxy(i), rxy(i) * scalar));
      EXPECT_TRUE(specfem::utilities::is_close(result.Rxz(i), rxz(i) * scalar));
      EXPECT_TRUE(specfem::utilities::is_close(result.Ryz(i), ryz(i) * scalar));
      EXPECT_TRUE(
          specfem::utilities::is_close(result.Rkappa(i), rk(i) * scalar));
    }
  }
}

// ============================================================
// Equality operator — 2D
// ============================================================

TYPED_TEST(PointAttenuationTest, EqualityOperator2D) {
  constexpr bool using_simd = TypeParam::value;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(1.0);
  typename att_type::common_factor_type mu_val(2.0);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;
  typename att_type::value_type rxx(1.0), rxz(2.0), rk(3.0);
  typename att_type::value_type rxx_alt(9.0);

  att_type att1(kappa_val, mu_val, alpha, beta, gamma, rxx, rxz, rk);
  att_type att2(kappa_val, mu_val, alpha, beta, gamma, rxx, rxz, rk);
  // Different alpha_rk
  att_type att3(kappa_val, mu_val, alpha + static_cast<type_real>(0.1), beta,
                gamma, rxx, rxz, rk);
  // Different Rxx
  att_type att4(kappa_val, mu_val, alpha, beta, gamma, rxx_alt, rxz, rk);

  EXPECT_TRUE(att1 == att2);
  EXPECT_FALSE(att1 == att3);
  EXPECT_FALSE(att1 == att4);
}

// ============================================================
// Equality operator — 3D
// ============================================================

TYPED_TEST(PointAttenuationTest, EqualityOperator3D) {
  constexpr bool using_simd = TypeParam::value;

  using att_type = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename att_type::common_factor_type kappa_val(1.0);
  typename att_type::common_factor_type mu_val(2.0);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;
  typename att_type::value_type rxx(1.0), ryy(2.0), rxy(3.0), rxz(4.0),
      ryz(5.0), rk(6.0);
  typename att_type::value_type ryy_alt(9.0);

  att_type att1(kappa_val, mu_val, alpha, beta, gamma, rxx, ryy, rxy, rxz, ryz,
                rk);
  att_type att2(kappa_val, mu_val, alpha, beta, gamma, rxx, ryy, rxy, rxz, ryz,
                rk);
  att_type att3(kappa_val, mu_val, alpha, beta, gamma, rxx, ryy_alt, rxy, rxz,
                ryz, rk);

  EXPECT_TRUE(att1 == att2);
  EXPECT_FALSE(att1 == att3);
}

// ============================================================
// print() method — non-SIMD only
// ============================================================

TYPED_TEST(PointAttenuationTest, PrintMethod2D) {
  constexpr bool using_simd = TypeParam::value;

  using att_type = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename att_type::common_factor_type kappa_val(1.0);
    typename att_type::common_factor_type mu_val(2.0);
    typename att_type::value_type rxx(1.0), rxz(2.0), rk(3.0);
    type_real alpha = 0.5, beta = 0.3, gamma = 0.1;

    att_type att(kappa_val, mu_val, alpha, beta, gamma, rxx, rxz, rk);
    std::string s = att.print();

    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("Attenuation Factors"), std::string::npos);
    EXPECT_NE(s.find("kappa_common_factor"), std::string::npos);
    EXPECT_NE(s.find("mu_common_factor"), std::string::npos);
    EXPECT_NE(s.find("alpha_rk"), std::string::npos);
    EXPECT_NE(s.find("beta_rk"), std::string::npos);
    EXPECT_NE(s.find("gamma_rk"), std::string::npos);
    EXPECT_NE(s.find("Memory Variables"), std::string::npos);
  }
}

// ============================================================
// SIMD type verification
// ============================================================

TYPED_TEST(PointAttenuationTest, SIMDTypeVerification) {
  constexpr bool using_simd = TypeParam::value;

  using att2d = point::attenuation<
      element::dimension_tag::dim2, element::medium_tag::elastic_psv,
      element::attenuation_tag::constant_isotropic, using_simd>;
  using att3d = point::attenuation<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  bool simd_match_2d =
      std::is_same_v<typename att2d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_2d);

  bool simd_match_3d =
      std::is_same_v<typename att3d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_3d);
}
