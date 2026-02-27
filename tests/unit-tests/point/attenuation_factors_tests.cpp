// Tests for specfem::point::attenuation_factors

#include "specfem/constants.hpp"
#include "specfem/element.hpp"
#include "specfem/point/attenuation_factors.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"
#include "test_helper.hpp"
#include "test_macros.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

using namespace specfem;

template <bool UseSIMD>
class PointAttenuationFactorsTestUntyped : public ::testing::Test {
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
class PointAttenuationFactorsTest
    : public PointAttenuationFactorsTestUntyped<T::value> {};

TYPED_TEST_SUITE(PointAttenuationFactorsTest, TestTypes);

// ============================================================
// Static property checks
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, StaticProperties2D) {
  constexpr bool using_simd = TypeParam::value;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  EXPECT_EQ(af_type::dimension_tag, element::dimension_tag::dim2);
  EXPECT_EQ(af_type::attenuation_tag,
            element::attenuation_tag::constant_isotropic);
  EXPECT_EQ(af_type::N_SLS, specfem::constants::N_SLS);
  EXPECT_EQ(af_type::using_simd, using_simd);
}

TYPED_TEST(PointAttenuationFactorsTest, StaticProperties3D) {
  constexpr bool using_simd = TypeParam::value;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  EXPECT_EQ(af_type::dimension_tag, element::dimension_tag::dim3);
  EXPECT_EQ(af_type::attenuation_tag,
            element::attenuation_tag::constant_isotropic);
  EXPECT_EQ(af_type::N_SLS, specfem::constants::N_SLS);
  EXPECT_EQ(af_type::using_simd, using_simd);
}

// ============================================================
// Value constructor — 2D
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, ValueConstructor2D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  // Use fill-constructor: all N_SLS slots get the same value
  typename af_type::common_factor_type kappa_val(1.5);
  typename af_type::common_factor_type mu_val(2.5);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;

  af_type af(kappa_val, mu_val, alpha, beta, gamma);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(
        specfem::utilities::is_close(af.kappa_common_factor(i), kappa_val(i)))
        << "kappa_common_factor(" << i
        << "): " << ExpectedGot(kappa_val(i), af.kappa_common_factor(i));
    EXPECT_TRUE(specfem::utilities::is_close(af.mu_common_factor(i), mu_val(i)))
        << "mu_common_factor(" << i
        << "): " << ExpectedGot(mu_val(i), af.mu_common_factor(i));
  }

  EXPECT_TRUE(specfem::utilities::is_close(af.alpha_rk, alpha))
      << ExpectedGot(alpha, af.alpha_rk);
  EXPECT_TRUE(specfem::utilities::is_close(af.beta_rk, beta))
      << ExpectedGot(beta, af.beta_rk);
  EXPECT_TRUE(specfem::utilities::is_close(af.gamma_rk, gamma))
      << ExpectedGot(gamma, af.gamma_rk);
}

// ============================================================
// Value constructor — 3D
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, ValueConstructor3D) {
  constexpr bool using_simd = TypeParam::value;
  constexpr int N = specfem::constants::N_SLS;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename af_type::common_factor_type kappa_val(3.0);
  typename af_type::common_factor_type mu_val(4.0);
  type_real alpha = 0.7, beta = 0.2, gamma = 0.1;

  af_type af(kappa_val, mu_val, alpha, beta, gamma);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(
        specfem::utilities::is_close(af.kappa_common_factor(i), kappa_val(i)))
        << "kappa_common_factor(" << i
        << "): " << ExpectedGot(kappa_val(i), af.kappa_common_factor(i));
    EXPECT_TRUE(specfem::utilities::is_close(af.mu_common_factor(i), mu_val(i)))
        << "mu_common_factor(" << i
        << "): " << ExpectedGot(mu_val(i), af.mu_common_factor(i));
  }

  EXPECT_TRUE(specfem::utilities::is_close(af.alpha_rk, alpha))
      << ExpectedGot(alpha, af.alpha_rk);
  EXPECT_TRUE(specfem::utilities::is_close(af.beta_rk, beta))
      << ExpectedGot(beta, af.beta_rk);
  EXPECT_TRUE(specfem::utilities::is_close(af.gamma_rk, gamma))
      << ExpectedGot(gamma, af.gamma_rk);
}

// ============================================================
// Equality operator — 2D
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, EqualityOperator2D) {
  constexpr bool using_simd = TypeParam::value;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename af_type::common_factor_type kappa_val(1.0);
  typename af_type::common_factor_type mu_val(2.0);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;

  af_type af1(kappa_val, mu_val, alpha, beta, gamma);
  af_type af2(kappa_val, mu_val, alpha, beta, gamma);
  // Different alpha_rk
  af_type af3(kappa_val, mu_val, alpha + static_cast<type_real>(0.1), beta,
              gamma);
  // Different mu_common_factor
  typename af_type::common_factor_type mu_val_alt(9.9);
  af_type af4(kappa_val, mu_val_alt, alpha, beta, gamma);

  EXPECT_TRUE(af1 == af2);
  EXPECT_FALSE(af1 == af3);
  EXPECT_FALSE(af1 == af4);
}

// ============================================================
// Equality operator — 3D
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, EqualityOperator3D) {
  constexpr bool using_simd = TypeParam::value;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  typename af_type::common_factor_type kappa_val(1.0);
  typename af_type::common_factor_type mu_val(2.0);
  type_real alpha = 0.5, beta = 0.3, gamma = 0.1;

  af_type af1(kappa_val, mu_val, alpha, beta, gamma);
  af_type af2(kappa_val, mu_val, alpha, beta, gamma);
  af_type af3(kappa_val, mu_val, alpha, beta,
              gamma + static_cast<type_real>(0.1));

  EXPECT_TRUE(af1 == af2);
  EXPECT_FALSE(af1 == af3);
}

// ============================================================
// print() method (non-SIMD only — string output)
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, PrintMethod2D) {
  constexpr bool using_simd = TypeParam::value;

  using af_type = point::attenuation_factors<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  if constexpr (!using_simd) {
    typename af_type::common_factor_type kappa_val(1.0);
    typename af_type::common_factor_type mu_val(2.0);
    type_real alpha = 0.5, beta = 0.3, gamma = 0.1;

    af_type af(kappa_val, mu_val, alpha, beta, gamma);
    std::string s = af.print();

    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("Attenuation Factors"), std::string::npos);
    EXPECT_NE(s.find("kappa_common_factor"), std::string::npos);
    EXPECT_NE(s.find("mu_common_factor"), std::string::npos);
    EXPECT_NE(s.find("alpha_rk"), std::string::npos);
    EXPECT_NE(s.find("beta_rk"), std::string::npos);
    EXPECT_NE(s.find("gamma_rk"), std::string::npos);
  }
}

// ============================================================
// SIMD type verification
// ============================================================

TYPED_TEST(PointAttenuationFactorsTest, SIMDTypeVerification) {
  constexpr bool using_simd = TypeParam::value;

  using af2d = point::attenuation_factors<
      element::dimension_tag::dim2, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  using af3d = point::attenuation_factors<
      element::dimension_tag::dim3, element::medium_tag::elastic,
      element::attenuation_tag::constant_isotropic, using_simd>;

  bool simd_match_2d =
      std::is_same_v<typename af2d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_2d);

  bool simd_match_3d =
      std::is_same_v<typename af3d::simd,
                     specfem::datatype::simd<type_real, using_simd> >;
  EXPECT_TRUE(simd_match_3d);
}
