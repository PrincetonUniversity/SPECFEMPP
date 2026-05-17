#include "specfem/enums.hpp"
#include "specfem/medium/dim2/elastic/isotropic/attenuation.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

// N_SLS = 3 (specfem::constants::N_SLS)
// Tags: dim2, elastic_psv, isotropic, constant_isotropic, non-SIMD
// T layout: T(0,0)=sigma_xx, T(0,1)=T(1,0)=sigma_xz, T(1,1)=sigma_zz

namespace {

// -------------------------------------------------------------------------
// impl_add_relaxation_to_stress — dim2 PSV
// -------------------------------------------------------------------------

TEST(AttenuationRelaxation, Dim2PSV_ZeroMemoryVariables) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  // Default-constructed attenuation: all R = 0
  AttType att;

  StressType stress;
  stress.T(0, 0) = 1.0;
  stress.T(0, 1) = 2.0;
  stress.T(1, 0) = 2.0;
  stress.T(1, 1) = 3.0;

  StressType expected = stress;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim2PSV_PureDeviatoric) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 1.0;
    att.Rxz(j) = 0.0;
    att.Rkappa(j) = 0.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(1, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // R_xx_sum = 3, R_kappa_sum = 0
  // T(0,0) -= 3 + 0 = -3
  // T(1,1) += 3 - 0 = +3
  StressType expected;
  expected.T(0, 0) = -3.0;
  expected.T(0, 1) = 0.0;
  expected.T(1, 0) = 0.0;
  expected.T(1, 1) = 3.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim2PSV_PureVolumetric) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 0.0;
    att.Rxz(j) = 0.0;
    att.Rkappa(j) = 1.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(1, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // R_xx_sum = 0, R_kappa_sum = 3
  // T(0,0) -= 0 + 3 = -3
  // T(1,1) += 0 - 3 = -3
  StressType expected;
  expected.T(0, 0) = -3.0;
  expected.T(0, 1) = 0.0;
  expected.T(1, 0) = 0.0;
  expected.T(1, 1) = -3.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim2PSV_PureShear) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 0.0;
    att.Rxz(j) = 1.0;
    att.Rkappa(j) = 0.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(1, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // R_xz_sum = 3; normal stress unchanged, shear -= 3
  StressType expected;
  expected.T(0, 0) = 0.0;
  expected.T(0, 1) = -3.0;
  expected.T(1, 0) = -3.0;
  expected.T(1, 1) = 0.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim2PSV_AllNonzero) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  // Rxx(j)=1, Rxz(j)=2, Rkappa(j)=3 for all j
  // Sums: R_xx_sum=3, R_xz_sum=6, R_kappa_sum=9
  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 1.0;
    att.Rxz(j) = 2.0;
    att.Rkappa(j) = 3.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.5;
  stress.T(0, 1) = 0.5;
  stress.T(1, 0) = 0.5;
  stress.T(1, 1) = 0.5;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // T(0,0) = 0.5 - (3 + 9) = 0.5 - 12 = -11.5
  // T(1,1) = 0.5 + (3 - 9) = 0.5 - 6  = -5.5
  // T(0,1) = T(1,0) = 0.5 - 6 = -5.5
  StressType expected;
  expected.T(0, 0) = -11.5;
  expected.T(0, 1) = -5.5;
  expected.T(1, 0) = -5.5;
  expected.T(1, 1) = -5.5;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

// -------------------------------------------------------------------------
// impl_integrate_memory_variables — dim2 PSV
// -------------------------------------------------------------------------

TEST(AttenuationIntegrate, Dim2PSV_ZeroState) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using FieldDerivType = specfem::point::field_derivatives<Tags>;

  constexpr int N = specfem::constants::N_SLS;

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.alpha_rk(j) = 0.5;
    att.beta_rk(j) = 0.5;
    att.gamma_rk(j) = 0.5;
    att.mu_relaxation_rate(j) = 1.0;
    att.kappa_relaxation_rate(j) = 1.0;
  }
  // att.epsilon_* = 0 (default constructor)

  FieldDerivType du, dv;
  du.du(0, 0) = 0.0;
  du.du(1, 1) = 0.0;
  du.du(0, 1) = 0.0;
  du.du(1, 0) = 0.0;
  dv.du(0, 0) = 0.0;
  dv.du(1, 1) = 0.0;
  dv.du(0, 1) = 0.0;
  dv.du(1, 0) = 0.0;

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_EQ(att.Rxx(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Rxz(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Rkappa(j), static_cast<type_real>(0.0));
  }
}

TEST(AttenuationIntegrate, Dim2PSV_SingleStepRecurrence) {
  // du(0,0) = du(1,1) = 3, du(0,1) = du(1,0) = 2, dv = 0, dt = 1
  // alpha=beta=gamma=0.5, mu_rate=kappa_rate=1, R_init=0
  //
  // trace_Sn = trace_Snp1 = 6
  // epsilondev_xx = 3 - (1/3)*6 = 1
  // epsilondev_xz = 0.5*(2+2) = 2
  //
  // Rxx_new = 0 + 1*(0.5*1 + 0.5*1) = 1
  // Rxz_new = 0 + 1*(0.5*2 + 0.5*2) = 2
  // Rkappa_new = 0 + 1*(0.5*6 + 0.5*6) = 6
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using FieldDerivType = specfem::point::field_derivatives<Tags>;

  constexpr int N = specfem::constants::N_SLS;

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.alpha_rk(j) = 0.5;
    att.beta_rk(j) = 0.5;
    att.gamma_rk(j) = 0.5;
    att.mu_relaxation_rate(j) = 1.0;
    att.kappa_relaxation_rate(j) = 1.0;
    // R_init = 0 (set by default constructor)
  }
  // Sn stored in epsilon fields
  att.epsilon_xx = 3.0;
  att.epsilon_zz = 3.0;
  att.epsilon_xz = 2.0;

  FieldDerivType du, dv;
  du.du(0, 0) = 3.0;
  du.du(1, 1) = 3.0;
  du.du(0, 1) = 2.0;
  du.du(1, 0) = 2.0;
  dv.du(0, 0) = 0.0;
  dv.du(1, 1) = 0.0;
  dv.du(0, 1) = 0.0;
  dv.du(1, 0) = 0.0;

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_NEAR(static_cast<double>(att.Rxx(j)), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxz(j)), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rkappa(j)), 6.0, 1e-6);
  }
  // Snp1 written back
  EXPECT_NEAR(static_cast<double>(att.epsilon_xx), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(att.epsilon_zz), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(att.epsilon_xz), 2.0, 1e-6);
}

TEST(AttenuationIntegrate, Dim2PSV_AlphaDecay) {
  // beta = gamma = 0 => R_new = alpha * R_old (pure exponential decay)
  // alpha = 0.5, R_init = {Rxx=2, Rxz=3, Rkappa=4}
  // => R_new = {1, 1.5, 2}
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim2,
                          specfem::element::medium_tag::elastic_psv,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using FieldDerivType = specfem::point::field_derivatives<Tags>;

  constexpr int N = specfem::constants::N_SLS;

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.alpha_rk(j) = 0.5;
    att.beta_rk(j) = 0.0;
    att.gamma_rk(j) = 0.0;
    att.mu_relaxation_rate(j) = 1.0;
    att.kappa_relaxation_rate(j) = 1.0;
    att.Rxx(j) = 2.0;
    att.Rxz(j) = 3.0;
    att.Rkappa(j) = 4.0;
  }
  // Sn stored in epsilon fields (values don't affect R since beta=gamma=0)
  att.epsilon_xx = 5.0;
  att.epsilon_zz = 5.0;
  att.epsilon_xz = 5.0;

  FieldDerivType du, dv;
  du.du(0, 0) = 5.0;
  du.du(1, 1) = 5.0;
  du.du(0, 1) = 5.0;
  du.du(1, 0) = 5.0;
  dv.du(0, 0) = 0.0;
  dv.du(1, 1) = 0.0;
  dv.du(0, 1) = 0.0;
  dv.du(1, 0) = 0.0;

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_NEAR(static_cast<double>(att.Rxx(j)), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxz(j)), 1.5, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rkappa(j)), 2.0, 1e-6);
  }
}

} // namespace
