#include "specfem/enums.hpp"
#include "specfem/medium/dim3/elastic/isotropic/attenuation.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

// N_SLS = 3 (specfem::constants::N_SLS)
// Tags: dim3, elastic, isotropic, constant_isotropic, non-SIMD
// T layout: T(0,0)=sigma_xx, T(1,1)=sigma_yy, T(2,2)=sigma_zz
//           T(0,1)=T(1,0)=sigma_xy, T(0,2)=T(2,0)=sigma_xz,
//           T(1,2)=T(2,1)=sigma_yz

namespace {

// -------------------------------------------------------------------------
// impl_add_relaxation_to_stress — dim3 elastic
// -------------------------------------------------------------------------

TEST(AttenuationRelaxation, Dim3Elastic_ZeroMemoryVariables) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  // Default-constructed: all R = 0
  AttType att;

  StressType stress;
  stress.T(0, 0) = 1.0;
  stress.T(1, 1) = 2.0;
  stress.T(2, 2) = 3.0;
  stress.T(0, 1) = 4.0;
  stress.T(1, 0) = 4.0;
  stress.T(0, 2) = 5.0;
  stress.T(2, 0) = 5.0;
  stress.T(1, 2) = 6.0;
  stress.T(2, 1) = 6.0;

  StressType expected = stress;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim3Elastic_PureDeviatoric) {
  // Rxx nonzero only: affects T(0,0), T(1,1) [unchanged], T(2,2) [+Rxx]
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 1.0;
    // Ryy, Rxy, Rxz, Ryz, Rkappa = 0 (default)
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(1, 1) = 0.0;
  stress.T(2, 2) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(0, 2) = 0.0;
  stress.T(2, 0) = 0.0;
  stress.T(1, 2) = 0.0;
  stress.T(2, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // R_xx_sum = 3, all others = 0
  // T(0,0) -= 3 + 0 = -3
  // T(1,1) -= 0 + 0 = 0
  // T(2,2) += 3 + 0 - 0 = +3
  StressType expected;
  expected.T(0, 0) = -3.0;
  expected.T(1, 1) = 0.0;
  expected.T(2, 2) = 3.0;
  expected.T(0, 1) = 0.0;
  expected.T(1, 0) = 0.0;
  expected.T(0, 2) = 0.0;
  expected.T(2, 0) = 0.0;
  expected.T(1, 2) = 0.0;
  expected.T(2, 1) = 0.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim3Elastic_PureVolumetric) {
  // Rkappa nonzero only: shifts all three normal stress components
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rkappa(j) = 1.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(1, 1) = 0.0;
  stress.T(2, 2) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(0, 2) = 0.0;
  stress.T(2, 0) = 0.0;
  stress.T(1, 2) = 0.0;
  stress.T(2, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // R_kappa_sum = 3, all deviatoric sums = 0
  // T(0,0) -= 0 + 3 = -3
  // T(1,1) -= 0 + 3 = -3
  // T(2,2) += 0 + 0 - 3 = -3
  StressType expected;
  expected.T(0, 0) = -3.0;
  expected.T(1, 1) = -3.0;
  expected.T(2, 2) = -3.0;
  expected.T(0, 1) = 0.0;
  expected.T(1, 0) = 0.0;
  expected.T(0, 2) = 0.0;
  expected.T(2, 0) = 0.0;
  expected.T(1, 2) = 0.0;
  expected.T(2, 1) = 0.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim3Elastic_PureShear) {
  // Rxy, Rxz, Ryz nonzero only: only off-diagonal T entries change
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS; // 3

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxy(j) = 1.0;
    att.Rxz(j) = 2.0;
    att.Ryz(j) = 3.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(1, 1) = 0.0;
  stress.T(2, 2) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(0, 2) = 0.0;
  stress.T(2, 0) = 0.0;
  stress.T(1, 2) = 0.0;
  stress.T(2, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  // Sums: R_xy=3, R_xz=6, R_yz=9; normal unchanged
  StressType expected;
  expected.T(0, 0) = 0.0;
  expected.T(1, 1) = 0.0;
  expected.T(2, 2) = 0.0;
  expected.T(0, 1) = -3.0;
  expected.T(1, 0) = -3.0;
  expected.T(0, 2) = -6.0;
  expected.T(2, 0) = -6.0;
  expected.T(1, 2) = -9.0;
  expected.T(2, 1) = -9.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

TEST(AttenuationRelaxation, Dim3Elastic_AllNonzero) {
  // Rxx=1, Ryy=2, Rxy=3, Rxz=4, Ryz=5, Rkappa=6 for all j
  // Sums (N=3): R_xx=3, R_yy=6, R_xy=9, R_xz=12, R_yz=15, R_kappa=18
  // Initial T = 0 everywhere
  //
  // T(0,0) -= 3 + 18 = -21
  // T(1,1) -= 6 + 18 = -24
  // T(2,2) += 3 + 6 - 18 = -9
  // T(0,1) = T(1,0) -= 9  => -9
  // T(0,2) = T(2,0) -= 12 => -12
  // T(1,2) = T(2,1) -= 15 => -15
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
      specfem::element::attenuation_tag::constant_isotropic, false>;
  using StressType = specfem::point::stress<Tags>;

  constexpr int N = specfem::constants::N_SLS;

  AttType att;
  for (int j = 0; j < N; ++j) {
    att.Rxx(j) = 1.0;
    att.Ryy(j) = 2.0;
    att.Rxy(j) = 3.0;
    att.Rxz(j) = 4.0;
    att.Ryz(j) = 5.0;
    att.Rkappa(j) = 6.0;
  }

  StressType stress;
  stress.T(0, 0) = 0.0;
  stress.T(1, 1) = 0.0;
  stress.T(2, 2) = 0.0;
  stress.T(0, 1) = 0.0;
  stress.T(1, 0) = 0.0;
  stress.T(0, 2) = 0.0;
  stress.T(2, 0) = 0.0;
  stress.T(1, 2) = 0.0;
  stress.T(2, 1) = 0.0;

  specfem::medium_physics::impl_add_relaxation_to_stress<Tags>(att, stress);

  StressType expected;
  expected.T(0, 0) = -21.0;
  expected.T(1, 1) = -24.0;
  expected.T(2, 2) = -9.0;
  expected.T(0, 1) = -9.0;
  expected.T(1, 0) = -9.0;
  expected.T(0, 2) = -12.0;
  expected.T(2, 0) = -12.0;
  expected.T(1, 2) = -15.0;
  expected.T(2, 1) = -15.0;

  std::ostringstream msg;
  msg << "Computed: " << stress.print() << "\nExpected: " << expected.print();
  EXPECT_TRUE(stress == expected) << msg.str();
}

// -------------------------------------------------------------------------
// impl_integrate_memory_variables — dim3 elastic
// -------------------------------------------------------------------------

TEST(AttenuationIntegrate, Dim3Elastic_ZeroState) {
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
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
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      du.du(i, k) = 0.0;
      dv.du(i, k) = 0.0;
    }

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_EQ(att.Rxx(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Ryy(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Rxy(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Rxz(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Ryz(j), static_cast<type_real>(0.0));
    EXPECT_EQ(att.Rkappa(j), static_cast<type_real>(0.0));
  }
}

TEST(AttenuationIntegrate, Dim3Elastic_SingleStepRecurrence) {
  // Sn: du(0,0)=3, all others=0; du = Sn, dv = 0, dt = 1
  // alpha=beta=gamma=0.5, mu_rate=kappa_rate=1, R_init=0
  //
  // trace_Sn = trace_Snp1 = 3
  // epsilondev_xx = 3 - 3/3 = 2
  // epsilondev_yy = 0 - 1   = -1
  // epsilondev_xy = epsilondev_xz = epsilondev_yz = 0
  //
  // Rxx_new    = 1*(0.5*2 + 0.5*2)    = 2
  // Ryy_new    = 1*(0.5*(-1) + 0.5*(-1)) = -1
  // Rxy/xz/yz  = 0
  // Rkappa_new = 1*(0.5*3 + 0.5*3)    = 3
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
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
  // Sn stored in epsilon fields
  att.epsilon_xx = 3.0;
  // all other epsilon_* = 0 (default constructor)

  FieldDerivType du, dv;
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      du.du(i, k) = 0.0;
      dv.du(i, k) = 0.0;
    }
  du.du(0, 0) = 3.0;

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_NEAR(static_cast<double>(att.Rxx(j)), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Ryy(j)), -1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxy(j)), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxz(j)), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Ryz(j)), 0.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rkappa(j)), 3.0, 1e-6);
  }
  // Snp1 written back: du_att = du (dv=0), so epsilon_xx=3, others=0
  EXPECT_NEAR(static_cast<double>(att.epsilon_xx), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(att.epsilon_yy), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(att.epsilon_zz), 0.0, 1e-6);
}

TEST(AttenuationIntegrate, Dim3Elastic_AlphaDecay) {
  // beta = gamma = 0 => R_new = alpha * R_old (pure exponential decay)
  // alpha = 0.5
  // R_init = {Rxx=2, Ryy=4, Rxy=6, Rxz=8, Ryz=10, Rkappa=12}
  // => R_new = {1, 2, 3, 4, 5, 6}
  using Tags =
      specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                          specfem::element::medium_tag::elastic,
                          specfem::element::property_tag::isotropic,
                          specfem::element::attenuation_tag::constant_isotropic,
                          false>;

  using AttType = specfem::point::attenuation<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic,
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
    att.Ryy(j) = 4.0;
    att.Rxy(j) = 6.0;
    att.Rxz(j) = 8.0;
    att.Ryz(j) = 10.0;
    att.Rkappa(j) = 12.0;
  }
  // Sn stored in epsilon fields (values don't affect R since beta=gamma=0)
  att.epsilon_xx = 5.0;
  att.epsilon_yy = 5.0;
  att.epsilon_zz = 5.0;
  att.epsilon_xy = 5.0;
  att.epsilon_xz = 5.0;
  att.epsilon_yz = 5.0;

  FieldDerivType du, dv;
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      du.du(i, k) = 5.0;
      dv.du(i, k) = 0.0;
    }

  specfem::medium_physics::impl_integrate_memory_variables<Tags>(att, du, dv,
                                                                 1.0);

  for (int j = 0; j < N; ++j) {
    EXPECT_NEAR(static_cast<double>(att.Rxx(j)), 1.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Ryy(j)), 2.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxy(j)), 3.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rxz(j)), 4.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Ryz(j)), 5.0, 1e-6);
    EXPECT_NEAR(static_cast<double>(att.Rkappa(j)), 6.0, 1e-6);
  }
}

} // namespace
