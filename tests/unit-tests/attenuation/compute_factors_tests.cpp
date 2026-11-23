#include "specfem/attenuation.hpp"
#include "specfem/attenuation/compute_factors.tpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::get_attenuation_modulus_scales;
using specfem::attenuation::get_attenuation_property_values;
using specfem::utilities::is_close;

// =============================================================================
// get_attenuation_property_values tests
//
// All tests use hand-picked values with analytically known results.
// =============================================================================

// Simple integer ratios:
//   tau_s = [1, 2, 5], tau_eps = [2, 3, 10]
//   ratio = [2, 1.5, 2]
//   beta  = [1, 0.5, 1] / 3  ((ratio - 1) / N_SLS)
//   unrelaxed_ratio = 1 + sum(beta) = 5.5 / 3
TEST(Attenuation_PropertyValues, SimpleIntegerRatios) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  tau_s(0) = 1.0;
  tau_s(1) = 2.0;
  tau_s(2) = 5.0;
  tau_eps(0) = 2.0;
  tau_eps(1) = 3.0;
  tau_eps(2) = 10.0;

  auto result = get_attenuation_property_values<N_SLS>(tau_s, tau_eps);

  EXPECT_EQ(result.beta.extent(0), N_SLS);
  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(1.0 / 3)))
      << expected_got(static_cast<type_real>(1.0 / 3), result.beta(0));
  EXPECT_TRUE(is_close(result.beta(1), static_cast<type_real>(0.5 / 3)))
      << expected_got(static_cast<type_real>(0.5 / 3), result.beta(1));
  EXPECT_TRUE(is_close(result.beta(2), static_cast<type_real>(1.0 / 3)))
      << expected_got(static_cast<type_real>(1.0 / 3), result.beta(2));
  EXPECT_TRUE(is_close(result.unrelaxed_ratio, static_cast<type_real>(5.5 / 3)))
      << expected_got(static_cast<type_real>(5.5 / 3), result.unrelaxed_ratio);
}

// Uniform ratio:
//   tau_s = [2, 4, 5], tau_eps = [4, 8, 10]
//   ratio = [2, 2, 2]
//   beta  = [1, 1, 1] / 3
//   unrelaxed_ratio = 2
TEST(Attenuation_PropertyValues, UniformRatio) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  tau_s(0) = 2.0;
  tau_s(1) = 4.0;
  tau_s(2) = 5.0;
  tau_eps(0) = 4.0;
  tau_eps(1) = 8.0;
  tau_eps(2) = 10.0;

  auto result = get_attenuation_property_values<N_SLS>(tau_s, tau_eps);

  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(1.0 / 3)))
      << expected_got(static_cast<type_real>(1.0 / 3), result.beta(0));
  EXPECT_TRUE(is_close(result.beta(1), static_cast<type_real>(1.0 / 3)))
      << expected_got(static_cast<type_real>(1.0 / 3), result.beta(1));
  EXPECT_TRUE(is_close(result.beta(2), static_cast<type_real>(1.0 / 3)))
      << expected_got(static_cast<type_real>(1.0 / 3), result.beta(2));
  EXPECT_TRUE(is_close(result.unrelaxed_ratio, static_cast<type_real>(2.0)))
      << expected_got(static_cast<type_real>(2.0), result.unrelaxed_ratio);
}

// No attenuation (tau_eps = tau_s):
//   ratio = [1, 1, 1]
//   beta  = [0, 0, 0]
//   unrelaxed_ratio = 1
TEST(Attenuation_PropertyValues, NoAttenuation) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  tau_s(0) = 1.0;
  tau_s(1) = 2.0;
  tau_s(2) = 5.0;
  tau_eps(0) = 1.0;
  tau_eps(1) = 2.0;
  tau_eps(2) = 5.0;

  auto result = get_attenuation_property_values<N_SLS>(tau_s, tau_eps);

  for (int i = 0; i < N_SLS; ++i) {
    EXPECT_TRUE(is_close(result.beta(i), static_cast<type_real>(0.0)))
        << "beta(" << i << ") should be 0 with no attenuation: "
        << expected_got(static_cast<type_real>(0.0), result.beta(i));
  }
  EXPECT_TRUE(is_close(result.unrelaxed_ratio, static_cast<type_real>(1.0)))
      << expected_got(static_cast<type_real>(1.0), result.unrelaxed_ratio);
}

// N_SLS = 2:
//   tau_s = [2, 5], tau_eps = [6, 10]
//   ratio = [3, 2]
//   beta  = [2, 1] / 2
//   unrelaxed_ratio = 2.5
TEST(Attenuation_PropertyValues, TwoSLS) {
  constexpr int N_SLS = 2;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  tau_s(0) = 2.0;
  tau_s(1) = 5.0;
  tau_eps(0) = 6.0;
  tau_eps(1) = 10.0;

  auto result = get_attenuation_property_values<N_SLS>(tau_s, tau_eps);

  EXPECT_EQ(result.beta.extent(0), 2);
  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(1.0)))
      << expected_got(static_cast<type_real>(1.0), result.beta(0));
  EXPECT_TRUE(is_close(result.beta(1), static_cast<type_real>(0.5)))
      << expected_got(static_cast<type_real>(0.5), result.beta(1));
  EXPECT_TRUE(is_close(result.unrelaxed_ratio, static_cast<type_real>(2.5)))
      << expected_got(static_cast<type_real>(2.5), result.unrelaxed_ratio);
}

// N_SLS = 5:
//   tau_s = [1, 1, 1, 1, 1], tau_eps = [1.5, 1.5, 1.5, 1.5, 1.5]
//   ratio = [1.5, 1.5, 1.5, 1.5, 1.5]
//   beta  = [0.5, 0.5, 0.5, 0.5, 0.5] / 5
//   unrelaxed_ratio = 1.5
TEST(Attenuation_PropertyValues, FiveSLS) {
  constexpr int N_SLS = 5;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  for (int j = 0; j < N_SLS; ++j) {
    tau_s(j) = 1.0;
    tau_eps(j) = 1.5;
  }

  auto result = get_attenuation_property_values<N_SLS>(tau_s, tau_eps);

  EXPECT_EQ(result.beta.extent(0), 5);
  for (int i = 0; i < N_SLS; ++i) {
    EXPECT_TRUE(is_close(result.beta(i), static_cast<type_real>(0.1)))
        << expected_got(static_cast<type_real>(0.1), result.beta(i));
  }
  EXPECT_TRUE(is_close(result.unrelaxed_ratio, static_cast<type_real>(1.5)))
      << expected_got(static_cast<type_real>(1.5), result.unrelaxed_ratio);
}

// =============================================================================
// get_attenuation_modulus_scales tests
//
// All tests use hand-picked values with analytically known results.
//
// relaxed = factor_scale_mu0 / sum_weighted, ratio = sum_unrelaxed, and the
// total (unrelaxed) scale factor is:
//   scale_factor = factor_scale_mu * factor_scale_mu0
// where:
//   factor_scale_mu0 = 1 + 2*ln(f_c/f_0) / (pi*Q)
//   factor_scale_mu  = sum_unrelaxed / sum_weighted
//   sum_unrelaxed = 1 + sum_i(defect_i / N_SLS)
//   sum_weighted  = 1 + sum_i(defect_i / (1 + 1/(w*tau_i)^2) / N_SLS)
//   defect_i = tau_eps_i/tau_sigma_i - 1
//   w = 2*pi*f_c
// =============================================================================

// No attenuation + f_c = f_0:
//   tau_eps = tau_sigma  =>  defect = 0  =>  factor_scale_mu = 1
//   f_c = f_0            =>  ln(1) = 0   =>  factor_scale_mu0 = 1
//   scale_factor = 1.0 exactly
TEST(Attenuation_ScaleFactor, NoAttenuationSameFreq) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  tau_sigma(0) = 1.0;
  tau_sigma(1) = 2.0;
  tau_sigma(2) = 5.0;
  tau_eps(0) = 1.0;
  tau_eps(1) = 2.0;
  tau_eps(2) = 5.0;

  const auto scales = get_attenuation_modulus_scales<N_SLS>(
      1.0, tau_eps, tau_sigma, 100.0, 1.0);
  const type_real scale_factor = scales.unrelaxed();

  EXPECT_TRUE(is_close(scale_factor, static_cast<type_real>(1.0)))
      << expected_got(static_cast<type_real>(1.0), scale_factor);
}

// Pure log correction (no defect):
//   tau_eps = tau_sigma  =>  factor_scale_mu = 1
//   f_c = e, f_0 = 1, Q = 200
//   factor_scale_mu0 = 1 + 2*ln(e) / (pi*200) = 1 + 2/(200*pi) = 1 + 1/(100*pi)
//   scale_factor = 1 + 1/(100*pi)
TEST(Attenuation_ScaleFactor, PureLogCorrection) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  tau_sigma(0) = 1.0;
  tau_sigma(1) = 2.0;
  tau_sigma(2) = 5.0;
  tau_eps(0) = 1.0;
  tau_eps(1) = 2.0;
  tau_eps(2) = 5.0;

  const type_real e = std::exp(1.0);
  const type_real expected = 1.0 + 1.0 / (100.0 * specfem::constants::pi);

  const auto scales =
      get_attenuation_modulus_scales<N_SLS>(e, tau_eps, tau_sigma, 200.0, 1.0);
  const type_real scale_factor = scales.unrelaxed();

  EXPECT_TRUE(is_close(scale_factor, expected))
      << expected_got(expected, scale_factor);
}

// Uniform defect with omega*tau = 1:
//   N_SLS = 3, all tau_sigma = 1/(2*pi), all tau_eps = 4/(3*2*pi)
//   ratio = 4/3, defect = 1/3
//   f_c = 1, f_0 = 1, Q = 100
//
//   omega_tau = 2*pi * 1 * 1/(2*pi) = 1
//   sum_unrelaxed = 1 + (1/3)/3 * 3 = 1 + 1/3 = 4/3
//   weight = 1/(1 + 1/1) = 1/2
//   sum_weighted = 1 + (1/3)*(1/2)/3 * 3 = 1 + 1/6 = 7/6
//   factor_scale_mu = (4/3) / (7/6) = 8/7
//   factor_scale_mu0 = 1 (since f_c = f_0)
//   scale_factor = 8/7
TEST(Attenuation_ScaleFactor, UniformDefectOmegaTauOne) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  const type_real tau_s_val = 1.0 / (2.0 * specfem::constants::pi);
  const type_real tau_e_val = 4.0 / (3.0 * 2.0 * specfem::constants::pi);

  for (int j = 0; j < N_SLS; ++j) {
    tau_sigma(j) = tau_s_val;
    tau_eps(j) = tau_e_val;
  }

  const auto scales = get_attenuation_modulus_scales<N_SLS>(
      1.0, tau_eps, tau_sigma, 100.0, 1.0);
  const type_real scale_factor = scales.unrelaxed();

  const type_real expected = static_cast<type_real>(8.0 / 7.0);
  EXPECT_TRUE(is_close(scale_factor, expected))
      << expected_got(expected, scale_factor);
  // M_U/M_R = 4/3 and M_R/M_0 = 1/A(omega_c) = 6/7.
  EXPECT_NEAR(scales.ratio, 4.0 / 3.0, 1e-6);
  EXPECT_NEAR(scales.relaxed, 6.0 / 7.0, 1e-6);
}

// Combined log correction and defect:
//   Same uniform defect as above (scale_mu = 8/7)
//   but now f_c = e, f_0 = 1, Q = 200
//   factor_scale_mu0 = 1 + 1/(100*pi)
//   scale_factor = (8/7) * (1 + 1/(100*pi))
TEST(Attenuation_ScaleFactor, CombinedLogAndDefect) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  const type_real tau_s_val = 1.0 / (2.0 * specfem::constants::pi);
  const type_real tau_e_val = 4.0 / (3.0 * 2.0 * specfem::constants::pi);

  for (int j = 0; j < N_SLS; ++j) {
    tau_sigma(j) = tau_s_val;
    tau_eps(j) = tau_e_val;
  }

  const type_real e = std::exp(1.0);
  // factor_scale_mu0 uses the same tau_sigma but now f_c = e
  // omega_tau = 2*pi*e*1/(2*pi) = e, so omega_tau^2 = e^2
  // Recompute sum_weighted with omega_tau = e:
  //   weight_denom = 1 + 1/e^2
  //   sum_weighted = 1 + (1/3) / (1 + 1/e^2) = 1 + (1/3)*e^2/(e^2+1)
  //   sum_unrelaxed stays the same: 4/3
  const type_real e2 = e * e;
  const type_real expected_sum_weighted = 1.0 + (1.0 / 3.0) * e2 / (e2 + 1.0);
  const type_real expected_factor_mu = (4.0 / 3.0) / expected_sum_weighted;
  const type_real expected_factor_mu0 =
      1.0 + 1.0 / (100.0 * specfem::constants::pi);
  const type_real expected = expected_factor_mu * expected_factor_mu0;

  const auto scales =
      get_attenuation_modulus_scales<N_SLS>(e, tau_eps, tau_sigma, 200.0, 1.0);
  const type_real scale_factor = scales.unrelaxed();

  EXPECT_TRUE(is_close(scale_factor, expected))
      << expected_got(expected, scale_factor);
}

// N_SLS = 2 with distinct tau_sigma values:
//   tau_sigma = [1, 2], tau_eps = [1.5, 2.5]
//   defect = [0.5, 0.25]
//   f_c = 1/(2*pi), f_0 = 1/(2*pi), Q = 100
//
//   omega = 2*pi * 1/(2*pi) = 1
//   omega_tau_0 = 1*1 = 1,  omega_tau_1 = 1*2 = 2
//   sum_unrelaxed = 1 + (0.5 + 0.25)/2 = 1 + 0.375 = 1.375
//   w0 = 1/(1 + 1/1)  = 0.5,   w1 = 1/(1 + 1/4) = 4/5
//   sum_weighted = 1 + (0.5*0.5 + 0.25*0.8)/2 = 1 + (0.25 + 0.2)/2 = 1 + 0.225
//   = 1.225 factor_scale_mu = 1.375 / 1.225 = 55/49 factor_scale_mu0 = 1 (f_c =
//   f_0) scale_factor = 55/49
TEST(Attenuation_ScaleFactor, TwoSLSDistinctTau) {
  constexpr int N_SLS = 2;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  tau_sigma(0) = 1.0;
  tau_sigma(1) = 2.0;
  tau_eps(0) = 1.5;
  tau_eps(1) = 2.5;

  const type_real f_c = 1.0 / (2.0 * specfem::constants::pi);

  const auto scales = get_attenuation_modulus_scales<N_SLS>(
      f_c, tau_eps, tau_sigma, 100.0, f_c);
  const type_real scale_factor = scales.unrelaxed();

  const type_real expected = static_cast<type_real>(55.0 / 49.0);
  EXPECT_TRUE(is_close(scale_factor, expected))
      << expected_got(expected, scale_factor);
  EXPECT_NEAR(scales.ratio, 1.375, 1e-6);
  EXPECT_NEAR(scales.relaxed, 1.0 / 1.225, 1e-6);
}

// Throws when scale_factor > 1.5:
//   tau_eps = tau_sigma (factor_scale_mu = 1)
//   f_c = 10, f_0 = 1, Q = 2
//   factor_scale_mu0 = 1 + 2*ln(10)/(pi*2) ≈ 1.733 > 1.5
TEST(Attenuation_ScaleFactor, ThrowsWhenOutOfRange) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  for (int j = 0; j < N_SLS; ++j) {
    tau_sigma(j) = 1.0;
    tau_eps(j) = 1.0;
  }

  // factor_scale_mu0 = 1 + 2*ln(10)/(pi*2) ≈ 1.733 > 1.5
  EXPECT_THROW(
      get_attenuation_modulus_scales<N_SLS>(10.0, tau_eps, tau_sigma, 2.0, 1.0),
      std::runtime_error);
}
