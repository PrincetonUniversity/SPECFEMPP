#include "specfem/attenuation/compute_integration_factors.hpp"
#include "specfem/utilities/is_close.hpp"
#include "test_macros.hpp"
#include <gtest/gtest.h>

using specfem::attenuation::compute_integration_factors;
using specfem::utilities::is_close;

// =============================================================================
// compute_integration_factors tests
//
// All tests use hand-picked values with analytically known results.
//
// The three Runge-Kutta coefficients for mechanism j are
// (Savage et al. BSSA 2010, eq. 11):
//
//   alpha[j] = 1 - dt/tau_sigma
//                + dt^2 / (2*tau_sigma^2)
//                - dt^3 / (6*tau_sigma^3)
//                + dt^4 / (24*tau_sigma^4)
//
//   beta[j]  = dt/2
//                - dt^2 / (3*tau_sigma)
//                + dt^3 / (8*tau_sigma^2)
//                - dt^4 / (24*tau_sigma^3)
//
//   gamma[j] = dt/2
//                - dt^2 / (6*tau_sigma)
//                + dt^3 / (24*tau_sigma^2)
//
// The alternating signs are the Taylor expansion of e^{-dt/tau_sigma}.
// =============================================================================

// Zero timestep:
//   dt = 0 => all dt terms vanish.
//   alpha[j] = 1 for all j
//   beta[j]  = 0 for all j
//   gamma[j] = 0 for all j
TEST(Attenuation_IntegrationFactors, ZeroTimestep) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");

  tau_sigma(0) = 1.0;
  tau_sigma(1) = 2.0;
  tau_sigma(2) = 5.0;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 0.0);

  EXPECT_EQ(result.alpha.extent(0), N_SLS);
  EXPECT_EQ(result.beta.extent(0), N_SLS);
  EXPECT_EQ(result.gamma.extent(0), N_SLS);

  for (int j = 0; j < N_SLS; ++j) {
    EXPECT_TRUE(is_close(result.alpha(j), static_cast<type_real>(1.0)))
        << "alpha(" << j
        << "): " << expected_got(static_cast<type_real>(1.0), result.alpha(j));
    EXPECT_TRUE(is_close(result.beta(j), static_cast<type_real>(0.0)))
        << "beta(" << j
        << "): " << expected_got(static_cast<type_real>(0.0), result.beta(j));
    EXPECT_TRUE(is_close(result.gamma(j), static_cast<type_real>(0.0)))
        << "gamma(" << j
        << "): " << expected_got(static_cast<type_real>(0.0), result.gamma(j));
  }
}

// Single SLS, dt = 1, tau_sigma = 1  (dt/tau_sigma = 1):
//   alpha = 1 - 1 + 1/2 - 1/6 + 1/24 = 9/24 = 3/8
//   beta  = 1/2 - 1/3 + 1/8 - 1/24   = 6/24 = 1/4
//   gamma = 1/2 - 1/6 + 1/24          = 9/24 = 3/8
TEST(Attenuation_IntegrationFactors, SingleSLS_UnitTauUnitDt) {
  constexpr int N_SLS = 1;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");
  tau_sigma(0) = 1.0;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 1.0);

  EXPECT_TRUE(is_close(result.alpha(0), static_cast<type_real>(3.0 / 8.0)))
      << expected_got(static_cast<type_real>(3.0 / 8.0), result.alpha(0));
  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(1.0 / 4.0)))
      << expected_got(static_cast<type_real>(1.0 / 4.0), result.beta(0));
  EXPECT_TRUE(is_close(result.gamma(0), static_cast<type_real>(3.0 / 8.0)))
      << expected_got(static_cast<type_real>(3.0 / 8.0), result.gamma(0));
}

// Single SLS, dt = 2, tau_sigma = 1  (dt/tau_sigma = 2):
//   dt=2, dt^2=4, dt^3=8, dt^4=16, 1/tau_sigma=1
//   alpha = 1 - 2 + 4/2 - 8/6 + 16/24 = 1 - 2 + 2 - 4/3 + 2/3 = 1/3
//   beta  = 1 - 4/3 + 8/8 - 16/24     = 1 - 4/3 + 1 - 2/3 = 0
//   gamma = 1 - 4/6 + 8/24             = 1 - 2/3 + 1/3     = 2/3
TEST(Attenuation_IntegrationFactors, SingleSLS_DoubleDt) {
  constexpr int N_SLS = 1;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");
  tau_sigma(0) = 1.0;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 2.0);

  EXPECT_TRUE(is_close(result.alpha(0), static_cast<type_real>(1.0 / 3.0)))
      << expected_got(static_cast<type_real>(1.0 / 3.0), result.alpha(0));
  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(0.0)))
      << expected_got(static_cast<type_real>(0.0), result.beta(0));
  EXPECT_TRUE(is_close(result.gamma(0), static_cast<type_real>(2.0 / 3.0)))
      << expected_got(static_cast<type_real>(2.0 / 3.0), result.gamma(0));
}

// Three SLS, all identical tau_sigma = 0.1, dt = 0.1
//   (dt/tau_sigma = 1 for every mechanism =>
//    same Taylor coefficients as SingleSLS_UnitTauUnitDt, but scaled):
//
//   1/tau_sigma = 10, dt = 0.1 => dt/tau_sigma = 1
//
//   alpha[j] = 1 - 1 + 1/2 - 1/6 + 1/24  = 9/24 = 3/8
//   beta[j]  = 0.1*(1/2 - 1/3 + 1/8 - 1/24)
//     = 0.1 * 6/24 = 0.1/4               = 1/40
//   gamma[j] = 0.1*(1/2 - 1/6 + 1/24)
//     = 0.1 * 9/24 = 0.1 * 3/8           = 3/80
TEST(Attenuation_IntegrationFactors, ThreeSLS_UniformTau) {
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");
  for (int j = 0; j < N_SLS; ++j)
    tau_sigma(j) = 0.1;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 0.1);

  EXPECT_EQ(result.alpha.extent(0), N_SLS);
  EXPECT_EQ(result.beta.extent(0), N_SLS);
  EXPECT_EQ(result.gamma.extent(0), N_SLS);

  for (int j = 0; j < N_SLS; ++j) {
    EXPECT_TRUE(is_close(result.alpha(j), static_cast<type_real>(3.0 / 8.0)))
        << "alpha(" << j << "): "
        << expected_got(static_cast<type_real>(3.0 / 8.0), result.alpha(j));
    EXPECT_TRUE(is_close(result.beta(j), static_cast<type_real>(1.0 / 40.0)))
        << "beta(" << j << "): "
        << expected_got(static_cast<type_real>(1.0 / 40.0), result.beta(j));
    EXPECT_TRUE(is_close(result.gamma(j), static_cast<type_real>(3.0 / 80.0)))
        << "gamma(" << j << "): "
        << expected_got(static_cast<type_real>(3.0 / 80.0), result.gamma(j));
  }
}

// Two SLS with distinct tau_sigma, dt = 0.1:
//   tau_sigma = [0.1, 1.0]
//
//   Mechanism 0 (tau=0.1, product dt/tau=1 — same as ThreeSLS_UniformTau):
//     alpha[0] = 3/8
//     beta[0]  = 1/40
//     gamma[0] = 3/80
//
//   Mechanism 1 (tau=1.0, 1/tau=1):
//     dt=0.1, dt^2=0.01, dt^3=0.001, dt^4=0.0001
//     alpha[1] = 1 - 0.1 + 0.01/2 - 0.001/6 + 0.0001/24
//              = (240000 - 24000 + 1200 - 40 + 1) / 240000
//              = 217161 / 240000
//     beta[1]  = 0.05 - 0.01/3 + 0.001/8 - 0.0001/24
//              = (12000 - 800 + 30 - 1) / 240000
//              = 11229 / 240000
//     gamma[1] = 0.05 - 0.01/6 + 0.001/24
//              = (1200 - 40 + 1) / 24000
//              = 1161 / 24000
TEST(Attenuation_IntegrationFactors, TwoSLS_DistinctTau) {
  constexpr int N_SLS = 2;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");
  tau_sigma(0) = 0.1;
  tau_sigma(1) = 1.0;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 0.1);

  // Mechanism 0
  EXPECT_TRUE(is_close(result.alpha(0), static_cast<type_real>(3.0 / 8.0)))
      << expected_got(static_cast<type_real>(3.0 / 8.0), result.alpha(0));
  EXPECT_TRUE(is_close(result.beta(0), static_cast<type_real>(1.0 / 40.0)))
      << expected_got(static_cast<type_real>(1.0 / 40.0), result.beta(0));
  EXPECT_TRUE(is_close(result.gamma(0), static_cast<type_real>(3.0 / 80.0)))
      << expected_got(static_cast<type_real>(3.0 / 80.0), result.gamma(0));

  // Mechanism 1
  EXPECT_TRUE(
      is_close(result.alpha(1), static_cast<type_real>(217161.0 / 240000.0)))
      << expected_got(static_cast<type_real>(217161.0 / 240000.0),
                      result.alpha(1));
  EXPECT_TRUE(
      is_close(result.beta(1), static_cast<type_real>(11229.0 / 240000.0)))
      << expected_got(static_cast<type_real>(11229.0 / 240000.0),
                      result.beta(1));
  EXPECT_TRUE(
      is_close(result.gamma(1), static_cast<type_real>(1161.0 / 24000.0)))
      << expected_got(static_cast<type_real>(1161.0 / 24000.0),
                      result.gamma(1));
}

// Five SLS with N_SLS=5, dt=1, all tau_sigma=2  (dt/tau_sigma=0.5):
//   1/tau_sigma = 0.5
//   alpha = 1 - 0.5 + 0.25/2 - 0.125/6 + 0.0625/24
//         = 1 - 1/2 + 1/8 - 1/48 + 1/384
//         = (384 - 192 + 48 - 8 + 1) / 384
//         = 233/384
//   beta  = 1/2 - 0.5/3 + 0.25/8 - 0.125/24
//         = 1/2 - 1/6 + 1/32 - 1/192
//         = (96 - 32 + 6 - 1) / 192
//         = 69/192 = 23/64
//   gamma = 1/2 - 0.5/6 + 0.25/24
//         = 1/2 - 1/12 + 1/96
//         = (48 - 8 + 1) / 96
//         = 41/96
TEST(Attenuation_IntegrationFactors, FiveSLS_UniformTauHalf) {
  constexpr int N_SLS = 5;

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma("tau_sigma");
  for (int j = 0; j < N_SLS; ++j)
    tau_sigma(j) = 2.0;

  auto result = compute_integration_factors<N_SLS>(tau_sigma, 1.0);

  EXPECT_EQ(result.alpha.extent(0), N_SLS);
  EXPECT_EQ(result.beta.extent(0), N_SLS);
  EXPECT_EQ(result.gamma.extent(0), N_SLS);

  for (int j = 0; j < N_SLS; ++j) {
    EXPECT_TRUE(
        is_close(result.alpha(j), static_cast<type_real>(233.0 / 384.0)))
        << "alpha(" << j << "): "
        << expected_got(static_cast<type_real>(233.0 / 384.0), result.alpha(j));
    EXPECT_TRUE(is_close(result.beta(j), static_cast<type_real>(23.0 / 64.0)))
        << "beta(" << j << "): "
        << expected_got(static_cast<type_real>(23.0 / 64.0), result.beta(j));
    EXPECT_TRUE(is_close(result.gamma(j), static_cast<type_real>(41.0 / 96.0)))
        << "gamma(" << j << "): "
        << expected_got(static_cast<type_real>(41.0 / 96.0), result.gamma(j));
  }
}
