#include "specfem/attenuation/maxwell.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::maxwell;
using specfem::attenuation::MaxwellFactors;

// Test that A and B have correct sizes
TEST(Attenuation_Maxwell, ReturnsCorrectSize) {
  constexpr int NF = 10;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  // Fill with dummy values
  for (int i = 0; i < NF; ++i) {
    f(i) = 0.1 + i * 0.1; // 0.1 to 1.0 Hz
  }
  for (int j = 0; j < N_SLS; ++j) {
    tau_s(j) = 0.1 * (j + 1);
    tau_eps(j) = tau_s(j) * 1.1; // tau_eps > tau_s
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  EXPECT_EQ(result.real.extent(0), NF);
  EXPECT_EQ(result.imag.extent(0), NF);
}

// Test that A values are positive (real modulus > 0)
TEST(Attenuation_Maxwell, AValuesPositive) {
  constexpr int NF = 100;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  // Frequency range similar to seismic applications
  for (int i = 0; i < NF; ++i) {
    f(i) = std::pow(10.0, -2.0 + i * 4.0 / (NF - 1)); // 0.01 to 100 Hz
  }

  // Set up SLS parameters
  tau_s(0) = 0.1;
  tau_s(1) = 1.0;
  tau_s(2) = 10.0;
  for (int j = 0; j < N_SLS; ++j) {
    tau_eps(j) = tau_s(j) * 1.2;
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  for (int i = 0; i < NF; ++i) {
    EXPECT_GT(result.real(i), 0.0) << "real(" << i << ") should be positive";
  }
}

// Test that B values are positive when tau_eps > tau_s
TEST(Attenuation_Maxwell, BValuesPositiveWhenTauEpsGreater) {
  constexpr int NF = 50;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  for (int i = 0; i < NF; ++i) {
    f(i) = std::pow(10.0, -1.0 + i * 2.0 / (NF - 1));
  }

  tau_s(0) = 0.1;
  tau_s(1) = 1.0;
  tau_s(2) = 10.0;
  for (int j = 0; j < N_SLS; ++j) {
    tau_eps(j) = tau_s(j) * 1.5; // tau_eps > tau_s for positive B
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  for (int i = 0; i < NF; ++i) {
    EXPECT_GT(result.imag(i), 0.0) << "imag(" << i << ") should be positive";
  }
}

// Test that tan_delta = B/A gives approximately 1/Q
// For a well-designed SLS, Q should be approximately constant over bandwidth
TEST(Attenuation_Maxwell, TanDeltaRelationship) {
  constexpr int NF = 50;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  // Use frequencies that are centered in the SLS bandwidth
  for (int i = 0; i < NF; ++i) {
    f(i) = std::pow(10.0, -1.0 + i * 2.0 / (NF - 1));
  }

  tau_s(0) = 0.1;
  tau_s(1) = 1.0;
  tau_s(2) = 10.0;
  for (int j = 0; j < N_SLS; ++j) {
    tau_eps(j) = tau_s(j) * 1.1;
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  // Check that tan_delta = B/A is well-defined
  for (int i = 0; i < NF; ++i) {
    type_real tan_delta = result.imag(i) / result.real(i);
    type_real Q_computed = result.real(i) / result.imag(i);

    EXPECT_GT(tan_delta, 0.0) << "tan_delta should be positive";
    EXPECT_GT(Q_computed, 0.0) << "Q should be positive";
    EXPECT_TRUE(std::isfinite(tan_delta)) << "tan_delta should be finite";
    EXPECT_TRUE(std::isfinite(Q_computed)) << "Q should be finite";
  }
}

// Test with single SLS
TEST(Attenuation_Maxwell, SingleSLS) {
  constexpr int NF = 20;
  constexpr int N_SLS = 1;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  for (int i = 0; i < NF; ++i) {
    f(i) = 0.1 * (i + 1);
  }

  tau_s(0) = 1.0;
  tau_eps(0) = 1.5;

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  // For single SLS, verify the Fortran formula directly:
  // A = (1 + w^2 * tau_eps * tau_s) / denom
  // B = w * (tau_eps - tau_s) / denom
  for (int i = 0; i < NF; ++i) {
    type_real w = 2.0 * pi * f(i);
    type_real w2 = w * w;
    type_real denom = 1.0 + w2 * tau_s(0) * tau_s(0);

    type_real expected_A = (1.0 + w2 * tau_eps(0) * tau_s(0)) / denom;
    type_real expected_B = w * (tau_eps(0) - tau_s(0)) / denom;

    EXPECT_NEAR(result.real(i), expected_A, 1e-10)
        << "real(" << i << ") should match analytical formula";
    EXPECT_NEAR(result.imag(i), expected_B, 1e-10)
        << "imag(" << i << ") should match analytical formula";
  }
}

// Test low frequency limit: A → 1, B → 0 as w → 0 (with 1/L normalization)
TEST(Attenuation_Maxwell, LowFrequencyLimit) {
  constexpr int NF = 5;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  // Very low frequencies
  for (int i = 0; i < NF; ++i) {
    f(i) = 1e-6 * (i + 1); // µHz range
  }

  tau_s(0) = 0.1;
  tau_s(1) = 1.0;
  tau_s(2) = 10.0;
  for (int j = 0; j < N_SLS; ++j) {
    tau_eps(j) = tau_s(j) * 1.2;
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  // With 1/L normalization (Dahlen & Tromp eq. 43-44):
  // A = (1/L) * Σ[(1 + w²*τε*τs)/(1 + w²*τs²)] → 1 as w → 0
  // B = (1/L) * Σ[w*(τε - τs)/denom] → 0 as w → 0
  for (int i = 0; i < NF; ++i) {
    EXPECT_NEAR(result.real(i), 1.0, 1e-6)
        << "real should approach 1 at low freq";
    // B approaches 0 linearly with w, use tolerance of 1e-4
    EXPECT_NEAR(result.imag(i), 0.0, 1e-4)
        << "imag should approach 0 at low freq";
  }
}

// Test high frequency limit: A approaches tau_eps/tau_s ratio
TEST(Attenuation_Maxwell, HighFrequencyLimit) {
  constexpr int NF = 5;
  constexpr int N_SLS = 1;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  // Very high frequencies
  for (int i = 0; i < NF; ++i) {
    f(i) = 1e6 * (i + 1); // MHz range
  }

  tau_s(0) = 1.0;
  tau_eps(0) = 1.5;

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  // At high frequency limit for single SLS (Fortran formula):
  // A = (1 + w² τ_ε τ_σ) / (1 + w² τ_σ²) → τ_ε / τ_σ as w → ∞
  // B = w (τ_ε - τ_σ) / (1 + w² τ_σ²) → 0 as w → ∞
  type_real expected_A_limit = tau_eps(0) / tau_s(0);

  for (int i = 0; i < NF; ++i) {
    EXPECT_NEAR(result.real(i), expected_A_limit, 1e-3)
        << "real should approach tau_eps/tau_s at high freq";
    EXPECT_NEAR(result.imag(i), 0.0, 1e-3)
        << "imag should approach 0 at high freq";
  }
}

// Test with identical tau_eps and tau_s (no attenuation)
TEST(Attenuation_Maxwell, NoAttenuation) {
  constexpr int NF = 20;
  constexpr int N_SLS = 3;

  Kokkos::View<type_real[NF], Kokkos::LayoutRight, Kokkos::HostSpace> f("f");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_s");
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_eps");

  for (int i = 0; i < NF; ++i) {
    f(i) = std::pow(10.0, -1.0 + i * 2.0 / (NF - 1));
  }

  tau_s(0) = 0.1;
  tau_s(1) = 1.0;
  tau_s(2) = 10.0;
  // tau_eps = tau_s means no attenuation
  for (int j = 0; j < N_SLS; ++j) {
    tau_eps(j) = tau_s(j);
  }

  auto result = maxwell<NF, N_SLS>(f, tau_s, tau_eps);

  // With no attenuation (tau_eps = tau_s) and 1/L normalization:
  // A = (1/L) * Σ[(1 + w²τ²)/(1 + w²τ²)] = (1/L) * L = 1
  // B = (1/L) * Σ[w * 0 / denom] = 0
  // Use looser tolerance for A due to floating point precision
  for (int i = 0; i < NF; ++i) {
    EXPECT_NEAR(result.real(i), 1.0, 1e-6)
        << "real should be 1 with no attenuation";
    EXPECT_NEAR(result.imag(i), 0.0, 1e-10)
        << "imag should be 0 with no attenuation";
  }
}
