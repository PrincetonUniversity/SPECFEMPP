#include "specfem/attenuation.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::compute_tau_sigma;

// Test that the function returns the correct number of elements
TEST(Attenuation_ComputeTauSigma, ReturnsCorrectSize) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_s_3 = compute_tau_sigma<3>(min_period, max_period);
  EXPECT_EQ(tau_s_3.extent(0), 3);

  auto tau_s_5 = compute_tau_sigma<5>(min_period, max_period);
  EXPECT_EQ(tau_s_5.extent(0), 5);
}

// Test that tau_sigma values are positive
TEST(Attenuation_ComputeTauSigma, ValuesArePositive) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_s = compute_tau_sigma<3>(min_period, max_period);

  for (int i = 0; i < 3; ++i) {
    EXPECT_GT(tau_s(i), 0.0) << "tau_s(" << i << ") should be positive";
  }
}

// Test that tau_sigma values are monotonically decreasing
// (higher frequency = smaller tau)
TEST(Attenuation_ComputeTauSigma, ValuesAreMonotonicallyDecreasing) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_s = compute_tau_sigma<5>(min_period, max_period);

  for (int i = 1; i < 5; ++i) {
    EXPECT_LT(tau_s(i), tau_s(i - 1))
        << "tau_s(" << i << ") should be less than tau_s(" << i - 1 << ")";
  }
}

// Test that tau_sigma values are equally spaced in log10 frequency
TEST(Attenuation_ComputeTauSigma, EquallySpacedInLog10Frequency) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;
  constexpr int N_SLS = 5;

  auto tau_s = compute_tau_sigma<N_SLS>(min_period, max_period);

  // Convert tau_s to frequencies: f = 1 / (2 * pi * tau_s)
  // Then check that log10(f) values are equally spaced
  std::array<type_real, N_SLS> log_freq;
  for (int i = 0; i < N_SLS; ++i) {
    type_real freq = 1.0 / (2.0 * pi * tau_s(i));
    log_freq[i] = std::log10(freq);
  }

  // Check that differences between consecutive log10(freq) are equal
  type_real expected_spacing = log_freq[1] - log_freq[0];
  for (int i = 2; i < N_SLS; ++i) {
    type_real actual_spacing = log_freq[i] - log_freq[i - 1];
    EXPECT_NEAR(actual_spacing, expected_spacing, 1e-10)
        << "Log10 frequency spacing should be constant";
  }
}

// Test boundary values match expected frequencies
TEST(Attenuation_ComputeTauSigma, BoundaryFrequenciesMatch) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;
  constexpr int N_SLS = 3;

  auto tau_s = compute_tau_sigma<N_SLS>(min_period, max_period);

  // Expected frequencies at boundaries
  type_real f_min = 1.0 / max_period; // 0.1 Hz
  type_real f_max = 1.0 / min_period; // 100 Hz

  // Convert tau_s to frequency: f = 1 / (2 * pi * tau_s)
  type_real freq_first = 1.0 / (2.0 * pi * tau_s(0));
  type_real freq_last = 1.0 / (2.0 * pi * tau_s(N_SLS - 1));

  // First tau_s corresponds to f_min, last to f_max
  EXPECT_NEAR(freq_first, f_min, f_min * 1e-10)
      << "First frequency should match f_min";
  EXPECT_NEAR(freq_last, f_max, f_max * 1e-10)
      << "Last frequency should match f_max";
}

// Test with different period ranges
TEST(Attenuation_ComputeTauSigma, DifferentPeriodRanges) {
  // Narrow range
  {
    auto tau_s = compute_tau_sigma<3>(1.0, 10.0);
    EXPECT_GT(tau_s(0), 0.0);
    EXPECT_GT(tau_s(1), 0.0);
    EXPECT_GT(tau_s(2), 0.0);
  }

  // Wide range
  {
    auto tau_s = compute_tau_sigma<3>(0.001, 100.0);
    EXPECT_GT(tau_s(0), 0.0);
    EXPECT_GT(tau_s(1), 0.0);
    EXPECT_GT(tau_s(2), 0.0);
  }
}

// Test N_SLS = 2 edge case (minimum valid value)
// Note: N_SLS=1 causes division by zero in dexpval calculation
TEST(Attenuation_ComputeTauSigma, TwoSLS) {
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_s = compute_tau_sigma<2>(min_period, max_period);

  EXPECT_EQ(tau_s.extent(0), 2);
  EXPECT_GT(tau_s(0), 0.0);
  EXPECT_GT(tau_s(1), 0.0);

  // First should correspond to f_min, second to f_max
  type_real f_min = 1.0 / max_period;
  type_real f_max = 1.0 / min_period;
  type_real freq_first = 1.0 / (2.0 * pi * tau_s(0));
  type_real freq_last = 1.0 / (2.0 * pi * tau_s(1));

  EXPECT_NEAR(freq_first, f_min, f_min * 1e-10);
  EXPECT_NEAR(freq_last, f_max, f_max * 1e-10);
}
