#include "specfem/attenuation.hpp"
#include "specfem/attenuation/compute_tau_eps.tpp"
#include "specfem/attenuation/compute_tau_sigma.tpp"
#include "specfem/constants.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::compute_tau_eps;
using specfem::attenuation::compute_tau_sigma;
using specfem::attenuation::maxwell;
using specfem::constants::NF_ATTENUATION;

// Helper function to compute achieved Q from tau_eps and tau_sigma
template <int N_SLS>
type_real compute_achieved_Q(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps,
    type_real min_period, type_real max_period) {

  // Set up evaluation frequencies
  Kokkos::View<type_real[NF_ATTENUATION], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f("frequencies");

  const type_real f1 = 1.0 / max_period;
  const type_real f2 = 1.0 / min_period;
  const type_real log_f1 = std::log10(f1);
  const type_real log_f2 = std::log10(f2);
  const type_real d_log_f =
      (log_f2 - log_f1) / (static_cast<type_real>(NF_ATTENUATION) - 1);

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    f(i) = std::pow(10.0, log_f1 + i * d_log_f);
  }

  // Compute Maxwell moduli
  auto maxwell_factors = specfem::attenuation::maxwell<NF_ATTENUATION, N_SLS>(
      f, tau_sigma, tau_eps);

  // Compute average Q over the frequency range
  type_real avg_inv_Q = 0.0;
  for (int i = 0; i < NF_ATTENUATION; ++i) {
    avg_inv_Q += maxwell_factors.imag(i) / maxwell_factors.real(i);
  }
  avg_inv_Q /= static_cast<type_real>(NF_ATTENUATION);

  return 1.0 / avg_inv_Q;
}

/**
 * @brief Compute the least-squares error for constant-Q approximation
 *
 * Following Savage et al. (2010), the LSQ error is defined as:
 *   error = sqrt( (1/N) * sum_i ( (Q_achieved(f_i) - Q_target) / Q_target )^2 )
 *         = sqrt( (1/N) * sum_i ( (1/Q_target - 1/Q_achieved(f_i)) /
 * (1/Q_target) )^2 )
 *
 * This is the RMS relative error in Q (or equivalently 1/Q) over the frequency
 * band.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_sigma Stress relaxation times
 * @param tau_eps Strain relaxation times
 * @param target_Q Target quality factor
 * @param min_period Minimum period (s)
 * @param max_period Maximum period (s)
 * @return LSQ error as a percentage (multiply by 100)
 */
template <int N_SLS>
type_real compute_lsq_error(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps,
    type_real target_Q, type_real min_period, type_real max_period) {

  // Set up evaluation frequencies
  Kokkos::View<type_real[NF_ATTENUATION], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f("frequencies");

  const type_real f1 = 1.0 / max_period;
  const type_real f2 = 1.0 / min_period;
  const type_real log_f1 = std::log10(f1);
  const type_real log_f2 = std::log10(f2);
  const type_real d_log_f =
      (log_f2 - log_f1) / (static_cast<type_real>(NF_ATTENUATION) - 1);

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    f(i) = std::pow(10.0, log_f1 + i * d_log_f);
  }

  // Compute Maxwell moduli
  auto maxwell_factors = specfem::attenuation::maxwell<NF_ATTENUATION, N_SLS>(
      f, tau_sigma, tau_eps);

  // Compute LSQ error: RMS relative error in 1/Q
  const type_real target_inv_Q = 1.0 / target_Q;
  type_real sum_sq_error = 0.0;

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    type_real achieved_inv_Q =
        maxwell_factors.imag(i) / maxwell_factors.real(i);
    type_real rel_error = (achieved_inv_Q - target_inv_Q) / target_inv_Q;
    sum_sq_error += rel_error * rel_error;
  }

  return std::sqrt(sum_sq_error / static_cast<type_real>(NF_ATTENUATION));
}

/**
 * @brief Compute bandwidth in decades from period range
 */
inline type_real bandwidth_decades(type_real min_period, type_real max_period) {
  return std::log10(max_period / min_period);
}

// =============================================================================
// Savage et al. (2010) Validation Tests
//
// These tests verify that the LSQ error matches the theoretical predictions
// from Savage et al. (2010) "Effects of 3D Attenuation on Seismic Wave
// Amplitude and Phase Measurements", BSSA.
//
// Key finding: For N_SLS standard linear solids with ~1% LSQ error, the
// optimal absorption band widths are approximately:
//   N_SLS=2: ~0.9 decades
//   N_SLS=3: ~1.7 decades
//   N_SLS=4: ~2.5 decades
//   N_SLS=5: ~3.0 decades
// =============================================================================

// Test Savage 2010 prediction: N_SLS=3 with 1.7 decades should give ~1% error
TEST(Attenuation_ComputeTauEps, Savage2010_NSLS3_1p7Decades) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 80.0; // Q=80 as used in Savage 2010

  // 1.7 decades: e.g., 0.1s to 5.01s (log10(5.01/0.1) ≈ 1.7)
  const type_real min_period = 0.1;
  const type_real max_period = min_period * std::pow(10.0, 1.7);

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);
  type_real lsq_error_percent = lsq_error * 100.0;

  // Savage 2010 predicts ~1% error for 1.7 decades with 3 SLS
  // Allow some tolerance for numerical differences
  EXPECT_LT(lsq_error_percent, 2.0)
      << "N_SLS=3 with 1.7 decades should achieve <2% LSQ error (Savage 2010 "
         "predicts ~1%)";
  EXPECT_GT(lsq_error_percent, 0.1)
      << "LSQ error should be non-trivial for 1.7 decades";

  // Print for reference
  std::cout << "N_SLS=3, bandwidth="
            << bandwidth_decades(min_period, max_period)
            << " decades, LSQ error=" << lsq_error_percent << "%" << std::endl;
}

// Test Savage 2010 prediction: N_SLS=2 with 0.9 decades should give ~1% error
TEST(Attenuation_ComputeTauEps, Savage2010_NSLS2_0p9Decades) {
  constexpr int N_SLS = 2;
  constexpr type_real target_Q = 80.0;

  // 0.9 decades: e.g., 0.1s to 0.794s
  const type_real min_period = 0.1;
  const type_real max_period = min_period * std::pow(10.0, 0.9);

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);
  type_real lsq_error_percent = lsq_error * 100.0;

  EXPECT_LT(lsq_error_percent, 2.0)
      << "N_SLS=2 with 0.9 decades should achieve <2% LSQ error";

  std::cout << "N_SLS=2, bandwidth="
            << bandwidth_decades(min_period, max_period)
            << " decades, LSQ error=" << lsq_error_percent << "%" << std::endl;
}

// Test Savage 2010 prediction: N_SLS=4 with 2.5 decades should give ~1% error
TEST(Attenuation_ComputeTauEps, Savage2010_NSLS4_2p5Decades) {
  constexpr int N_SLS = 4;
  constexpr type_real target_Q = 80.0;

  // 2.5 decades: e.g., 0.1s to 31.6s
  const type_real min_period = 0.1;
  const type_real max_period = min_period * std::pow(10.0, 2.5);

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);
  type_real lsq_error_percent = lsq_error * 100.0;

  EXPECT_LT(lsq_error_percent, 2.0)
      << "N_SLS=4 with 2.5 decades should achieve <2% LSQ error";

  std::cout << "N_SLS=4, bandwidth="
            << bandwidth_decades(min_period, max_period)
            << " decades, LSQ error=" << lsq_error_percent << "%" << std::endl;
}

// Test Savage 2010 prediction: N_SLS=5 with 3.0 decades should give ~1% error
TEST(Attenuation_ComputeTauEps, Savage2010_NSLS5_3p0Decades) {
  constexpr int N_SLS = 5;
  constexpr type_real target_Q = 80.0;

  // 3.0 decades: e.g., 0.1s to 100s
  const type_real min_period = 0.1;
  const type_real max_period = min_period * std::pow(10.0, 3.0);

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);
  type_real lsq_error_percent = lsq_error * 100.0;

  EXPECT_LT(lsq_error_percent, 2.0)
      << "N_SLS=5 with 3.0 decades should achieve <2% LSQ error";

  std::cout << "N_SLS=5, bandwidth="
            << bandwidth_decades(min_period, max_period)
            << " decades, LSQ error=" << lsq_error_percent << "%" << std::endl;
}

// Test that wider bandwidth increases error (Savage 2010 curve shape)
TEST(Attenuation_ComputeTauEps, Savage2010_ErrorIncreasesWithBandwidth) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 80.0;
  constexpr type_real min_period = 0.1;

  // Compute error at 1.7 decades (optimal for N_SLS=3)
  type_real max_period_optimal = min_period * std::pow(10.0, 1.7);
  auto tau_sigma_opt = compute_tau_sigma<N_SLS>(min_period, max_period_optimal);
  auto tau_eps_opt = compute_tau_eps<N_SLS>(target_Q, tau_sigma_opt, min_period,
                                            max_period_optimal);
  type_real error_optimal = compute_lsq_error<N_SLS>(
      tau_sigma_opt, tau_eps_opt, target_Q, min_period, max_period_optimal);

  // Compute error at 3.0 decades (too wide for N_SLS=3)
  type_real max_period_wide = min_period * std::pow(10.0, 3.0);
  auto tau_sigma_wide = compute_tau_sigma<N_SLS>(min_period, max_period_wide);
  auto tau_eps_wide = compute_tau_eps<N_SLS>(target_Q, tau_sigma_wide,
                                             min_period, max_period_wide);
  type_real error_wide = compute_lsq_error<N_SLS>(
      tau_sigma_wide, tau_eps_wide, target_Q, min_period, max_period_wide);

  EXPECT_GT(error_wide, error_optimal)
      << "Error should increase when bandwidth exceeds optimal for given N_SLS";

  std::cout << "N_SLS=3: error at 1.7 decades = " << error_optimal * 100.0
            << "%, "
            << "error at 3.0 decades = " << error_wide * 100.0 << "%"
            << std::endl;
}

// Test that more SLS elements reduce error for same bandwidth
TEST(Attenuation_ComputeTauEps, Savage2010_MoreSLSReducesError) {
  constexpr type_real target_Q = 80.0;
  const type_real min_period = 0.1;
  const type_real max_period = min_period * std::pow(10.0, 2.0); // 2 decades

  // N_SLS = 2
  {
    constexpr int N_SLS = 2;
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real error_2 = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);

    // N_SLS = 4
    constexpr int N_SLS_4 = 4;
    auto tau_sigma_4 = compute_tau_sigma<N_SLS_4>(min_period, max_period);
    auto tau_eps_4 =
        compute_tau_eps<N_SLS_4>(target_Q, tau_sigma_4, min_period, max_period);
    type_real error_4 = compute_lsq_error<N_SLS_4>(
        tau_sigma_4, tau_eps_4, target_Q, min_period, max_period);

    EXPECT_LT(error_4, error_2)
        << "More SLS elements should reduce error for same bandwidth";

    std::cout << "2 decades: N_SLS=2 error=" << error_2 * 100.0 << "%, "
              << "N_SLS=4 error=" << error_4 * 100.0 << "%" << std::endl;
  }
}

// =============================================================================
// Original Tests (updated to use LSQ error where appropriate)
// =============================================================================

// Test that computed tau_eps achieves target Q within tolerance
TEST(Attenuation_ComputeTauEps, AchievesTargetQ_Medium) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real achieved_Q =
      compute_achieved_Q<N_SLS>(tau_sigma, tau_eps, min_period, max_period);

  // Allow 5% tolerance on Q
  EXPECT_NEAR(achieved_Q, target_Q, target_Q * 0.05)
      << "Achieved Q should be close to target Q";
}

// Test with low Q (high attenuation)
TEST(Attenuation_ComputeTauEps, AchievesTargetQ_Low) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 50.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real achieved_Q =
      compute_achieved_Q<N_SLS>(tau_sigma, tau_eps, min_period, max_period);

  EXPECT_NEAR(achieved_Q, target_Q, target_Q * 0.05)
      << "Achieved Q should be close to target Q for low Q";
}

// Test with high Q (low attenuation)
TEST(Attenuation_ComputeTauEps, AchievesTargetQ_High) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 1000.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  type_real achieved_Q =
      compute_achieved_Q<N_SLS>(tau_sigma, tau_eps, min_period, max_period);

  EXPECT_NEAR(achieved_Q, target_Q, target_Q * 0.05)
      << "Achieved Q should be close to target Q for high Q";
}

// Test physical constraint: tau_eps > tau_sigma for positive Q
TEST(Attenuation_ComputeTauEps, TauEpsGreaterThanTauS) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  for (int j = 0; j < N_SLS; ++j) {
    EXPECT_GT(tau_eps(j), tau_sigma(j))
        << "tau_eps(" << j << ") should be greater than tau_sigma(" << j << ")";
  }
}

// Test that output has correct size
TEST(Attenuation_ComputeTauEps, ReturnsCorrectSize) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  EXPECT_EQ(tau_eps.extent(0), N_SLS);
}

// Test with different number of SLS using LSQ error metric
// Based on Savage 2010: 3 decades bandwidth requires different N_SLS for ~1%
// error
TEST(Attenuation_ComputeTauEps, DifferentNumberOfSLS) {
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0; // 3 decades

  type_real bandwidth = bandwidth_decades(min_period, max_period);
  std::cout << "Testing N_SLS comparison at " << bandwidth << " decades"
            << std::endl;

  // Test with 2 SLS
  // Savage 2010: N_SLS=2 can only achieve ~1% error for ~0.9 decades
  // At 3 decades, expect much higher error
  {
    constexpr int N_SLS = 2;
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    // At 3 decades, N_SLS=2 should have significant error (>10% per Savage
    // curve)
    EXPECT_GT(lsq_error * 100.0, 5.0)
        << "N_SLS=2 at 3 decades should have significant LSQ error";
    std::cout << "  N_SLS=2: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }

  // Test with 4 SLS
  // Savage 2010: N_SLS=4 achieves ~1% error at ~2.5 decades
  // At 3 decades, expect slightly higher but still reasonable error
  {
    constexpr int N_SLS = 4;
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    EXPECT_LT(lsq_error * 100.0, 5.0)
        << "N_SLS=4 at 3 decades should achieve <5% LSQ error";
    std::cout << "  N_SLS=4: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }

  // Test with 5 SLS
  // Savage 2010: N_SLS=5 achieves ~1% error at ~3.0 decades
  {
    constexpr int N_SLS = 5;
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    EXPECT_LT(lsq_error * 100.0, 2.0)
        << "N_SLS=5 at 3 decades should achieve ~1% LSQ error (Savage 2010)";
    std::cout << "  N_SLS=5: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }
}

// Test with different period ranges using LSQ error metric
// Validates Savage 2010 bandwidth-error relationship for N_SLS=3
TEST(Attenuation_ComputeTauEps, DifferentPeriodRanges) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;

  std::cout << "Testing bandwidth vs LSQ error for N_SLS=3:" << std::endl;

  // Narrow period range: 1 decade (well within optimal for N_SLS=3)
  {
    constexpr type_real min_period = 1.0;
    constexpr type_real max_period = 10.0; // 1 decade
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    // 1 decade is well below optimal 1.7 decades, should have very low error
    EXPECT_LT(lsq_error * 100.0, 1.1)
        << "1 decade bandwidth should achieve <1% LSQ error for N_SLS=3";
    std::cout << "  1.0 decades: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }

  // Optimal period range: ~1.7 decades (Savage 2010 optimal for N_SLS=3)
  {
    constexpr type_real min_period = 0.1;
    constexpr type_real max_period = 5.01; // ~1.7 decades
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    EXPECT_LT(lsq_error * 100.0, 2.0)
        << "1.7 decades should achieve ~1% LSQ error for N_SLS=3 (Savage 2010)";
    std::cout << "  " << bandwidth_decades(min_period, max_period)
              << " decades: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }

  // Wide period range: 5 decades (far exceeds optimal for N_SLS=3)
  // Savage 2010 predicts ~10% error at 3.5 decades, higher at 5 decades
  {
    constexpr type_real min_period = 0.001;
    constexpr type_real max_period = 100.0; // 5 decades
    auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
    auto tau_eps =
        compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);
    type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                   min_period, max_period);
    // 5 decades far exceeds optimal, expect significant error
    EXPECT_GT(lsq_error * 100.0, 5.0)
        << "5 decades bandwidth should have >5% LSQ error for N_SLS=3";
    std::cout << "  " << bandwidth_decades(min_period, max_period)
              << " decades: LSQ error=" << lsq_error * 100.0 << "%"
              << std::endl;
  }
}

// Test Q reconstruction: verify tan_delta ≈ 1/Q over frequency range
// Reports both max deviation and LSQ error (Savage 2010 metric)
TEST(Attenuation_ComputeTauEps, QReconstructionOverFrequencies) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0; // 3 decades

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  // Set up evaluation frequencies
  Kokkos::View<type_real[NF_ATTENUATION], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f("frequencies");

  const type_real f1 = 1.0 / max_period;
  const type_real f2 = 1.0 / min_period;
  const type_real log_f1 = std::log10(f1);
  const type_real log_f2 = std::log10(f2);
  const type_real d_log_f =
      (log_f2 - log_f1) / (static_cast<type_real>(NF_ATTENUATION) - 1);

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    f(i) = std::pow(10.0, log_f1 + i * d_log_f);
  }

  auto maxwell_factors = specfem::attenuation::maxwell<NF_ATTENUATION, N_SLS>(
      f, tau_sigma, tau_eps);

  // Check that 1/Q is approximately constant over frequency range
  type_real target_inv_Q = 1.0 / target_Q;
  type_real max_deviation = 0.0;

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    type_real tan_delta = maxwell_factors.imag(i) / maxwell_factors.real(i);
    type_real deviation = std::abs(tan_delta - target_inv_Q) / target_inv_Q;
    if (deviation > max_deviation) {
      max_deviation = deviation;
    }
  }

  // Compute LSQ error (Savage 2010 metric)
  type_real lsq_error = compute_lsq_error<N_SLS>(tau_sigma, tau_eps, target_Q,
                                                 min_period, max_period);

  std::cout << "Q reconstruction at "
            << bandwidth_decades(min_period, max_period)
            << " decades:" << std::endl;
  std::cout << "  Max deviation: " << max_deviation * 100.0 << "%" << std::endl;
  std::cout << "  LSQ error (Savage 2010): " << lsq_error * 100.0 << "%"
            << std::endl;

  // Savage 2010: N_SLS=3 at 3 decades is beyond optimal (1.7 decades)
  // Error rises steeply beyond optimal bandwidth, expect ~15-20% LSQ error
  EXPECT_LT(lsq_error * 100.0, 20.0)
      << "LSQ error should be reasonable for N_SLS=3 at 3 decades";
  EXPECT_LT(max_deviation, 0.30)
      << "Max deviation should be <30% (occurs at band edges)";
}

// Test that tau_eps values are finite and positive
TEST(Attenuation_ComputeTauEps, ValuesAreFiniteAndPositive) {
  constexpr int N_SLS = 3;
  constexpr type_real target_Q = 200.0;
  constexpr type_real min_period = 0.01;
  constexpr type_real max_period = 10.0;

  auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
  auto tau_eps =
      compute_tau_eps<N_SLS>(target_Q, tau_sigma, min_period, max_period);

  for (int j = 0; j < N_SLS; ++j) {
    EXPECT_TRUE(std::isfinite(tau_eps(j)))
        << "tau_eps(" << j << ") should be finite";
    EXPECT_GT(tau_eps(j), 0.0) << "tau_eps(" << j << ") should be positive";
  }
}
