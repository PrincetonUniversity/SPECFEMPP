#include "specfem/attenuation.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::compute_band;
using specfem::utilities::is_close;

// ---------------------------------------------------------------------------
// compute_band tests
// ---------------------------------------------------------------------------

// min_period is always equal to the input min_resolved_period
TEST(Attenuation_ComputeBand, MinPeriodEqualsInput) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_TRUE(
      is_close(compute_band<2>(min_resolved).min_period(), min_resolved));
  EXPECT_TRUE(
      is_close(compute_band<3>(min_resolved).min_period(), min_resolved));
  EXPECT_TRUE(
      is_close(compute_band<4>(min_resolved).min_period(), min_resolved));
  EXPECT_TRUE(
      is_close(compute_band<5>(min_resolved).min_period(), min_resolved));
}

// max_period must be strictly larger than min_period for all valid N_SLS
TEST(Attenuation_ComputeBand, MaxPeriodExceedsMin) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_GT(compute_band<2>(min_resolved).max_period(), min_resolved);
  EXPECT_GT(compute_band<3>(min_resolved).max_period(), min_resolved);
  EXPECT_GT(compute_band<4>(min_resolved).max_period(), min_resolved);
  EXPECT_GT(compute_band<5>(min_resolved).max_period(), min_resolved);
}

// The decade width theta(N_SLS) gives max_period = min_period * 10^theta
// Verified values: theta = {0.75, 1.75, 2.25, 2.85} for N_SLS = {2,3,4,5}
TEST(Attenuation_ComputeBand, DecadeWidthMatchesTheta) {
  constexpr type_real min_resolved = 1.0;

  auto check = [&](auto band, double theta) {
    type_real expected_ratio = static_cast<type_real>(std::pow(10.0, theta));
    EXPECT_TRUE(
        is_close(band.max_period() / band.min_period(), expected_ratio));
  };

  check(compute_band<2>(min_resolved), 0.75);
  check(compute_band<3>(min_resolved), 1.75);
  check(compute_band<4>(min_resolved), 2.25);
  check(compute_band<5>(min_resolved), 2.85);
}

// The band scales linearly with min_resolved_period
TEST(Attenuation_ComputeBand, ScalesWithInput) {
  constexpr type_real base = 0.1;
  constexpr type_real scale = 5.0;

  auto b1 = compute_band<3>(base);
  auto b2 = compute_band<3>(base * scale);

  EXPECT_TRUE(is_close(b2.min_period(), b1.min_period() * scale));
  EXPECT_TRUE(is_close(b2.max_period(), b1.max_period() * scale));
}

// Increasing N_SLS widens the band (larger theta -> larger max_period)
TEST(Attenuation_ComputeBand, MoreSLSGivesWiderBand) {
  constexpr type_real min_resolved = 1.0;

  auto b2 = compute_band<2>(min_resolved);
  auto b3 = compute_band<3>(min_resolved);
  auto b4 = compute_band<4>(min_resolved);
  auto b5 = compute_band<5>(min_resolved);

  EXPECT_LT(b2.max_period(), b3.max_period());
  EXPECT_LT(b3.max_period(), b4.max_period());
  EXPECT_LT(b4.max_period(), b5.max_period());
}
