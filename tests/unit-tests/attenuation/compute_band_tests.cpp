#include "specfem/attenuation.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::compute_band;
using specfem::units::Omega;
using specfem::units::Seconds;
using specfem::utilities::Band;
using specfem::utilities::is_close;
using specfem::utilities::unit_cast;

// ---------------------------------------------------------------------------
// compute_band tests
// ---------------------------------------------------------------------------

// min_period is always equal to the input min_resolved_period
TEST(Attenuation_ComputeBand, MinPeriodEqualsInput) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_TRUE(is_close(
      unit_cast<Seconds>(compute_band<2>(Seconds(min_resolved))).min.raw(),
      min_resolved));
  EXPECT_TRUE(is_close(
      unit_cast<Seconds>(compute_band<3>(Seconds(min_resolved))).min.raw(),
      min_resolved));
  EXPECT_TRUE(is_close(
      unit_cast<Seconds>(compute_band<4>(Seconds(min_resolved))).min.raw(),
      min_resolved));
  EXPECT_TRUE(is_close(
      unit_cast<Seconds>(compute_band<5>(Seconds(min_resolved))).min.raw(),
      min_resolved));
}

// max_period must be strictly larger than min_period for all valid N_SLS
TEST(Attenuation_ComputeBand, MaxPeriodExceedsMin) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_GT(
      unit_cast<Seconds>(compute_band<2>(Seconds(min_resolved))).max.raw(),
      min_resolved);
  EXPECT_GT(
      unit_cast<Seconds>(compute_band<3>(Seconds(min_resolved))).max.raw(),
      min_resolved);
  EXPECT_GT(
      unit_cast<Seconds>(compute_band<4>(Seconds(min_resolved))).max.raw(),
      min_resolved);
  EXPECT_GT(
      unit_cast<Seconds>(compute_band<5>(Seconds(min_resolved))).max.raw(),
      min_resolved);
}

// The decade width theta(N_SLS) gives max_period = min_period * 10^theta
// Verified values: theta = {0.75, 1.75, 2.25, 2.85} for N_SLS = {2,3,4,5}
TEST(Attenuation_ComputeBand, DecadeWidthMatchesTheta) {
  constexpr type_real min_resolved = 1.0;

  auto check = [&](auto band, double theta) {
    auto period_band = unit_cast<Seconds>(band);
    type_real expected_ratio = static_cast<type_real>(std::pow(10.0, theta));
    EXPECT_TRUE(
        is_close((period_band.max / period_band.min).raw(), expected_ratio));
  };

  check(compute_band<2>(Seconds(min_resolved)), 0.75);
  check(compute_band<3>(Seconds(min_resolved)), 1.75);
  check(compute_band<4>(Seconds(min_resolved)), 2.25);
  check(compute_band<5>(Seconds(min_resolved)), 2.85);
}

// The band scales linearly with min_resolved_period
TEST(Attenuation_ComputeBand, ScalesWithInput) {
  constexpr type_real base = 0.1;
  constexpr type_real scale = 5.0;

  auto b1 = unit_cast<Seconds>(compute_band<3>(Seconds(base)));
  auto b2 = unit_cast<Seconds>(compute_band<3>(Seconds(base * scale)));

  EXPECT_TRUE(is_close(b2.min.raw(), b1.min.raw() * scale));
  EXPECT_TRUE(is_close(b2.max.raw(), b1.max.raw() * scale));
}

// Increasing N_SLS widens the band (larger theta -> larger max_period)
TEST(Attenuation_ComputeBand, MoreSLSGivesWiderBand) {
  constexpr type_real min_resolved = 1.0;

  auto b2 = unit_cast<Seconds>(compute_band<2>(Seconds(min_resolved)));
  auto b3 = unit_cast<Seconds>(compute_band<3>(Seconds(min_resolved)));
  auto b4 = unit_cast<Seconds>(compute_band<4>(Seconds(min_resolved)));
  auto b5 = unit_cast<Seconds>(compute_band<5>(Seconds(min_resolved)));

  EXPECT_LT(b2.max, b3.max);
  EXPECT_LT(b3.max, b4.max);
  EXPECT_LT(b4.max, b5.max);
}
