#include "specfem/attenuation.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::compute_band;
using specfem::units::Hertz;
using specfem::units::Omega;
using specfem::units::Seconds;
using specfem::units::unit_cast;
using specfem::utilities::Band;
using specfem::utilities::is_close;

// ---------------------------------------------------------------------------
// compute_band tests
// ---------------------------------------------------------------------------

// max frequency equals 1/min_resolved_period
TEST(Attenuation_ComputeBand, MaxFreqEqualsInverseInput) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_TRUE(is_close(compute_band<2>(Seconds(min_resolved)).max.raw(),
                       1.0 / min_resolved));
  EXPECT_TRUE(is_close(compute_band<3>(Seconds(min_resolved)).max.raw(),
                       1.0 / min_resolved));
  EXPECT_TRUE(is_close(compute_band<4>(Seconds(min_resolved)).max.raw(),
                       1.0 / min_resolved));
  EXPECT_TRUE(is_close(compute_band<5>(Seconds(min_resolved)).max.raw(),
                       1.0 / min_resolved));
}

// max frequency must be strictly larger than min frequency for all valid N_SLS
TEST(Attenuation_ComputeBand, MaxFreqExceedsMin) {
  constexpr type_real min_resolved = 0.5;

  EXPECT_GT(compute_band<2>(Seconds(min_resolved)).max.raw(),
            compute_band<2>(Seconds(min_resolved)).min.raw());
  EXPECT_GT(compute_band<3>(Seconds(min_resolved)).max.raw(),
            compute_band<3>(Seconds(min_resolved)).min.raw());
  EXPECT_GT(compute_band<4>(Seconds(min_resolved)).max.raw(),
            compute_band<4>(Seconds(min_resolved)).min.raw());
  EXPECT_GT(compute_band<5>(Seconds(min_resolved)).max.raw(),
            compute_band<5>(Seconds(min_resolved)).min.raw());
}

// The decade width theta(N_SLS) gives max_freq/min_freq = 10^theta
// Verified values: theta = {0.75, 1.75, 2.25, 2.85} for N_SLS = {2,3,4,5}
TEST(Attenuation_ComputeBand, DecadeWidthMatchesTheta) {
  constexpr type_real min_resolved = 1.0;

  auto check = [&](auto band, double theta) {
    type_real expected_ratio = static_cast<type_real>(std::pow(10.0, theta));
    EXPECT_TRUE(is_close((band.max / band.min).raw(), expected_ratio));
  };

  check(compute_band<2>(Seconds(min_resolved)), 0.75);
  check(compute_band<3>(Seconds(min_resolved)), 1.75);
  check(compute_band<4>(Seconds(min_resolved)), 2.25);
  check(compute_band<5>(Seconds(min_resolved)), 2.85);
}

// The band scales inversely with min_resolved_period
TEST(Attenuation_ComputeBand, ScalesWithInput) {
  constexpr type_real base = 0.1;
  constexpr type_real scale = 5.0;

  auto b1 = compute_band<3>(Seconds(base));
  auto b2 = compute_band<3>(Seconds(base * scale));

  EXPECT_TRUE(is_close(b2.min.raw(), b1.min.raw() / scale));
  EXPECT_TRUE(is_close(b2.max.raw(), b1.max.raw() / scale));
}

// Increasing N_SLS widens the band (larger theta -> lower min frequency)
TEST(Attenuation_ComputeBand, MoreSLSGivesWiderBand) {
  constexpr type_real min_resolved = 1.0;

  auto b2 = compute_band<2>(Seconds(min_resolved));
  auto b3 = compute_band<3>(Seconds(min_resolved));
  auto b4 = compute_band<4>(Seconds(min_resolved));
  auto b5 = compute_band<5>(Seconds(min_resolved));

  EXPECT_GT(b2.min, b3.min);
  EXPECT_GT(b3.min, b4.min);
  EXPECT_GT(b4.min, b5.min);
}
