#include "specfem/utilities/frequency_band.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::utilities::FrequencyBand;
using specfem::utilities::is_close;

// ---------------------------------------------------------------------------
// from_period
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromPeriodRoundTrip) {
  constexpr type_real min_p = 1.0, max_p = 100.0;
  auto b = FrequencyBand::from_period(min_p, max_p);
  EXPECT_TRUE(is_close(b.min_period(), min_p));
  EXPECT_TRUE(is_close(b.max_period(), max_p));
}

// ---------------------------------------------------------------------------
// from_frequency
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromFrequencyRoundTrip) {
  constexpr type_real min_f = 0.01, max_f = 1.0;
  auto b = FrequencyBand::from_frequency(min_f, max_f);
  EXPECT_TRUE(is_close(b.min_frequency(), min_f));
  EXPECT_TRUE(is_close(b.max_frequency(), max_f));
}

// ---------------------------------------------------------------------------
// from_omega
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromOmegaRoundTrip) {
  constexpr type_real min_w = 0.5, max_w = 50.0;
  auto b = FrequencyBand::from_omega(min_w, max_w);
  EXPECT_TRUE(is_close(b.min_omega, min_w));
  EXPECT_TRUE(is_close(b.max_omega, max_w));
}

// ---------------------------------------------------------------------------
// All three factories produce equivalent bands for the same physical range.
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FactoriesAreEquivalent) {
  constexpr type_real min_p = 2.0, max_p = 200.0;
  constexpr double two_pi = 2.0 * M_PI;

  auto bp = FrequencyBand::from_period(min_p, max_p);
  auto bf = FrequencyBand::from_frequency(1.0 / max_p, 1.0 / min_p);
  auto bw = FrequencyBand::from_omega(two_pi / max_p, two_pi / min_p);

  EXPECT_TRUE(is_close(bp.min_omega, bf.min_omega));
  EXPECT_TRUE(is_close(bp.max_omega, bf.max_omega));
  EXPECT_TRUE(is_close(bp.min_omega, bw.min_omega));
  EXPECT_TRUE(is_close(bp.max_omega, bw.max_omega));
}

// ---------------------------------------------------------------------------
// Derived getter consistency
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, PeriodFrequencyConsistency) {
  // f = 1/T  and  omega = 2*pi*f
  constexpr type_real min_p = 0.5, max_p = 50.0;
  auto b = FrequencyBand::from_period(min_p, max_p);

  EXPECT_TRUE(is_close(b.min_frequency(), type_real(1.0) / max_p));
  EXPECT_TRUE(is_close(b.max_frequency(), type_real(1.0) / min_p));
  EXPECT_TRUE(is_close(b.min_omega, type_real(2.0 * M_PI) * b.min_frequency()));
  EXPECT_TRUE(is_close(b.max_omega, type_real(2.0 * M_PI) * b.max_frequency()));
}

// min_omega < max_omega for a valid band
TEST(Utilities_FrequencyBand, OmegaOrdering) {
  auto b = FrequencyBand::from_period(1.0, 100.0);
  EXPECT_LT(b.min_omega, b.max_omega);
}
