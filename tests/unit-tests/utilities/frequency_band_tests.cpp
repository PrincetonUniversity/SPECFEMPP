#include "specfem/constants.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/frequency_band.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::units::Hertz;
using specfem::units::Omega;
using specfem::units::Seconds;
using specfem::utilities::FrequencyBand;
using specfem::utilities::is_close;
using specfem::utilities::unit_cast;

// ---------------------------------------------------------------------------
// FrequencyBand unit_cast
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FrequencyBandSecondsToHertz) {
  constexpr type_real min_p = 1.0, max_p = 100.0;
  FrequencyBand<Seconds> bp{ Seconds(min_p), Seconds(max_p) };
  auto bf = unit_cast<Hertz>(bp);

  // min period -> max frequency
  EXPECT_TRUE(is_close(bf.max.raw(), type_real(1.0) / min_p));
  // max period -> min frequency
  EXPECT_TRUE(is_close(bf.min.raw(), type_real(1.0) / max_p));
}

TEST(Utilities_FrequencyBand, FrequencyBandSecondsToOmega) {
  constexpr type_real min_p = 2.0, max_p = 10.0;
  FrequencyBand<Seconds> bp{ Seconds(min_p), Seconds(max_p) };
  auto bw = unit_cast<Omega>(bp);

  EXPECT_TRUE(is_close(bw.max.raw(), unit_cast<Omega>(Seconds(min_p))));
  EXPECT_TRUE(is_close(bw.min.raw(), unit_cast<Omega>(Seconds(max_p))));
}

// ---------------------------------------------------------------------------
// Period (Seconds) round-trip
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromPeriodRoundTrip) {
  constexpr type_real min_p = 1.0, max_p = 100.0;
  FrequencyBand<Seconds> b{ Seconds(min_p), Seconds(max_p) };
  EXPECT_TRUE(is_close(b.min.raw(), min_p));
  EXPECT_TRUE(is_close(b.max.raw(), max_p));
}

// ---------------------------------------------------------------------------
// Frequency (Hertz) round-trip
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromFrequencyRoundTrip) {
  constexpr type_real min_f = 0.01, max_f = 1.0;
  FrequencyBand<Hertz> b{ Hertz(min_f), Hertz(max_f) };
  EXPECT_TRUE(is_close(b.min.raw(), min_f));
  EXPECT_TRUE(is_close(b.max.raw(), max_f));
}

// ---------------------------------------------------------------------------
// Angular frequency (Omega) round-trip
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, FromOmegaRoundTrip) {
  constexpr type_real min_w = 0.5, max_w = 50.0;
  FrequencyBand<Omega> b{ Omega(min_w), Omega(max_w) };
  EXPECT_TRUE(is_close(b.min.raw(), min_w));
  EXPECT_TRUE(is_close(b.max.raw(), max_w));
}

// ---------------------------------------------------------------------------
// All three representations are equivalent for the same physical band.
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, RepresentationsAreEquivalent) {
  constexpr type_real min_p = 2.0, max_p = 200.0;

  FrequencyBand<Seconds> bp{ Seconds(min_p), Seconds(max_p) };
  auto bf = unit_cast<Hertz>(bp);
  auto bw = unit_cast<Omega>(bp);

  // Both should give the same omega band when converted
  auto bw_from_f = unit_cast<Omega>(bf);
  EXPECT_TRUE(is_close(bw.min.raw(), bw_from_f.min.raw()));
  EXPECT_TRUE(is_close(bw.max.raw(), bw_from_f.max.raw()));
}

// ---------------------------------------------------------------------------
// Derived unit consistency: f = 1/T, omega = 2*pi*f
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, PeriodFrequencyConsistency) {
  constexpr type_real min_p = 0.5, max_p = 50.0;
  FrequencyBand<Seconds> bp{ Seconds(min_p), Seconds(max_p) };

  auto bf = unit_cast<Hertz>(bp);
  auto bw = unit_cast<Omega>(bp);

  // min period -> max frequency
  EXPECT_TRUE(is_close(bf.max.raw(), type_real(1.0) / min_p));
  // max period -> min frequency
  EXPECT_TRUE(is_close(bf.min.raw(), type_real(1.0) / max_p));
  // omega = 2*pi*f
  EXPECT_TRUE(is_close(bw.min.raw(), type_real(2.0 * M_PI) * bf.min.raw()));
  EXPECT_TRUE(is_close(bw.max.raw(), type_real(2.0 * M_PI) * bf.max.raw()));
}

// ---------------------------------------------------------------------------
// Omega band is ordered min < max after conversion from period
// ---------------------------------------------------------------------------

TEST(Utilities_FrequencyBand, OmegaOrdering) {
  FrequencyBand<Seconds> bp{ Seconds(1.0), Seconds(100.0) };
  auto bw = unit_cast<Omega>(bp);
  EXPECT_LT(bw.min.raw(), bw.max.raw());
}
