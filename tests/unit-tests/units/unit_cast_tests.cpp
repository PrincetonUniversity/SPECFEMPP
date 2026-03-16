#include "specfem/units.hpp"
#include "specfem/utilities/frequency_band.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::units::Hertz;
using specfem::units::Kilometers;
using specfem::units::KilometersPerSecond;
using specfem::units::Meters;
using specfem::units::Omega;
using specfem::units::Seconds;
using specfem::units::Velocity;
using specfem::utilities::FrequencyBand;
using specfem::utilities::is_close;
using specfem::utilities::unit_cast;

static constexpr type_real two_pi = type_real(6.28318530717958647692);

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, IdentitySeconds) {
  EXPECT_TRUE(is_close(specfem::units::unit_cast<Seconds>(Seconds(3.0)).raw(),
                       type_real(3.0)));
}

TEST(Units_UnitCast, IdentityHertz) {
  EXPECT_TRUE(is_close(specfem::units::unit_cast<Hertz>(Hertz(0.5)).raw(),
                       type_real(0.5)));
}

TEST(Units_UnitCast, IdentityOmega) {
  EXPECT_TRUE(is_close(specfem::units::unit_cast<Omega>(Omega(3.14)).raw(),
                       type_real(3.14)));
}

// ---------------------------------------------------------------------------
// Seconds <-> Hertz
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, SecondsToHertz) {
  auto f = specfem::units::unit_cast<Hertz>(Seconds(2.0));
  EXPECT_TRUE(is_close(f.raw(), type_real(0.5)));
}

TEST(Units_UnitCast, HertzToSeconds) {
  auto s = specfem::units::unit_cast<Seconds>(Hertz(0.5));
  EXPECT_TRUE(is_close(s.raw(), type_real(2.0)));
}

TEST(Units_UnitCast, SecondsHertzRoundTrip) {
  constexpr type_real T = 3.7;
  auto s_back = specfem::units::unit_cast<Seconds>(
      specfem::units::unit_cast<Hertz>(Seconds(T)));
  EXPECT_TRUE(is_close(s_back.raw(), T));
}

// ---------------------------------------------------------------------------
// Seconds <-> Omega
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, SecondsToOmega) {
  auto w = specfem::units::unit_cast<Omega>(Seconds(1.0));
  EXPECT_TRUE(is_close(w.raw(), two_pi));
}

TEST(Units_UnitCast, OmegaToSeconds) {
  auto s = specfem::units::unit_cast<Seconds>(Omega(two_pi));
  EXPECT_TRUE(is_close(s.raw(), type_real(1.0)));
}

TEST(Units_UnitCast, SecondsOmegaRoundTrip) {
  constexpr type_real T = 5.0;
  auto s_back = specfem::units::unit_cast<Seconds>(
      specfem::units::unit_cast<Omega>(Seconds(T)));
  EXPECT_TRUE(is_close(s_back.raw(), T));
}

// ---------------------------------------------------------------------------
// Hertz <-> Omega
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, HertzToOmega) {
  auto w = specfem::units::unit_cast<Omega>(Hertz(1.0));
  EXPECT_TRUE(is_close(w.raw(), two_pi));
}

TEST(Units_UnitCast, OmegaToHertz) {
  auto f = specfem::units::unit_cast<Hertz>(Omega(two_pi));
  EXPECT_TRUE(is_close(f.raw(), type_real(1.0)));
}

TEST(Units_UnitCast, HertzOmegaRoundTrip) {
  constexpr type_real f0 = 2.5;
  auto f_back = specfem::units::unit_cast<Hertz>(
      specfem::units::unit_cast<Omega>(Hertz(f0)));
  EXPECT_TRUE(is_close(f_back.raw(), f0));
}

// ---------------------------------------------------------------------------
// Consistency: chained conversion equals direct conversion
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, ChainedConsistency) {
  constexpr type_real T = 4.0;
  auto direct = specfem::units::unit_cast<Omega>(Seconds(T));
  auto chained = specfem::units::unit_cast<Omega>(
      specfem::units::unit_cast<Hertz>(Seconds(T)));
  EXPECT_TRUE(is_close(direct.raw(), chained.raw()));
}

// ---------------------------------------------------------------------------
// Identity for scaled types
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, IdentityKilometers) {
  auto k = specfem::units::unit_cast<Kilometers>(Kilometers(3.0));
  EXPECT_TRUE(is_close(k.raw(), type_real(3.0)));
}

// ---------------------------------------------------------------------------
// Length scale conversions
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, MetersToKilometers) {
  auto k = specfem::units::unit_cast<Kilometers>(Meters(5000.0));
  EXPECT_TRUE(is_close(k.raw(), type_real(5.0)));
}

TEST(Units_UnitCast, KilometersToMeters) {
  auto m = specfem::units::unit_cast<Meters>(Kilometers(5.0));
  EXPECT_TRUE(is_close(m.raw(), type_real(5000.0)));
}

TEST(Units_UnitCast, MetersKilometersRoundTrip) {
  constexpr type_real dist = 1234.5;
  auto back = specfem::units::unit_cast<Meters>(
      specfem::units::unit_cast<Kilometers>(Meters(dist)));
  EXPECT_TRUE(is_close(back.raw(), dist));
}

// ---------------------------------------------------------------------------
// Velocity scale conversions
// ---------------------------------------------------------------------------

TEST(Units_UnitCast, VelocityToKilometersPerSecond) {
  auto k = specfem::units::unit_cast<KilometersPerSecond>(Velocity(3000.0));
  EXPECT_TRUE(is_close(k.raw(), type_real(3.0)));
}

TEST(Units_UnitCast, KilometersPerSecondToVelocity) {
  auto v = specfem::units::unit_cast<Velocity>(KilometersPerSecond(3.0));
  EXPECT_TRUE(is_close(v.raw(), type_real(3000.0)));
}

TEST(Units_UnitCast, VelocityKilometersPerSecondRoundTrip) {
  constexpr type_real speed = 6789.0;
  auto back = specfem::units::unit_cast<Velocity>(
      specfem::units::unit_cast<KilometersPerSecond>(Velocity(speed)));
  EXPECT_TRUE(is_close(back.raw(), speed));
}
