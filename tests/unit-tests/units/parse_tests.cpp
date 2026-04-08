#include "specfem/units.hpp"
#include "specfem/utilities/is_close.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

using specfem::units::AnyQuantity;
using specfem::units::Dimensionless;
using specfem::units::GramPerCubicMeter;
using specfem::units::Grams;
using specfem::units::Hertz;
using specfem::units::KilogramPerCubicMeter;
using specfem::units::Kilograms;
using specfem::units::Kilometers;
using specfem::units::KilometersPerSecond;
using specfem::units::Megapascal;
using specfem::units::Meters;
using specfem::units::MetersPerSecond;
using specfem::units::Omega;
using specfem::units::parse;
using specfem::units::Pascal;
using specfem::units::quantity_cast;
using specfem::units::Radians;
using specfem::units::Seconds;
using specfem::utilities::is_close;

namespace {
constexpr type_real parse_two_pi = type_real(6.28318530717958647692);
}

// ---------------------------------------------------------------------------
// parse — basic unit recognition
// ---------------------------------------------------------------------------

TEST(Units_Parse, Dimensionless) {
  auto v = parse("1.0 1");
  ASSERT_TRUE(std::holds_alternative<Dimensionless>(v));
  EXPECT_TRUE(is_close(std::get<Dimensionless>(v).raw(), type_real(1.0)));
}

TEST(Units_Parse, Seconds) {
  auto v = parse("0.5 s");
  ASSERT_TRUE(std::holds_alternative<Seconds>(v));
  EXPECT_TRUE(is_close(std::get<Seconds>(v).raw(), type_real(0.5)));
}

TEST(Units_Parse, Hertz) {
  auto v = parse("20.0 Hz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(20.0)));
}

TEST(Units_Parse, Omega) {
  auto v = parse("6.28 rad/s");
  ASSERT_TRUE(std::holds_alternative<Omega>(v));
  EXPECT_TRUE(is_close(std::get<Omega>(v).raw(), type_real(6.28)));
}

TEST(Units_Parse, Radians) {
  auto v = parse("3.14 rad");
  ASSERT_TRUE(std::holds_alternative<Radians>(v));
  EXPECT_TRUE(is_close(std::get<Radians>(v).raw(), type_real(3.14)));
}

TEST(Units_Parse, Meters) {
  auto v = parse("100.0 m");
  ASSERT_TRUE(std::holds_alternative<Meters>(v));
  EXPECT_TRUE(is_close(std::get<Meters>(v).raw(), type_real(100.0)));
}

TEST(Units_Parse, Kilometers) {
  auto v = parse("1.5 km");
  ASSERT_TRUE(std::holds_alternative<Kilometers>(v));
  EXPECT_TRUE(is_close(std::get<Kilometers>(v).raw(), type_real(1.5)));
}

TEST(Units_Parse, MetersPerSecond) {
  auto v = parse("3000.0 m/s");
  ASSERT_TRUE(std::holds_alternative<MetersPerSecond>(v));
  EXPECT_TRUE(is_close(std::get<MetersPerSecond>(v).raw(), type_real(3000.0)));
}

TEST(Units_Parse, KilometersPerSecond) {
  auto v = parse("3.0 km/s");
  ASSERT_TRUE(std::holds_alternative<KilometersPerSecond>(v));
  EXPECT_TRUE(is_close(std::get<KilometersPerSecond>(v).raw(), type_real(3.0)));
}

TEST(Units_Parse, Grams) {
  auto v = parse("5.0 g");
  ASSERT_TRUE(std::holds_alternative<Grams>(v));
  EXPECT_TRUE(is_close(std::get<Grams>(v).raw(), type_real(5.0)));
}

TEST(Units_Parse, Kilograms) {
  auto v = parse("2.5 kg");
  ASSERT_TRUE(std::holds_alternative<Kilograms>(v));
  EXPECT_TRUE(is_close(std::get<Kilograms>(v).raw(), type_real(2.5)));
}

TEST(Units_Parse, GramPerCubicMeter) {
  auto v = parse("2700.0 g/m3");
  ASSERT_TRUE(std::holds_alternative<GramPerCubicMeter>(v));
  EXPECT_TRUE(
      is_close(std::get<GramPerCubicMeter>(v).raw(), type_real(2700.0)));
}

TEST(Units_Parse, KilogramPerCubicMeter) {
  auto v = parse("2.7 kg/m3");
  ASSERT_TRUE(std::holds_alternative<KilogramPerCubicMeter>(v));
  EXPECT_TRUE(
      is_close(std::get<KilogramPerCubicMeter>(v).raw(), type_real(2.7)));
}

TEST(Units_Parse, Pascal) {
  auto v = parse("101325.0 Pa");
  ASSERT_TRUE(std::holds_alternative<Pascal>(v));
  EXPECT_TRUE(is_close(std::get<Pascal>(v).raw(), type_real(101325.0)));
}

TEST(Units_Parse, Megapascal) {
  auto v = parse("1.0 MPa");
  ASSERT_TRUE(std::holds_alternative<Megapascal>(v));
  EXPECT_TRUE(is_close(std::get<Megapascal>(v).raw(), type_real(1.0)));
}

// ---------------------------------------------------------------------------
// parse — explicit prefixed aliases in table
// ---------------------------------------------------------------------------

TEST(Units_Parse, Milliseconds_ExactEntry) {
  // "ms" is an exact table entry → Seconds(v * 1e-3)
  auto v = parse("10.0 ms");
  ASSERT_TRUE(std::holds_alternative<Seconds>(v));
  EXPECT_TRUE(is_close(std::get<Seconds>(v).raw(), type_real(10.0e-3)));
}

TEST(Units_Parse, Microseconds_us) {
  auto v = parse("500.0 us");
  ASSERT_TRUE(std::holds_alternative<Seconds>(v));
  EXPECT_TRUE(is_close(std::get<Seconds>(v).raw(), type_real(500.0e-6)));
}

TEST(Units_Parse, KiloHertz_ExactEntry) {
  // "kHz" is an exact table entry → Hertz(v * 1e3)
  auto v = parse("2.0 kHz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(2000.0)));
}

TEST(Units_Parse, MegaHertz) {
  auto v = parse("1.0 MHz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(1.0e6)));
}

TEST(Units_Parse, MilliHertz) {
  auto v = parse("1.0 mHz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(1.0e-3)));
}

TEST(Units_Parse, KiloPascal) {
  auto v = parse("100.0 kPa");
  ASSERT_TRUE(std::holds_alternative<Pascal>(v));
  EXPECT_TRUE(is_close(std::get<Pascal>(v).raw(), type_real(100.0e3)));
}

TEST(Units_Parse, GigaPascal) {
  // GPa → stored as Megapascal(v * 1e3); raw = 1000 for 1 GPa
  auto v = parse("1.0 GPa");
  ASSERT_TRUE(std::holds_alternative<Megapascal>(v));
  EXPECT_TRUE(is_close(std::get<Megapascal>(v).raw(), type_real(1000.0)));
}

// ---------------------------------------------------------------------------
// parse — SI prefix fallback (not in exact table)
// ---------------------------------------------------------------------------

TEST(Units_Parse, PrefixFallback_kPa_via_prefix) {
  // "kPa" is in the exact table; try a unit that exercises the fallback path:
  // "km/s" is exact, but "Mm/s" is not — strip "M" prefix, look up "m/s"
  auto v = parse("1.0 Mm/s"); // 1 mega-m/s = 1e6 m/s stored as MetersPerSecond
  ASSERT_TRUE(std::holds_alternative<MetersPerSecond>(v));
  EXPECT_TRUE(is_close(std::get<MetersPerSecond>(v).raw(), type_real(1.0e6)));
}

// ---------------------------------------------------------------------------
// parse — whitespace and notation variants
// ---------------------------------------------------------------------------

TEST(Units_Parse, NoWhitespaceBetweenNumberAndUnit) {
  auto v = parse("20.0Hz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(20.0)));
}

TEST(Units_Parse, LeadingWhitespace) {
  auto v = parse("  20.0 Hz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(20.0)));
}

TEST(Units_Parse, ScientificNotation) {
  auto v = parse("2.0e1 Hz");
  ASSERT_TRUE(std::holds_alternative<Hertz>(v));
  EXPECT_TRUE(is_close(std::get<Hertz>(v).raw(), type_real(20.0)));
}

TEST(Units_Parse, NegativeExponent) {
  auto v = parse("1.0e-3 s");
  ASSERT_TRUE(std::holds_alternative<Seconds>(v));
  EXPECT_TRUE(is_close(std::get<Seconds>(v).raw(), type_real(1.0e-3)));
}

// ---------------------------------------------------------------------------
// parse — error handling
// ---------------------------------------------------------------------------

TEST(Units_Parse, ThrowsOnBadNumber) {
  EXPECT_THROW(parse("abc Hz"), std::invalid_argument);
}

TEST(Units_Parse, ThrowsOnUnknownUnit) {
  EXPECT_THROW(parse("1.0 lightyear"), std::invalid_argument);
}

TEST(Units_Parse, ThrowsOnEmptyString) {
  EXPECT_THROW(parse(""), std::invalid_argument);
}

TEST(Units_Parse, ThrowsOnNumberOnly) {
  // No unit → unit string is empty → not in table → throws
  EXPECT_THROW(parse("1.0 "), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// quantity_cast — identity (same type)
// ---------------------------------------------------------------------------

TEST(Units_QuantityCast, IdentityHertz) {
  auto f = quantity_cast<Hertz>(parse("20.0 Hz"));
  EXPECT_TRUE(is_close(f.raw(), type_real(20.0)));
}

TEST(Units_QuantityCast, IdentitySeconds) {
  auto t = quantity_cast<Seconds>(parse("0.5 s"));
  EXPECT_TRUE(is_close(t.raw(), type_real(0.5)));
}

TEST(Units_QuantityCast, IdentityOmega) {
  auto w = quantity_cast<Omega>(parse("6.28 rad/s"));
  EXPECT_TRUE(is_close(w.raw(), type_real(6.28)));
}

// ---------------------------------------------------------------------------
// quantity_cast — cross-dimension spectral conversions
// ---------------------------------------------------------------------------

TEST(Units_QuantityCast, HertzToOmega) {
  auto w = quantity_cast<Omega>(parse("20.0 Hz"));
  EXPECT_TRUE(is_close(w.raw(), type_real(20.0) * parse_two_pi));
}

TEST(Units_QuantityCast, SecondsToHertz) {
  auto f = quantity_cast<Hertz>(parse("0.5 s"));
  EXPECT_TRUE(is_close(f.raw(), type_real(2.0)));
}

TEST(Units_QuantityCast, SecondsToOmega) {
  auto w = quantity_cast<Omega>(parse("1.0 s"));
  EXPECT_TRUE(is_close(w.raw(), parse_two_pi));
}

TEST(Units_QuantityCast, KiloHertzToOmega) {
  // parse("2.0 kHz") → Hertz(2000), then cast through unit_cast<Omega,Hertz>
  auto w = quantity_cast<Omega>(parse("2.0 kHz"));
  EXPECT_TRUE(is_close(w.raw(), type_real(2000.0) * parse_two_pi));
}

TEST(Units_QuantityCast, MillisecondsToHertz) {
  // 1 ms period → 1000 Hz
  auto f = quantity_cast<Hertz>(parse("1.0 ms"));
  EXPECT_TRUE(is_close(f.raw(), type_real(1000.0)));
}

// ---------------------------------------------------------------------------
// quantity_cast — same-dimension scale conversions
// ---------------------------------------------------------------------------

TEST(Units_QuantityCast, KilometersToMeters) {
  auto m = quantity_cast<Meters>(parse("1.5 km"));
  EXPECT_TRUE(is_close(m.raw(), type_real(1500.0)));
}

TEST(Units_QuantityCast, MetersToKilometers) {
  auto km = quantity_cast<Kilometers>(parse("3000.0 m"));
  EXPECT_TRUE(is_close(km.raw(), type_real(3.0)));
}

TEST(Units_QuantityCast, KilometersPerSecondToMetersPerSecond) {
  auto v = quantity_cast<MetersPerSecond>(parse("3.0 km/s"));
  EXPECT_TRUE(is_close(v.raw(), type_real(3000.0)));
}

// ---------------------------------------------------------------------------
// quantity_cast — unsupported conversion throws
// ---------------------------------------------------------------------------

TEST(Units_QuantityCast, ThrowsHertzToMeters) {
  EXPECT_THROW(quantity_cast<Meters>(parse("20.0 Hz")), std::invalid_argument);
}

TEST(Units_QuantityCast, ThrowsMetersToSeconds) {
  EXPECT_THROW(quantity_cast<Seconds>(parse("100.0 m")), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// quantity_cast string overload
// ---------------------------------------------------------------------------

TEST(Units_QuantityCast, StringOverloadHertz) {
  auto f = quantity_cast<Hertz>("20.0 Hz");
  EXPECT_TRUE(is_close(f.raw(), type_real(20.0)));
}

TEST(Units_QuantityCast, StringOverloadCrossDimension) {
  auto w = quantity_cast<Omega>("20.0 Hz");
  EXPECT_TRUE(is_close(w.raw(), type_real(20.0) * parse_two_pi));
}

TEST(Units_QuantityCast, StringOverloadThrowsUnknownUnit) {
  EXPECT_THROW((quantity_cast<Hertz>("1.0 lightyear")), std::invalid_argument);
}
