#include "specfem/units.hpp"
#include "specfem/utilities/is_close.hpp"
#include <gtest/gtest.h>

using specfem::units::GramPerCubicMeter;
using specfem::units::Hertz;
using specfem::units::Kilometers;
using specfem::units::KilometersPerSecond;
using specfem::units::Meters;
using specfem::units::MetersPerSecond;
using specfem::units::Omega;
using specfem::units::Radians;
using specfem::units::Seconds;
using specfem::utilities::is_close;

// ---------------------------------------------------------------------------
// Construction / raw access
// ---------------------------------------------------------------------------

TEST(Units_Quantity, Construction) {
  EXPECT_EQ(Seconds(2.5).raw(), type_real(2.5));
  EXPECT_EQ(Seconds().raw(), type_real(0.0));
  // explicit operator type_real round-trip
  Seconds s(3.14);
  EXPECT_EQ(static_cast<type_real>(s), type_real(3.14));
}

// ---------------------------------------------------------------------------
// Same-dim arithmetic
// ---------------------------------------------------------------------------

TEST(Units_Quantity, Addition) {
  auto result = Seconds(1.0) + Seconds(2.0);
  EXPECT_TRUE(is_close(result.raw(), type_real(3.0)));
}

TEST(Units_Quantity, Subtraction) {
  auto result = Seconds(5.0) - Seconds(2.0);
  EXPECT_TRUE(is_close(result.raw(), type_real(3.0)));
}

TEST(Units_Quantity, ScalarMultiply) {
  auto result = Seconds(4.0) * type_real(2.5);
  EXPECT_TRUE(is_close(result.raw(), type_real(10.0)));
}

TEST(Units_Quantity, ScalarMultiplyCommutative) {
  auto a = Seconds(4.0) * type_real(2.5);
  auto b = type_real(2.5) * Seconds(4.0);
  EXPECT_TRUE(is_close(a.raw(), b.raw()));
}

TEST(Units_Quantity, ScalarDivide) {
  auto result = Seconds(9.0) / type_real(3.0);
  EXPECT_TRUE(is_close(result.raw(), type_real(3.0)));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

TEST(Units_Quantity, Comparisons) {
  Seconds a(1.0), b(2.0), c(1.0);

  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(a <= c);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a >= c);
  EXPECT_TRUE(a == c);
  EXPECT_TRUE(a != b);

  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(a != c);
}

// ---------------------------------------------------------------------------
// Cross-dim multiply
// ---------------------------------------------------------------------------

TEST(Units_Quantity, CrossDimMultiply) {
  auto result = MetersPerSecond(3.0) * GramPerCubicMeter(2.0);
  // MetersPerSecond = Dim<0,1,-1>, GramPerCubicMeter = Dim<1,-3,0> ->
  // Dim<1,-2,-1>
  EXPECT_TRUE(is_close(result.raw(), type_real(6.0)));
}

// ---------------------------------------------------------------------------
// Cross-dim divide
// ---------------------------------------------------------------------------

TEST(Units_Quantity, CrossDimDivide) {
  auto result = MetersPerSecond(6.0) / Seconds(2.0);
  // MetersPerSecond = Dim<0,1,-1>, Seconds = Dim<0,0,1> -> Dim<0,1,-2>
  EXPECT_TRUE(is_close(result.raw(), type_real(3.0)));
}

// ---------------------------------------------------------------------------
// Same-dim divide -> dimensionless
// ---------------------------------------------------------------------------

TEST(Units_Quantity, SameDimDivide) {
  auto result = Seconds(10.0) / Seconds(2.0);
  EXPECT_TRUE(is_close(result.raw(), type_real(5.0)));
}

// ---------------------------------------------------------------------------
// Angle dimension: Omega and Hertz are distinct; Omega / Radians -> Hertz
// ---------------------------------------------------------------------------

TEST(Units_Quantity, AngleDimension) {
  // Omega and Hertz have the same T exponent but differ in angle exponent
  // so they are distinct types (verified at compile time via aliases)
  Omega w(type_real(2.0));
  Hertz f(type_real(2.0));
  EXPECT_TRUE(is_close(w.raw(), f.raw())); // same numeric value, different type

  // Omega / Radians -> Hertz (A exponent cancels)
  auto result = Omega(6.28) / Radians(1.0);
  EXPECT_TRUE(is_close(result.raw(), type_real(6.28)));
}

// ---------------------------------------------------------------------------
// Scaled aliases: construction
// ---------------------------------------------------------------------------

TEST(Units_Quantity, ScaledConstruction) {
  EXPECT_EQ(Meters(5.0).raw(), type_real(5.0));
  EXPECT_EQ(Kilometers(3.0).raw(), type_real(3.0));
  EXPECT_EQ(KilometersPerSecond(2.5).raw(), type_real(2.5));
}

// ---------------------------------------------------------------------------
// Mixed-scale same-dim arithmetic
// ---------------------------------------------------------------------------

TEST(Units_Quantity, MixedScaleAddition) {
  auto result = Kilometers(1.0) + Meters(1.0); // 1000 m + 1 m = 1001 m
  EXPECT_TRUE(is_close(result.raw(), type_real(1001.0)));
}

TEST(Units_Quantity, MixedScaleSubtraction) {
  auto result = Kilometers(1.0) - Meters(1.0); // 1000 m - 1 m = 999 m
  EXPECT_TRUE(is_close(result.raw(), type_real(999.0)));
}

// ---------------------------------------------------------------------------
// Mixed-scale comparisons
// ---------------------------------------------------------------------------

TEST(Units_Quantity, MixedScaleEquality) {
  EXPECT_TRUE(Kilometers(1.0) == Meters(1000.0));
  EXPECT_FALSE(Kilometers(1.0) == Meters(999.0));
  EXPECT_TRUE(Kilometers(1.0) != Meters(999.0));
}

TEST(Units_Quantity, MixedScaleOrdering) {
  EXPECT_TRUE(Meters(999.0) < Kilometers(1.0));
  EXPECT_TRUE(Meters(999.0) <= Kilometers(1.0));
  EXPECT_TRUE(Kilometers(1.0) > Meters(999.0));
  EXPECT_TRUE(Kilometers(1.0) >= Meters(999.0));
  EXPECT_TRUE(Meters(1000.0) >= Kilometers(1.0));
  EXPECT_TRUE(Kilometers(1.0) <= Meters(1000.0));
}

// ---------------------------------------------------------------------------
// Cross-dim ops with scale propagation
// ---------------------------------------------------------------------------

TEST(Units_Quantity, CrossDimWithScale) {
  // km / s -> KilometersPerSecond (Scale = ratio<1000>)
  auto v = Kilometers(3.0) / Seconds(1.0);
  EXPECT_TRUE(is_close(v.raw(), type_real(3.0)));

  // KilometersPerSecond * Seconds -> Kilometers (Scale = ratio<1000>)
  auto d = KilometersPerSecond(2.0) * Seconds(3.0);
  EXPECT_TRUE(is_close(d.raw(), type_real(6.0)));
}

// ---------------------------------------------------------------------------
// unit_symbols construction tags
// ---------------------------------------------------------------------------

TEST(Units_Quantity, UnitSymbolsTags) {
  using namespace specfem::units::unit_symbols;

  auto d1 = type_real(5.0) * km;
  EXPECT_TRUE(is_close(d1.raw(), type_real(5.0)));

  auto d2 = type_real(3.0) * m;
  EXPECT_TRUE(is_close(d2.raw(), type_real(3.0)));

  auto t = type_real(2.0) * s;
  EXPECT_TRUE(is_close(t.raw(), type_real(2.0)));

  auto v1 = type_real(4.0) * mps;
  EXPECT_TRUE(is_close(v1.raw(), type_real(4.0)));

  auto v2 = type_real(1.5) * kmps;
  EXPECT_TRUE(is_close(v2.raw(), type_real(1.5)));

  // tag / tag cross-dim
  auto v3 = type_real(2.0) * km / (type_real(1.0) * s);
  EXPECT_TRUE(is_close(v3.raw(), type_real(2.0)));
}
