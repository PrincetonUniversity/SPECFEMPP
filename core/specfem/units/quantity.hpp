#pragma once
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <compare>
#include <ratio>
#include <type_traits>

namespace specfem::units {

/**
 * @brief Compile-time physical dimension representation.
 *
 * Encodes dimensional exponents for mass, length, time, and angle.
 * The angle dimension distinguishes frequency types (Hertz vs Omega).
 *
 * @tparam M Mass exponent
 * @tparam L Length exponent
 * @tparam T Time exponent
 * @tparam A Angle exponent (0 for Hertz, 1 for rad/s)
 */

// Compile-time dimension: [mass, length, time, angle]
// angle distinguishes e.g. Hertz (A=0) from Omega/@f$ \mathrm{rad} \cdot
// \mathrm{s}^{-1} @f$ (A=1)
template <int M, int L, int T, int A = 0> struct Dim {
  static constexpr int mass = M;   ///< Mass dimension exponent
  static constexpr int length = L; ///< Length dimension exponent
  static constexpr int time = T;   ///< Time dimension exponent
  static constexpr int angle = A;  ///< Angle dimension exponent
};

/**
 * @brief Internal helpers for compile-time ratio arithmetic.
 */
namespace impl {

/**
 * @brief Compile-time greatest common divisor.
 *
 * Provides GCD for std::ratio operations (not all C++ implementations include
 * std::ratio_gcd).
 */
template <intmax_t A, intmax_t B> struct ct_gcd {
  static constexpr intmax_t value = ct_gcd<B, A % B>::value;
};
template <intmax_t A> struct ct_gcd<A, 0> {
  static constexpr intmax_t value = (A < 0 ? -A : A);
};
template <intmax_t B> struct ct_gcd<0, B> {
  static constexpr intmax_t value = (B < 0 ? -B : B);
};
template <> struct ct_gcd<0, 0> {
  static constexpr intmax_t value = 1;
};

/// GCD of two std::ratio types
template <typename R1, typename R2>
using ratio_gcd =
    std::ratio<ct_gcd<(R1::num < 0 ? -R1::num : R1::num),
                      (R2::num < 0 ? -R2::num : R2::num)>::value,
               (R1::den / ct_gcd<R1::den, R2::den>::value) * R2::den>;

} // namespace impl

/**
 * @brief Compile-time conversion of std::ratio to a scalar value.
 *
 * @tparam R std::ratio type
 * @return constexpr type_real Ratio as a floating-point value
 */
template <typename R>
constexpr type_real ratio_value =
    static_cast<type_real>(R::num) / static_cast<type_real>(R::den);

/**
 * @brief Type-safe physical quantity with compile-time dimensional analysis.
 *
 * Represents a physical quantity with dimensions D and scale factor Scale.
 * Prevents unit mismatches at compile time through template parameters.
 * Arithmetic operations preserve dimensional correctness.
 *
 * @tparam D Dimension type (Dim<M,L,T,A>)
 * @tparam Scale Unit scale as std::ratio (default: 1/1 for SI base units)
 *
 * @code
 * Meters distance(100.0);
 * Seconds time(10.0);
 * Velocity speed = distance / time;  // Returns Velocity(10.0)
 * @endcode
 */
template <typename D, typename Scale = std::ratio<1, 1>> class Quantity {
  type_real value_;

public:
  /**
   * @brief Construct a quantity from a raw value.
   *
   * @param v Value in the specified units (default: 0.0)
   */
  KOKKOS_FUNCTION constexpr explicit Quantity(type_real v = 0.0) noexcept
      : value_(v) {}

  /**
   * @brief Extract the raw numeric value.
   *
   * Explicit method to emphasize stripping units is intentional.
   *
   * @return constexpr type_real Numeric value without units
   */
  [[nodiscard]] KOKKOS_FUNCTION constexpr type_real raw() const noexcept {
    return value_;
  }

  /**
   * @brief Explicit conversion to raw numeric value.
   *
   * @return constexpr type_real Numeric value without units
   */
  KOKKOS_FUNCTION explicit constexpr operator type_real() const noexcept {
    return value_;
  }

  /// Unary negation
  KOKKOS_FUNCTION constexpr Quantity operator-() const noexcept {
    return Quantity(-value_);
  }

  // Arithmetic within the same dimension and scale
  KOKKOS_FUNCTION constexpr Quantity operator+(Quantity o) const noexcept {
    return Quantity(value_ + o.value_);
  }
  KOKKOS_FUNCTION constexpr Quantity operator-(Quantity o) const noexcept {
    return Quantity(value_ - o.value_);
  }
  KOKKOS_FUNCTION constexpr Quantity operator*(type_real s) const noexcept {
    return Quantity(value_ * s);
  }
  KOKKOS_FUNCTION constexpr Quantity operator/(type_real s) const noexcept {
    return Quantity(value_ / s);
  }

  // Comparisons (same scale) — C++20 synthesizes !=, <, <=, >, >= from these
  // clang-format off
  KOKKOS_FUNCTION constexpr auto operator<=>(Quantity o) const noexcept {
    return value_ <=> o.value_;
  }
  // clang-format on
  KOKKOS_FUNCTION constexpr bool operator==(Quantity o) const noexcept {
    return value_ == o.value_;
  }
};

/// Scalar multiplication from left
template <typename... Args>
KOKKOS_FUNCTION constexpr Quantity<Args...>
operator*(type_real s, Quantity<Args...> q) noexcept {
  return q * s;
}

/// Scalar division from left: s / Quantity<Dim<M,L,T,A>,S> ->
/// Quantity<Dim<-M,-L,-T,-A>,S>
template <int M, int L, int T, int A, typename S>
KOKKOS_FUNCTION constexpr auto operator/(type_real s,
                                         Quantity<Dim<M, L, T, A>, S> q)
    -> Quantity<Dim<-M, -L, -T, -A>, S> {
  return Quantity<Dim<-M, -L, -T, -A>, S>(s / q.raw());
}

/**
 * @name Mixed-scale arithmetic
 * @brief Operations between quantities with different scales but same
 * dimension.
 *
 * Results are in the GCD scale of both inputs (e.g., m + km → m).
 * @{
 */

template <typename D, typename S1, typename S2>
  requires(!std::is_same_v<S1, S2>)
KOKKOS_FUNCTION constexpr auto operator+(Quantity<D, S1> a, Quantity<D, S2> b)
    -> Quantity<D, impl::ratio_gcd<S1, S2>> {
  using Rgcd = impl::ratio_gcd<S1, S2>;
  return Quantity<D, Rgcd>(a.raw() * ratio_value<std::ratio_divide<S1, Rgcd>> +
                           b.raw() * ratio_value<std::ratio_divide<S2, Rgcd>>);
}

template <typename D, typename S1, typename S2>
  requires(!std::is_same_v<S1, S2>)
KOKKOS_FUNCTION constexpr auto operator-(Quantity<D, S1> a, Quantity<D, S2> b)
    -> Quantity<D, impl::ratio_gcd<S1, S2>> {
  using Rgcd = impl::ratio_gcd<S1, S2>;
  return Quantity<D, Rgcd>(a.raw() * ratio_value<std::ratio_divide<S1, Rgcd>> -
                           b.raw() * ratio_value<std::ratio_divide<S2, Rgcd>>);
}

/// @}

/**
 * @name Mixed-scale comparisons
 * @brief Comparisons between quantities with different scales but same
 * dimension.
 *
 * C++20 synthesizes !=, <, <=, >, >= from <=> and ==.
 * @{
 */

// clang-format off
template <typename D, typename S1, typename S2>
  requires(!std::is_same_v<S1, S2>)
KOKKOS_FUNCTION constexpr auto operator<=>(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  using Rgcd = impl::ratio_gcd<S1, S2>;
  return a.raw() * ratio_value<std::ratio_divide<S1, Rgcd> > <=>
         b.raw() * ratio_value<std::ratio_divide<S2, Rgcd> >;
}
// clang-format on

template <typename D, typename S1, typename S2>
  requires(!std::is_same_v<S1, S2>)
KOKKOS_FUNCTION constexpr bool operator==(Quantity<D, S1> a,
                                          Quantity<D, S2> b) noexcept {
  using Rgcd = impl::ratio_gcd<S1, S2>;
  return a.raw() * ratio_value<std::ratio_divide<S1, Rgcd>> ==
         b.raw() * ratio_value<std::ratio_divide<S2, Rgcd>>;
}

/// @}

/**
 * @name Cross-dimensional arithmetic
 * @brief Multiply/divide quantities of different dimensions.
 *
 * Dimensional exponents add/subtract; scales multiply/divide.
 * Enables derived quantities (e.g., distance / time = velocity).
 * @{
 */

template <int M1, int L1, int T1, int A1, int M2, int L2, int T2, int A2,
          typename S1, typename S2>
KOKKOS_FUNCTION constexpr auto operator*(Quantity<Dim<M1, L1, T1, A1>, S1> a,
                                         Quantity<Dim<M2, L2, T2, A2>, S2> b)
    -> Quantity<Dim<M1 + M2, L1 + L2, T1 + T2, A1 + A2>,
                std::ratio_multiply<S1, S2>> {
  return Quantity<Dim<M1 + M2, L1 + L2, T1 + T2, A1 + A2>,
                  std::ratio_multiply<S1, S2>>(a.raw() * b.raw());
}

template <int M1, int L1, int T1, int A1, int M2, int L2, int T2, int A2,
          typename S1, typename S2>
KOKKOS_FUNCTION constexpr auto operator/(Quantity<Dim<M1, L1, T1, A1>, S1> a,
                                         Quantity<Dim<M2, L2, T2, A2>, S2> b)
    -> Quantity<Dim<M1 - M2, L1 - L2, T1 - T2, A1 - A2>,
                std::ratio_divide<S1, S2>> {
  return Quantity<Dim<M1 - M2, L1 - L2, T1 - T2, A1 - A2>,
                  std::ratio_divide<S1, S2>>(a.raw() / b.raw());
}

/// @}

/**
 * @namespace specfem::units::SI
 * @brief SI dimension definitions.
 */
namespace SI {
using DimDimensionless = Dim<0, 0, 0>; ///< Dimensionless ratios
using DimMass = Dim<1, 0, 0>;          ///< Mass (kg)
using DimLength = Dim<0, 1, 0>;        ///< Length (m)
using DimTime = Dim<0, 0, 1>;          ///< Time (s)
using DimAngle = Dim<0, 0, 0, 1>;      ///< Angle (rad)
using DimVelocity = Dim<0, 1, -1>;     ///< Velocity (m/s)

using DimFrequency = Dim<0, 0, -1>; ///< Frequency (@f$ \mathrm{s}^{-1} @f$)
using DimAngularFrequency =
    Dim<0, 0, -1, 1>; ///< Angular frequency (@f$ \mathrm{rad/s} @f$)
using DimDensity = Dim<1, -3, 0>;   ///< Density (@f$ \mathrm{kg/m^3} @f$)
using DimPressure = Dim<1, -1, -2>; ///< Pressure (Pa)

} // namespace SI

/**
 * @name Quantity type aliases
 * @brief Convenient names for common physical quantities.
 *
 * Scale defaults to std::ratio<1,1> for SI base units.
 *
 * NOTE: When adding a new quantity type alias here, also update:
 *   1. AnyQuantity variant in parse.hpp
 *   2. Parse table in parse.hpp
 *   3. Unit symbol tag in units.hpp (SPECFEM_UNIT_TAG invocation)
 * @{
 */

// Dimensionless
using Dimensionless = Quantity<SI::DimDimensionless>; ///< Unit-less ratios

// Mass
using Grams = Quantity<SI::DimMass>; ///< Mass in grams
using Kilograms =
    Quantity<SI::DimMass, std::ratio<1000, 1>>; ///< Mass in kilograms

// Time
using Seconds = Quantity<SI::DimTime>; ///< Time in seconds

// Length
using Meters = Quantity<SI::DimLength>; ///< Length in meters
using Kilometers =
    Quantity<SI::DimLength, std::ratio<1000, 1>>; ///< Length in kilometers

// Angle
using Radians = Quantity<SI::DimAngle>; ///< Angle in radians

// Density
using GramPerCubicMeter =
    Quantity<SI::DimDensity>; ///< Density (@f$ \mathrm{g/m^3} @f$)
using KilogramPerCubicMeter =
    Quantity<SI::DimDensity, std::ratio<1000, 1>>; ///< Density (@f$
                                                   ///< \mathrm{kg/m^3} @f$)

// Velocity
using MetersPerSecond = Quantity<SI::DimVelocity>; ///< Velocity (m/s)
using KilometersPerSecond =
    Quantity<SI::DimVelocity, std::ratio<1000, 1>>; ///< Velocity (km/s)

// Frequency
using Hertz = Quantity<SI::DimFrequency>; ///< Frequency in Hertz (cycles/s)
using Omega = Quantity<SI::DimAngularFrequency>; ///< Angular frequency (rad/s)

// Pressure
using Pascal = Quantity<SI::DimPressure>; ///< Pressure (Pa)
using Megapascal =
    Quantity<SI::DimPressure, std::ratio<1000000, 1>>; ///< Pressure (MPa)

/// @}

} // namespace specfem::units
