#pragma once
#include "specfem/setup.hpp"
#include <ratio>
#include <type_traits>

namespace specfem::units {

// Compile-time dimension: [mass, length, time, angle]
// angle distinguishes e.g. Hertz (A=0) from Omega/rad·s⁻¹ (A=1)
template <int M, int L, int T, int A = 0> struct Dim {
  static constexpr int mass = M;
  static constexpr int length = L;
  static constexpr int time = T;
  static constexpr int angle = A;
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace detail {

// Compile-time GCD — std::ratio_gcd is C++17 but absent in some libc++ builds
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

// ratio_gcd<R1,R2>: ratio whose value is gcd(R1,R2) for unit-denominator ratios
template <typename R1, typename R2>
using ratio_gcd =
    std::ratio<ct_gcd<(R1::num < 0 ? -R1::num : R1::num),
                      (R2::num < 0 ? -R2::num : R2::num)>::value,
               (R1::den / ct_gcd<R1::den, R2::den>::value) * R2::den>;

} // namespace detail

// Helper: compile-time std::ratio as a type_real value  (C++17 inline variable)
template <typename R>
inline constexpr type_real ratio_value =
    static_cast<type_real>(R::num) / static_cast<type_real>(R::den);

template <typename D, typename Scale = std::ratio<1, 1> > class Quantity {
  type_real value_;

public:
  constexpr explicit Quantity(type_real v = 0.0) noexcept : value_(v) {}

  // Explicit raw access only — stripping units is intentional
  [[nodiscard]] constexpr type_real raw() const noexcept { return value_; }
  explicit constexpr operator type_real() const noexcept { return value_; }

  // Arithmetic within the same dimension and scale
  constexpr Quantity operator+(Quantity o) const noexcept {
    return Quantity(value_ + o.value_);
  }
  constexpr Quantity operator-(Quantity o) const noexcept {
    return Quantity(value_ - o.value_);
  }
  constexpr Quantity operator*(type_real s) const noexcept {
    return Quantity(value_ * s);
  }
  constexpr Quantity operator/(type_real s) const noexcept {
    return Quantity(value_ / s);
  }
  constexpr Quantity operator*(type_real s, Quantity q) noexcept {
    return q * s;
  }

  // Comparisons (same scale)
  constexpr bool operator==(Quantity o) const noexcept {
    return value_ == o.value_;
  }
  constexpr bool operator!=(Quantity o) const noexcept {
    return value_ != o.value_;
  }
  constexpr bool operator<(Quantity o) const noexcept {
    return value_ < o.value_;
  }
  constexpr bool operator<=(Quantity o) const noexcept {
    return value_ <= o.value_;
  }
  constexpr bool operator>(Quantity o) const noexcept {
    return value_ > o.value_;
  }
  constexpr bool operator>=(Quantity o) const noexcept {
    return value_ >= o.value_;
  }
};

// ---------------------------------------------------------------------------
// Mixed-scale same-dim arithmetic (only when S1 != S2; result in GCD scale)
// ---------------------------------------------------------------------------

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr auto operator+(Quantity<D, S1> a, Quantity<D, S2> b)
    -> Quantity<D, detail::ratio_gcd<S1, S2> > {
  using Rgcd = detail::ratio_gcd<S1, S2>;
  return Quantity<D, Rgcd>(a.raw() * ratio_value<std::ratio_divide<S1, Rgcd> > +
                           b.raw() * ratio_value<std::ratio_divide<S2, Rgcd> >);
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr auto operator-(Quantity<D, S1> a, Quantity<D, S2> b)
    -> Quantity<D, detail::ratio_gcd<S1, S2> > {
  using Rgcd = detail::ratio_gcd<S1, S2>;
  return Quantity<D, Rgcd>(a.raw() * ratio_value<std::ratio_divide<S1, Rgcd> > -
                           b.raw() * ratio_value<std::ratio_divide<S2, Rgcd> >);
}

// ---------------------------------------------------------------------------
// Mixed-scale same-dim comparisons (only when S1 != S2)
// ---------------------------------------------------------------------------

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator==(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  using Rgcd = detail::ratio_gcd<S1, S2>;
  return a.raw() * ratio_value<std::ratio_divide<S1, Rgcd> > ==
         b.raw() * ratio_value<std::ratio_divide<S2, Rgcd> >;
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator!=(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  return !(a == b);
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator<(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  using Rgcd = detail::ratio_gcd<S1, S2>;
  return a.raw() * ratio_value<std::ratio_divide<S1, Rgcd> > <
         b.raw() * ratio_value<std::ratio_divide<S2, Rgcd> >;
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator<=(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  return !(b < a);
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator>(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  return b < a;
}

template <typename D, typename S1, typename S2,
          typename = std::enable_if_t<!std::is_same<S1, S2>::value> >
constexpr bool operator>=(Quantity<D, S1> a, Quantity<D, S2> b) noexcept {
  return !(a < b);
}

// ---------------------------------------------------------------------------
// Cross-dimension multiply/divide: exponents add/subtract, scales
// multiply/divide
// ---------------------------------------------------------------------------

template <int M1, int L1, int T1, int A1, int M2, int L2, int T2, int A2,
          typename S1, typename S2>
constexpr auto operator*(Quantity<Dim<M1, L1, T1, A1>, S1> a,
                         Quantity<Dim<M2, L2, T2, A2>, S2> b)
    -> Quantity<Dim<M1 + M2, L1 + L2, T1 + T2, A1 + A2>,
                std::ratio_multiply<S1, S2> > {
  return Quantity<Dim<M1 + M2, L1 + L2, T1 + T2, A1 + A2>,
                  std::ratio_multiply<S1, S2> >(a.raw() * b.raw());
}

template <int M1, int L1, int T1, int A1, int M2, int L2, int T2, int A2,
          typename S1, typename S2>
constexpr auto operator/(Quantity<Dim<M1, L1, T1, A1>, S1> a,
                         Quantity<Dim<M2, L2, T2, A2>, S2> b)
    -> Quantity<Dim<M1 - M2, L1 - L2, T1 - T2, A1 - A2>,
                std::ratio_divide<S1, S2> > {
  return Quantity<Dim<M1 - M2, L1 - L2, T1 - T2, A1 - A2>,
                  std::ratio_divide<S1, S2> >(a.raw() / b.raw());
}

// ---------------------------------------------------------------------------
// Standard SI base unit names
// ---------------------------------------------------------------------------

namespace SI {
using DimDimensionless = Dim<0, 0, 0>; // (e.g. ratios)
using DimMass = Dim<1, 0, 0>;          // g
using DimLength = Dim<0, 1, 0>;        // m
using DimTime = Dim<0, 0, 1>;          // s
using DimAngle = Dim<0, 0, 0, 1>;      // rad
using DimVelocity = Dim<0, 1, -1>;     // m / s

using DimFrequency =
    Dim<0, 0, -1>; // s⁻¹ (cycles or radians, distinguished by angle component)
using DimAngularFrequency = Dim<0, 0, -1, 1>; // rad / s
using DimDensity = Dim<1, -3, 0>;             // g / m³
using DimPressure = Dim<1, -1, -2>;           // Pa

} // namespace SI

// ---------------------------------------------------------------------------
// Named quantity aliases — Scale defaults to ratio<1,1> (SI base units)
// ---------------------------------------------------------------------------

// Dimensionless (e.g. ratios)
using Dimensionless = Quantity<SI::DimDimensionless>;

// Mass
using Grams = Quantity<SI::DimMass>;
using Kilograms = Quantity<SI::DimMass, std::ratio<1000, 1> >;

// Time
using Seconds = Quantity<SI::DimTime>;

// Length
using Meters = Quantity<SI::DimLength>;
using Kilometers = Quantity<SI::DimLength, std::ratio<1000, 1> >;

// Angle
using Radians = Quantity<SI::DimAngle>;

// Density
using GramPerCubicMeter = Quantity<SI::DimDensity>;
using KilogramPerCubicMeter = Quantity<SI::DimDensity, std::ratio<1000, 1> >;

// Velocity
using MetersPerSecond = Quantity<SI::DimVelocity>;
using KilometersPerSecond = Quantity<SI::DimVelocity, std::ratio<1000, 1> >;

// Frequency (angular, cycles etc.)
using Hertz = Quantity<SI::DimFrequency>;
using Omega = Quantity<SI::DimAngularFrequency>;

// Pressure
using Pascal = Quantity<SI::DimPressure>;
using Megapascal = Quantity<SI::DimPressure, std::ratio<1000000, 1> >;

} // namespace specfem::units
