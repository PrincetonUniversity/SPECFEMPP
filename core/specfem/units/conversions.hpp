#pragma once
#include "quantity.hpp"
#include <type_traits>

namespace specfem::units {

// constexpr 2π — avoids depending on the non-constexpr specfem::constants::pi
namespace impl {
constexpr type_real pi = 2 * std::atan(type_real(1.0));
constexpr type_real two_pi = 4 * std::arctan(type_real(1.0));
} // namespace impl

// ---------------------------------------------------------------------------
// unit_cast_impl — primary is undefined; partial/full specializations below.
// Unsupported conversion pairs produce a clean "incomplete type" compile error.
// ---------------------------------------------------------------------------

template <typename To, typename From, typename = void> struct unit_cast_impl;

// Identity: no-op
template <typename T> struct unit_cast_impl<T, T, void> {
  static constexpr T call(T q) noexcept { return q; }
};

// Same-dim scale conversion (only when To and From have the same D but
// different Scale)
template <typename D, typename S_to, typename S_from>
struct unit_cast_impl<Quantity<D, S_to>, Quantity<D, S_from>,
                      std::enable_if_t<!std::is_same<S_to, S_from>::value> > {
  static constexpr Quantity<D, S_to> call(Quantity<D, S_from> q) noexcept {
    constexpr type_real factor = ratio_value<std::ratio_divide<S_from, S_to> >;
    return Quantity<D, S_to>(q.raw() * factor);
  }
};

// ---------------------------------------------------------------------------
// Cross-dim conversions (full specializations — different D types, so the
// same-dim partial spec above never matches these)
// ---------------------------------------------------------------------------

// Seconds <-> Hertz
template <> struct unit_cast_impl<Hertz, Seconds, void> {
  static constexpr Hertz call(Seconds s) noexcept {
    return Hertz(type_real(1) / s.raw());
  }
};

template <> struct unit_cast_impl<Seconds, Hertz, void> {
  static constexpr Seconds call(Hertz f) noexcept {
    return Seconds(type_real(1) / f.raw());
  }
};

// Seconds <-> Omega
template <> struct unit_cast_impl<Omega, Seconds, void> {
  static constexpr Omega call(Seconds s) noexcept {
    return Omega(impl::two_pi / s.raw());
  }
};

template <> struct unit_cast_impl<Seconds, Omega, void> {
  static constexpr Seconds call(Omega w) noexcept {
    return Seconds(impl::two_pi / w.raw());
  }
};

// Hertz <-> Omega
template <> struct unit_cast_impl<Omega, Hertz, void> {
  static constexpr Omega call(Hertz f) noexcept {
    return Omega(f.raw() * impl::two_pi);
  }
};

template <> struct unit_cast_impl<Hertz, Omega, void> {
  static constexpr Hertz call(Omega w) noexcept {
    return Hertz(w.raw() / impl::two_pi);
  }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @brief Convert a quantity from one unit to another.
 *
 * Performs compile-time checked unit conversion. Supports:
 * - Identity conversions (no-op)
 * - Same-dimension scale changes (m → km)
 * - Cross-dimension spectral conversions (s → Hz → ω)
 *
 * Unsupported conversions produce a compile-time error.
 *
 * @tparam To Target quantity type
 * @tparam From Source quantity type
 * @param q Quantity to convert
 * @return constexpr To Converted quantity
 *
 * @code
 * auto km = unit_cast<Kilometers>(Meters(1500.0));  // 1.5 km
 * auto freq = unit_cast<Hertz>(Seconds(0.5));       // 2.0 Hz
 * @endcode
 */
template <typename To, typename From> constexpr To unit_cast(From q) {
  return unit_cast_impl<To, From>::call(q);
}

} // namespace specfem::units
