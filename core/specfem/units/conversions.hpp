#pragma once
#include "quantity.hpp"
#include <Kokkos_Core.hpp>
#include <numbers>
#include <type_traits>

namespace specfem::units {

/**
 * @brief Internal constants for spectral conversions.
 *
 * Provides compile-time @f$ \pi @f$ values to avoid runtime dependencies.
 */
namespace impl {
constexpr type_real pi = std::numbers::pi_v<type_real>; ///< @f$ \pi @f$
constexpr type_real two_pi =
    type_real(2) * std::numbers::pi_v<type_real>; ///< @f$ 2\pi @f$
} // namespace impl

/**
 * @brief Implementation dispatcher for unit conversions.
 *
 * The primary template has no @c call member; specializations add @c call for
 * supported conversions.  Attempting @c unit_cast for an unsupported pair
 * produces a compile-time error (no member @c call).  The empty primary body
 * also allows the @c convertible_unit concept (in parse.hpp) to detect
 * unsupported pairs without triggering an incomplete-type hard error.
 *
 * @tparam To Target quantity type
 * @tparam From Source quantity type
 */
template <typename To, typename From> struct unit_cast_impl {
  // Intentionally empty: no call() member.
  // Unsupported conversions are caught at compile time by unit_cast(), or at
  // runtime by quantity_cast() via the convertible_unit concept in parse.hpp.
};

/// Identity conversion (no-op)
template <typename T> struct unit_cast_impl<T, T> {
  KOKKOS_FUNCTION static constexpr T call(T q) noexcept { return q; }
};

/// Scale conversion within same dimension (e.g., meters ↔ kilometers)
template <typename D, typename S_to, typename S_from>
  requires(!std::is_same_v<S_to, S_from>)
struct unit_cast_impl<Quantity<D, S_to>, Quantity<D, S_from>> {
  KOKKOS_FUNCTION static constexpr Quantity<D, S_to>
  call(Quantity<D, S_from> q) noexcept {
    constexpr type_real factor = ratio_value<std::ratio_divide<S_from, S_to>>;
    return Quantity<D, S_to>(q.raw() * factor);
  }
};

/**
 * @name Cross-dimensional spectral conversions
 * @brief Convert between time and frequency representations.
 *
 * Supports bidirectional conversions:
 * - Seconds ↔ Hertz: @f$ f = 1/T @f$
 * - Seconds ↔ Omega: @f$ \omega = 2\pi/T @f$
 * - Hertz ↔ Omega: @f$ \omega = 2\pi f @f$
 * @{
 */

/// Seconds → Hertz
template <> struct unit_cast_impl<Hertz, Seconds> {
  KOKKOS_FUNCTION static constexpr Hertz call(Seconds s) noexcept {
    return Hertz(type_real(1) / s.raw());
  }
};

/// Hertz → Seconds
template <> struct unit_cast_impl<Seconds, Hertz> {
  KOKKOS_FUNCTION static constexpr Seconds call(Hertz f) noexcept {
    return Seconds(type_real(1) / f.raw());
  }
};

/// Seconds → Omega
template <> struct unit_cast_impl<Omega, Seconds> {
  KOKKOS_FUNCTION static constexpr Omega call(Seconds s) noexcept {
    return Omega(impl::two_pi / s.raw());
  }
};

/// Omega → Seconds
template <> struct unit_cast_impl<Seconds, Omega> {
  KOKKOS_FUNCTION static constexpr Seconds call(Omega w) noexcept {
    return Seconds(impl::two_pi / w.raw());
  }
};

/// Hertz → Omega
template <> struct unit_cast_impl<Omega, Hertz> {
  KOKKOS_FUNCTION static constexpr Omega call(Hertz f) noexcept {
    return Omega(f.raw() * impl::two_pi);
  }
};

/// Omega → Hertz
template <> struct unit_cast_impl<Hertz, Omega> {
  KOKKOS_FUNCTION static constexpr Hertz call(Omega w) noexcept {
    return Hertz(w.raw() / impl::two_pi);
  }
};

/// @}

/**
 * @brief Convert a quantity from one unit to another.
 *
 * Performs compile-time checked unit conversion. Supports:
 * - Identity conversions (no-op)
 * - Same-dimension scale changes (m → km)
 * - Cross-dimension spectral conversions (s → Hz → @f$ \omega @f$)
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
template <typename To, typename From>
KOKKOS_FUNCTION constexpr To unit_cast(From q) {
  return unit_cast_impl<To, From>::call(q);
}

} // namespace specfem::units
