#pragma once
#include "quantity.hpp"
#include <type_traits>

namespace specfem::units {

/**
 * @brief Internal constants for spectral conversions.
 *
 * Provides compile-time π values to avoid runtime dependencies.
 */
namespace impl {
// TODO (Lucas : CPP20): Replace with std::numbers::pi when available
constexpr type_real pi =
    type_real(3.141592653589793238462643383279502884L); ///< π
constexpr type_real two_pi =
    type_real(6.283185307179586476925286766559005768L); ///< 2π
} // namespace impl

/**
 * @brief Implementation dispatcher for unit conversions.
 *
 * The primary template has no @c call member; specializations add @c call for
 * supported conversions.  Attempting @c unit_cast for an unsupported pair
 * produces a compile-time error (no member @c call).  The empty primary body
 * also allows @c has_unit_cast<To,From> (in parse.hpp) to detect unsupported
 * pairs via SFINAE without triggering an incomplete-type hard error.
 *
 * @tparam To Target quantity type
 * @tparam From Source quantity type
 */
template <typename To, typename From, typename = void> struct unit_cast_impl {
  // Intentionally empty: no call() member.
  // Unsupported conversions are caught at compile time by unit_cast(), or at
  // runtime by quantity_cast() via has_unit_cast<> detection in parse.hpp.
};

/// Identity conversion (no-op)
template <typename T> struct unit_cast_impl<T, T, void> {
  static constexpr T call(T q) noexcept { return q; }
};

/// Scale conversion within same dimension (e.g., meters ↔ kilometers)
template <typename D, typename S_to, typename S_from>
struct unit_cast_impl<Quantity<D, S_to>, Quantity<D, S_from>,
                      std::enable_if_t<!std::is_same<S_to, S_from>::value> > {
  static constexpr Quantity<D, S_to> call(Quantity<D, S_from> q) noexcept {
    constexpr type_real factor = ratio_value<std::ratio_divide<S_from, S_to> >;
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
template <> struct unit_cast_impl<Hertz, Seconds, void> {
  static constexpr Hertz call(Seconds s) noexcept {
    return Hertz(type_real(1) / s.raw());
  }
};

/// Hertz → Seconds
template <> struct unit_cast_impl<Seconds, Hertz, void> {
  static constexpr Seconds call(Hertz f) noexcept {
    return Seconds(type_real(1) / f.raw());
  }
};

/// Seconds → Omega
template <> struct unit_cast_impl<Omega, Seconds, void> {
  static constexpr Omega call(Seconds s) noexcept {
    return Omega(impl::two_pi / s.raw());
  }
};

/// Omega → Seconds
template <> struct unit_cast_impl<Seconds, Omega, void> {
  static constexpr Seconds call(Omega w) noexcept {
    return Seconds(impl::two_pi / w.raw());
  }
};

/// Hertz → Omega
template <> struct unit_cast_impl<Omega, Hertz, void> {
  static constexpr Omega call(Hertz f) noexcept {
    return Omega(f.raw() * impl::two_pi);
  }
};

/// Omega → Hertz
template <> struct unit_cast_impl<Hertz, Omega, void> {
  static constexpr Hertz call(Omega w) noexcept {
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
