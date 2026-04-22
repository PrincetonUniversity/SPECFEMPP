#pragma once
#include "specfem/units/conversions.hpp"
#include <type_traits>

namespace specfem::utilities {

/**
 * @brief A closed frequency/period band [min, max] parameterised by unit type.
 *
 * The converting constructor accepts values in any unit that has a registered
 * @c specfem::units::unit_cast specialisation, so you can write e.g.
 * @code
 *   Band<Omega>{some_seconds_value, another_seconds_value}
 * @endcode
 * and the conversion to angular frequency is applied automatically.
 *
 * @tparam T Unit type of the stored bounds (e.g. @c specfem::units::Omega).
 */
template <typename T> struct Band {
  T min, max;

  Band() = default;

  /// Same-unit construction (identity).
  constexpr Band(T lo, T hi) noexcept : min(lo), max(hi) {}

  /// Converting constructor: accepts two bounds in a different unit @p U and
  /// converts to @p T.  Bounds are swapped when the conversion is
  /// monotone-decreasing (e.g. period→omega) so the result satisfies min ≤ max.
  template <typename U,
            typename = std::enable_if_t<!std::is_same<T, U>::value> >
  constexpr Band(U lo, U hi) {
    auto clo = specfem::units::unit_cast<T>(lo);
    auto chi = specfem::units::unit_cast<T>(hi);
    if (clo.raw() <= chi.raw()) {
      min = clo;
      max = chi;
    } else {
      min = chi;
      max = clo;
    }
  }
};

/// Convert a @c Band<From> to a @c Band<To> via @c specfem::units::unit_cast.
/// Bounds are reordered to satisfy min ≤ max (handles decreasing conversions
/// such as period→omega).
template <typename To, typename From>
constexpr Band<To> unit_cast(Band<From> b) {
  return Band<To>(b.min, b.max);
}

} // namespace specfem::utilities
