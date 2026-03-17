#pragma once
#include "units/conversions.hpp"
#include "units/quantity.hpp"

#include <Kokkos_Core.hpp>

/**
 * @namespace specfem::units
 * @brief Type-safe physical units and conversions.
 *
 * Provides compile-time dimensional analysis through the Quantity template,
 * preventing unit mismatches and enabling safe conversions between different
 * unit systems (e.g., meters/kilometers, seconds/hertz/omega).
 */

/**
 * @namespace specfem::units::unit_symbols
 * @brief Short aliases and literal constructors for unit types.
 *
 * Provides convenient short aliases (s, m, km) and tag-based constructors
 * for creating unit quantities. Use via `using namespace` in local scopes
 * only to avoid polluting headers.
 *
 * @code
 * using namespace specfem::units::unit_symbols;
 * auto distance = 5.0 * km_tag;  // Kilometers{5.0}
 * auto time = 2.0 * s_tag;        // Seconds{2.0}
 * @endcode
 */
namespace specfem::units::unit_symbols {

// ---------------------------------------------------------------------------
// Short type aliases (opt-in via `using namespace
// specfem::units::unit_symbols`) Use these in local scopes only — not in
// headers that others include.
// ---------------------------------------------------------------------------

using s = Seconds;                    ///< Short alias for Seconds
using m = Meters;                     ///< Short alias for Meters
using km = Kilometers;                ///< Short alias for Kilometers
using m_per_s = Velocity;             ///< Short alias for Velocity (m/s)
using km_per_s = KilometersPerSecond; ///< Short alias for KilometersPerSecond

// ---------------------------------------------------------------------------
// Construction tags — enables `1.0 * km_tag` → Kilometers{1.0} syntax
// ---------------------------------------------------------------------------

struct meter_tag {}; ///< Tag for constructing Meters via multiplication
KOKKOS_FORCEINLINE_FUNCTION constexpr meter_tag m_tag{};
struct kilometer_tag {}; ///< Tag for constructing Kilometers via multiplication
KOKKOS_FORCEINLINE_FUNCTION constexpr kilometer_tag km_tag{};
struct second_tag {}; ///< Tag for constructing Seconds via multiplication
KOKKOS_FORCEINLINE_FUNCTION constexpr second_tag s_tag{};

template <typename N> constexpr Meters operator*(N v, meter_tag) {
  return Meters(type_real(v));
}
template <typename N> constexpr Meters operator*(meter_tag, N v) {
  return Meters(type_real(v));
}
template <typename N> constexpr Kilometers operator*(N v, kilometer_tag) {
  return Kilometers(type_real(v));
}
template <typename N> constexpr Kilometers operator*(kilometer_tag, N v) {
  return Kilometers(type_real(v));
}
template <typename N> constexpr Seconds operator*(N v, second_tag) {
  return Seconds(type_real(v));
}
template <typename N> constexpr Seconds operator*(second_tag, N v) {
  return Seconds(type_real(v));
}

// Velocity tags
struct m_per_s_tag {
}; ///< Tag for constructing Velocity (m/s) via multiplication
KOKKOS_FORCEINLINE_FUNCTION constexpr m_per_s_tag mps{};
struct km_per_s_tag {
}; ///< Tag for constructing KilometersPerSecond via multiplication
KOKKOS_FORCEINLINE_FUNCTION constexpr km_per_s_tag kmps{};

template <typename N> constexpr Velocity operator*(N v, m_per_s_tag) {
  return Velocity(type_real(v));
}
template <typename N> constexpr Velocity operator*(m_per_s_tag, N v) {
  return Velocity(type_real(v));
}
template <typename N>
constexpr KilometersPerSecond operator*(N v, km_per_s_tag) {
  return KilometersPerSecond(type_real(v));
}
template <typename N>
constexpr KilometersPerSecond operator*(km_per_s_tag, N v) {
  return KilometersPerSecond(type_real(v));
}

} // namespace specfem::units::unit_symbols
