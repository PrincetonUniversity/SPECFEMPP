#pragma once
#include "units/conversions.hpp"
#include "units/parse.hpp"
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
 * @brief Short aliases and tag-based constructors for unit types.
 *
 * Provides convenient multiplication syntax for creating quantities.
 * Use via `using namespace` in local scopes only to avoid polluting headers.
 *
 * @code
 * using namespace specfem::units::unit_symbols;
 * auto distance = 5.0 * km;  // Kilometers{5.0}
 * auto time = 2.0 * s;       // Seconds{2.0}
 * auto speed = 10.0 * mps;   // MetersPerSecond{10.0}
 * @endcode
 */
namespace specfem::units::unit_symbols {

/// Tag for constructing Meters
struct meter_tag {};
KOKKOS_FORCEINLINE_FUNCTION constexpr meter_tag m{};

/// Tag for constructing Kilometers
struct kilometer_tag {};
KOKKOS_FORCEINLINE_FUNCTION constexpr kilometer_tag km{};

/// Tag for constructing Seconds
struct second_tag {};
KOKKOS_FORCEINLINE_FUNCTION constexpr second_tag s{};

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

/// Tag for constructing MetersPerSecond
struct m_per_s_tag {};
KOKKOS_FORCEINLINE_FUNCTION constexpr m_per_s_tag mps{};

/// Tag for constructing KilometersPerSecond
struct km_per_s_tag {};
KOKKOS_FORCEINLINE_FUNCTION constexpr km_per_s_tag kmps{};

template <typename N> constexpr MetersPerSecond operator*(N v, m_per_s_tag) {
  return MetersPerSecond(type_real(v));
}
template <typename N> constexpr MetersPerSecond operator*(m_per_s_tag, N v) {
  return MetersPerSecond(type_real(v));
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
