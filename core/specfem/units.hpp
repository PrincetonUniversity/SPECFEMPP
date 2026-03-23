#pragma once
#include "units/conversions.hpp"
#include "units/quantity.hpp"

namespace specfem::units::unit_symbols {

// ---------------------------------------------------------------------------
// Short type aliases (opt-in via `using namespace
// specfem::units::unit_symbols`) Use these in local scopes only — not in
// headers that others include.
// ---------------------------------------------------------------------------

using s = Seconds;
using m = Meters;
using km = Kilometers;
using mps = MetersPerSecond;
using kmps = KilometersPerSecond;

// ---------------------------------------------------------------------------
// Construction tags — enables `1.0 * km_tag` → Kilometers{1.0} syntax
// ---------------------------------------------------------------------------

struct meter_tag {};
inline constexpr meter_tag m_tag{};
struct kilometer_tag {};
inline constexpr kilometer_tag km_tag{};
struct second_tag {};
inline constexpr second_tag s_tag{};

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
struct meters_per_second_tag {};
inline constexpr meters_per_second_tag mps_tag{};
struct kilometers_per_second_tag {};
inline constexpr kilometers_per_second_tag kmps_tag{};

template <typename N>
constexpr MetersPerSecond operator*(N v, meters_per_second_tag) {
  return MetersPerSecond(type_real(v));
}
template <typename N>
constexpr MetersPerSecond operator*(meters_per_second_tag, N v) {
  return MetersPerSecond(type_real(v));
}
template <typename N>
constexpr KilometersPerSecond operator*(N v, kilometers_per_second_tag) {
  return KilometersPerSecond(type_real(v));
}
template <typename N>
constexpr KilometersPerSecond operator*(kilometers_per_second_tag, N v) {
  return KilometersPerSecond(type_real(v));
}

} // namespace specfem::units::unit_symbols
