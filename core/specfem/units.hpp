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
using m_per_s = Velocity;
using km_per_s = KilometersPerSecond;

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
struct m_per_s_tag {};
inline constexpr m_per_s_tag mps{};
struct km_per_s_tag {};
inline constexpr km_per_s_tag kmps{};

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
