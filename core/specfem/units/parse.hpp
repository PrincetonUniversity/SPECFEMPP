#pragma once
#include "conversions.hpp"
#include "quantity.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace specfem::units {

// ---------------------------------------------------------------------------
// AnyQuantity — runtime-erased physical quantity
// ---------------------------------------------------------------------------

/**
 * @brief Runtime variant of all supported physical quantity types.
 *
 * Holds exactly one of the known Quantity specializations, allowing quantities
 * parsed from strings to be stored without knowing the concrete type at compile
 * time.  Use quantity_cast<To>() to recover a statically-typed value.
 *
 * @see parse()
 * @see quantity_cast()
 */
using AnyQuantity =
    std::variant<Dimensionless, Seconds, Hertz, Omega, Meters, Kilometers,
                 MetersPerSecond, KilometersPerSecond, Radians, Grams,
                 Kilograms, GramPerCubicMeter, KilogramPerCubicMeter, Pascal,
                 Megapascal>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace impl {

/**
 * @brief Detects whether unit_cast_impl<To, From>::call is well-formed.
 *
 * The primary template of unit_cast_impl<> has an empty body (no call
 * member), so the SFINAE check below correctly returns false_type for any
 * unsupported conversion pair without triggering an incomplete-type hard
 * error.
 *
 * // TODO (Lucas : CPP20) update units - fix replace void_t detection with a
 * requires clause:
 *   requires { unit_cast_impl<To, From>::call(std::declval<From>()); }
 */
template <typename To, typename From, typename = void>
struct has_unit_cast : std::false_type {};

template <typename To, typename From>
struct has_unit_cast<To, From,
                     std::void_t<decltype(unit_cast_impl<To, From>::call(
                         std::declval<From>()))> > : std::true_type {};

/**
 * @brief std::visit visitor that converts an AnyQuantity to type @p To.
 *
 * For each alternative From in AnyQuantity:
 *   - if unit_cast_impl<To, From>::call is defined  → convert
 *   - otherwise                                     → throw at runtime
 *
 * // TODO (Lucas : CPP20) update units - fix replace this struct with a
 * generic auto-lambda passed directly to std::visit once C++20 templated
 * lambdas are available:
 *   std::visit([](auto q) -> To { ... }, v)
 */
template <typename To> struct quantity_cast_visitor {
  template <typename From> To operator()(From q) const {
    if constexpr (has_unit_cast<To, From>::value) {
      return unit_cast_impl<To, From>::call(q);
    } else {
      throw std::invalid_argument(
          "specfem::units::quantity_cast: no conversion defined from the "
          "parsed unit type to the requested target type");
    }
  }
};

/**
 * @brief Strip a leading SI prefix from @p unit and return whether one was
 * found.
 *
 * On success, @p factor is set to the prefix multiplier and @p base_unit to
 * the remaining unit string.  Prefixes are tried longest-first to avoid
 * ambiguity (e.g. "mu" is matched before "m").  The ASCII spellings "u" and
 * "mu" are accepted as alternatives to "µ" (UTF-8 U+00B5 = 0xC2 0xB5).
 *
 * Supported prefixes: k (×1e3), M (×1e6), m (×1e-3), u/mu/µ (×1e-6).
 *
 * // TODO (Lucas : CPP20) update units - fix prefix scan with
 * std::string_view::starts_with instead of std::string::compare.
 */
inline bool strip_si_prefix(std::string_view unit, type_real &factor,
                            std::string &base_unit) {
  // Ordered longest-first so "mu" is matched before "m".
  // UTF-8 encoding of µ (U+00B5): 0xC2 0xB5 (2 bytes).
  static const std::pair<std::string, type_real> prefixes[] = {
    { "mu", type_real(1e-6) },
    { "\xc2\xb5", type_real(1e-6) }, // µ (U+00B5, UTF-8)
    { "u", type_real(1e-6) },
    { "M", type_real(1e6) },
    { "k", type_real(1e3) },
    { "m", type_real(1e-3) },
  };
  const std::string u(unit);
  for (const auto &[pfx, fac] : prefixes) {
    if (u.size() > pfx.size() && u.compare(0, pfx.size(), pfx) == 0) {
      factor = fac;
      base_unit = u.substr(pfx.size());
      return true;
    }
  }
  return false;
}

} // namespace impl

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

/**
 * @brief Parse a string representation of a physical quantity.
 *
 * The string must contain a decimal number (including optional sign and
 * scientific notation) optionally separated from a unit symbol by whitespace,
 * e.g.:
 * @code
 *   parse("20.0 Hz")    // Hertz(20.0)
 *   parse("1.5 km/s")   // KilometersPerSecond(1.5)
 *   parse("10.0 ms")    // Seconds(0.01)
 *   parse("2.0 kHz")    // Hertz(2000.0)
 *   parse("20.0Hz")     // Hertz(20.0)  — no whitespace is fine
 * @endcode
 *
 * Supported unit symbols (case-sensitive):
 * | Category      | Symbols                                  |
 * |---------------|------------------------------------------|
 * | Dimensionless | 1                                        |
 * | Time          | s, ms, us, µs                            |
 * | Frequency     | Hz, kHz, MHz, mHz                        |
 * | Angular freq  | rad/s                                    |
 * | Angle         | rad                                      |
 * | Length        | m, km                                    |
 * | Velocity      | m/s, km/s                                |
 * | Mass          | g, kg                                    |
 * | Density       | g/m3, kg/m3                              |
 * | Pressure      | Pa, kPa, MPa, GPa                        |
 *
 * For unit strings not in the table above, an SI prefix (k, M, m, µ/u/mu) is
 * stripped and the remainder is looked up in the same table.
 *
 * @param s  Input string.
 * @return   AnyQuantity variant holding the parsed quantity.
 * @throws   std::invalid_argument on malformed number or unknown unit.
 *
 * // TODO (Lucas : CPP20) update units - fix use a std::string_view-keyed
 * unordered_map (heterogeneous lookup, P0919R3) to avoid constructing a
 * std::string from the unit string_view on each lookup call.
 */
inline AnyQuantity parse(std::string_view s) {
  // ── Tokenise: numeric prefix + unit suffix ────────────────────────────────
  std::size_t pos = 0;

  // Skip leading whitespace
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
    ++pos;

  const std::size_t num_start = pos;

  // Optional leading sign
  if (pos < s.size() && (s[pos] == '+' || s[pos] == '-'))
    ++pos;

  // Integer digits
  while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
    ++pos;

  // Optional decimal part
  if (pos < s.size() && s[pos] == '.') {
    ++pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
  }

  // Optional exponent
  if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
    ++pos;
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-'))
      ++pos;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
      ++pos;
  }

  if (pos == num_start) {
    throw std::invalid_argument(
        "specfem::units::parse: no numeric value found in \"" + std::string(s) +
        "\"");
  }

  const std::string num_str(s.substr(num_start, pos - num_start));

  // Skip whitespace between number and unit
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
    ++pos;

  const std::string unit(s.substr(pos));

  // ── Parse number ─────────────────────────────────────────────────────────
  type_real value;
  try {
    std::size_t consumed;
    // type_real may be float or double; parse as double then cast.
    value = static_cast<type_real>(std::stod(num_str, &consumed));
    if (consumed != num_str.size())
      throw std::invalid_argument("trailing characters");
  } catch (...) {
    throw std::invalid_argument("specfem::units::parse: invalid number \"" +
                                num_str + "\" in \"" + std::string(s) + "\"");
  }

  // ── Unit lookup table ─────────────────────────────────────────────────────
  // µs key uses the UTF-8 encoding of µ (U+00B5): bytes 0xC2 0xB5.
  using Factory = std::function<AnyQuantity(type_real)>;
  // TODO (Lucas : CPP20) update units - fix use std::string_view keys with
  // transparent hash (P0919R3) to avoid constructing a std::string for each
  // table lookup.
  static const std::unordered_map<std::string, Factory> table = {
    // ── Dimensionless ────────────────────────────────────────────────────
    { "1", [](type_real v) -> AnyQuantity { return Dimensionless(v); } },

    // ── Time ─────────────────────────────────────────────────────────────
    { "s", [](type_real v) -> AnyQuantity { return Seconds(v); } },
    { "ms",
      [](type_real v) -> AnyQuantity { return Seconds(v * type_real(1e-3)); } },
    { "us",
      [](type_real v) -> AnyQuantity { return Seconds(v * type_real(1e-6)); } },
    { "\xc2\xb5s",
      [](type_real v) -> AnyQuantity { // µs (U+00B5, UTF-8)
        return Seconds(v * type_real(1e-6));
      } },

    // ── Frequency ────────────────────────────────────────────────────────
    { "Hz", [](type_real v) -> AnyQuantity { return Hertz(v); } },
    { "kHz",
      [](type_real v) -> AnyQuantity { return Hertz(v * type_real(1e3)); } },
    { "MHz",
      [](type_real v) -> AnyQuantity { return Hertz(v * type_real(1e6)); } },
    { "mHz",
      [](type_real v) -> AnyQuantity { return Hertz(v * type_real(1e-3)); } },

    // ── Angular frequency ─────────────────────────────────────────────────
    { "rad/s", [](type_real v) -> AnyQuantity { return Omega(v); } },

    // ── Angle ─────────────────────────────────────────────────────────────
    { "rad", [](type_real v) -> AnyQuantity { return Radians(v); } },

    // ── Length ────────────────────────────────────────────────────────────
    { "m", [](type_real v) -> AnyQuantity { return Meters(v); } },
    { "km", [](type_real v) -> AnyQuantity { return Kilometers(v); } },

    // ── Velocity ──────────────────────────────────────────────────────────
    { "m/s", [](type_real v) -> AnyQuantity { return MetersPerSecond(v); } },
    { "km/s",
      [](type_real v) -> AnyQuantity { return KilometersPerSecond(v); } },

    // ── Mass ──────────────────────────────────────────────────────────────
    { "g", [](type_real v) -> AnyQuantity { return Grams(v); } },
    { "kg", [](type_real v) -> AnyQuantity { return Kilograms(v); } },

    // ── Density ───────────────────────────────────────────────────────────
    { "g/m3", [](type_real v) -> AnyQuantity { return GramPerCubicMeter(v); } },
    { "kg/m3",
      [](type_real v) -> AnyQuantity { return KilogramPerCubicMeter(v); } },

    // ── Pressure ─────────────────────────────────────────────────────────
    { "Pa", [](type_real v) -> AnyQuantity { return Pascal(v); } },
    { "kPa",
      [](type_real v) -> AnyQuantity { return Pascal(v * type_real(1e3)); } },
    { "MPa", [](type_real v) -> AnyQuantity { return Megapascal(v); } },
    // GPa stored as Megapascal (raw × 1000 = MPa-equivalent raw value)
    { "GPa",
      [](type_real v) -> AnyQuantity {
        return Megapascal(v * type_real(1e3));
      } },
  };

  // Exact match
  auto it = table.find(unit);
  if (it != table.end())
    return it->second(value);

  // SI prefix fallback: strip prefix, look up base unit
  type_real prefix_factor;
  std::string base_unit;
  if (impl::strip_si_prefix(unit, prefix_factor, base_unit)) {
    auto base_it = table.find(base_unit);
    if (base_it != table.end())
      return base_it->second(value * prefix_factor);
  }

  throw std::invalid_argument("specfem::units::parse: unknown unit \"" + unit +
                              "\" in \"" + std::string(s) + "\"");
}

// ---------------------------------------------------------------------------
// quantity_cast
// ---------------------------------------------------------------------------

/**
 * @brief Convert an AnyQuantity to a specific statically-typed Quantity.
 *
 * Internally calls unit_cast_impl, so all conversions supported by unit_cast
 * are available — same-dimension scale changes (e.g. Km → m) and cross-
 * dimension spectral conversions (e.g. Hz ↔ Omega ↔ Seconds).  Unsupported
 * pairs throw at runtime.
 *
 * @tparam To  Target Quantity type (e.g. Hertz, Omega, Seconds, Meters, …).
 * @param  v   Runtime-typed AnyQuantity obtained from parse().
 * @return     Converted quantity in the target unit.
 * @throws std::invalid_argument if the held unit cannot be converted to To.
 *
 * @code
 * auto any   = specfem::units::parse("20.0 Hz");
 * auto omega = specfem::units::quantity_cast<Omega>(any); // 2π·20 rad/s
 * @endcode
 *
 * // TODO (Lucas : CPP20) update units - fix replace
 * impl::quantity_cast_visitor struct with a generic auto-lambda once C++20
 * templated lambdas are fully supported by all targeted compilers: return
 * std::visit([](auto q) -> To { ... }, v);
 */
template <typename To> To quantity_cast(const AnyQuantity &v) {
  return std::visit(impl::quantity_cast_visitor<To>{}, v);
}

/**
 * @brief Parse a string and convert directly to a specific quantity type.
 *
 * Convenience overload combining parse() and quantity_cast().
 *
 * @code
 * auto f = specfem::units::quantity_cast<Hertz>("20.0 Hz");
 * auto w = specfem::units::quantity_cast<Omega>("20.0 Hz"); // 2π·20 rad/s
 * @endcode
 *
 * @tparam To  Target Quantity type.
 * @param  s   Input string (number + unit symbol).
 * @return     Converted quantity in the target unit.
 * @throws std::invalid_argument on malformed input or unsupported conversion.
 */
template <typename To> To quantity_cast(std::string_view s) {
  return quantity_cast<To>(parse(s));
}

} // namespace specfem::units
