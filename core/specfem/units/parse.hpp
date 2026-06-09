#pragma once
#include "conversions.hpp"
#include "quantity.hpp"

#include <concepts>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
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
                 Megapascal, DyneCentimeter, NewtonMeter>;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace impl {

/**
 * @brief Detects whether unit_cast_impl<To, From>::call is well-formed.
 *
 * The primary template of unit_cast_impl<> has an empty body (no call
 * member), so unsupported conversion pairs cause the concept to evaluate
 * to false without triggering an incomplete-type hard error.
 *
 * @tparam To Target quantity type
 * @tparam From Source quantity type
 */
template <typename To, typename From>
concept convertible_unit = requires(From f) {
  { unit_cast_impl<To, From>::call(f) } -> std::convertible_to<To>;
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
 * Supported prefixes: k (@f$ \times 10^3 @f$), M (@f$ \times 10^6 @f$),
 * m (@f$ \times 10^{-3} @f$), u/mu/µ (@f$ \times 10^{-6} @f$).
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
  for (const auto &[pfx, fac] : prefixes) {
    if (unit.size() > pfx.size() && unit.starts_with(pfx)) {
      factor = fac;
      base_unit = std::string(unit.substr(pfx.size()));
      return true;
    }
  }
  return false;
}

/// Transparent hash for heterogeneous lookup in unordered_map (P0919R3).
/// Avoids constructing a std::string when looking up by std::string_view.
struct string_hash {
  using is_transparent = void;
  size_t operator()(std::string_view sv) const noexcept {
    return std::hash<std::string_view>{}(sv);
  }
};

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
 * Supported scaling symbols are SI prefixes: k (@f$ \times 10^3 @f$),
 * M (@f$ \times 10^6 @f$), m (@f$ \times 10^{-3} @f$),
 * u/mu/µ (@f$ \times 10^{-6} @f$). The table below shows a non-exhaustive list
 * of supported unit symbols; any symbol not in the table may still be parsed if
 * the SI prefix and base unit are both supported.
 *
 * (Some) supported unit symbols (case-sensitive):
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
 */
inline AnyQuantity parse(std::string_view s) {

  // ── Tokenise: numeric prefix + unit suffix ────────────────────────────────
  const std::string str(s);
  static const std::regex num_unit_re(
      R"(^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(.*?)\s*$)");
  std::smatch m;
  if (!std::regex_match(str, m, num_unit_re) || m[1].length() == 0) {
    throw std::invalid_argument(
        "specfem::units::parse: no numeric value found in \"" + str + "\"");
  }
  const std::string num_str = m[1].str();
  const std::string unit = m[2].str();

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

#define SPECFEM_PARSE_ENTRY(str, Type)                                         \
  { str, [](type_real v) -> AnyQuantity { return Type(v); } }
#define SPECFEM_PARSE_ENTRY_SCALED(str, Type, factor)                          \
  { str,                                                                       \
    [](type_real v) -> AnyQuantity { return Type(v * type_real(factor)); } }

  static const std::unordered_map<std::string, Factory, impl::string_hash,
                                  std::equal_to<>>
      table = {
        // ── Dimensionless ────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("1", Dimensionless),

        // ── Time ─────────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("s", Seconds),
        SPECFEM_PARSE_ENTRY_SCALED("ms", Seconds, 1e-3),
        SPECFEM_PARSE_ENTRY_SCALED("us", Seconds, 1e-6),
        SPECFEM_PARSE_ENTRY_SCALED("\xc2\xb5s", Seconds, 1e-6), // µs (UTF-8)

        // ── Frequency ────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("Hz", Hertz),
        SPECFEM_PARSE_ENTRY_SCALED("kHz", Hertz, 1e3),
        SPECFEM_PARSE_ENTRY_SCALED("MHz", Hertz, 1e6),
        SPECFEM_PARSE_ENTRY_SCALED("mHz", Hertz, 1e-3),

        // ── Angular frequency
        // ─────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("rad/s", Omega),

        // ── Angle
        // ─────────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("rad", Radians),

        // ── Length
        // ────────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("m", Meters),
        SPECFEM_PARSE_ENTRY("km", Kilometers),

        // ── Velocity
        // ──────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("m/s", MetersPerSecond),
        SPECFEM_PARSE_ENTRY("km/s", KilometersPerSecond),

        // ── Mass
        // ──────────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("g", Grams),
        SPECFEM_PARSE_ENTRY("kg", Kilograms),

        // ── Density
        // ───────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("g/m3", GramPerCubicMeter),
        SPECFEM_PARSE_ENTRY("kg/m3", KilogramPerCubicMeter),

        // ── Pressure ─────────────────────────────────────────────────────────
        SPECFEM_PARSE_ENTRY("Pa", Pascal),
        SPECFEM_PARSE_ENTRY_SCALED("kPa", Pascal, 1e3),
        SPECFEM_PARSE_ENTRY("MPa", Megapascal),
        // GPa stored as Megapascal (raw x 1000 = MPa-equivalent raw value)
        SPECFEM_PARSE_ENTRY_SCALED("GPa", Megapascal, 1e3),
      };

#undef SPECFEM_PARSE_ENTRY
#undef SPECFEM_PARSE_ENTRY_SCALED

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
 * auto omega = specfem::units::quantity_cast<Omega>(any); // @f$ 2\pi \cdot 20
 * @f$ rad/s
 * @endcode
 *
 */
template <typename To> To quantity_cast(const AnyQuantity &v) {
  return std::visit(
      [](auto q) -> To {
        if constexpr (impl::convertible_unit<To, decltype(q)>) {
          return unit_cast_impl<To, decltype(q)>::call(q);
        } else {
          throw std::invalid_argument(
              "specfem::units::quantity_cast: no conversion defined from the "
              "parsed unit type to the requested target type");
        }
      },
      v);
}

/**
 * @brief Parse a string and convert directly to a specific quantity type.
 *
 * Convenience overload combining parse() and quantity_cast().
 *
 * @code
 * auto f = specfem::units::quantity_cast<Hertz>("20.0 Hz");
 * auto w = specfem::units::quantity_cast<Omega>("20.0 Hz"); // @f$ 2\pi \cdot
 * 20 @f$ rad/s
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
