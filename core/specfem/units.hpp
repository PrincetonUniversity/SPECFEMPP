#pragma once
#include "units/conversions.hpp"
#include "units/parse.hpp"
#include "units/quantity.hpp"

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

/// Generates a tag struct, constexpr instance, and operator* overloads for a
/// quantity type, enabling `value * tag` and `tag * value` construction syntax.
#define SPECFEM_UNIT_TAG(TagName, ShortName, QuantityType)                     \
  struct TagName##_tag {};                                                     \
  constexpr TagName##_tag ShortName{};                                         \
  template <typename N> constexpr QuantityType operator*(N v, TagName##_tag) { \
    return QuantityType(type_real(v));                                         \
  }                                                                            \
  template <typename N> constexpr QuantityType operator*(TagName##_tag, N v) { \
    return QuantityType(type_real(v));                                         \
  }

// Dimensionless
SPECFEM_UNIT_TAG(dimensionless, one, Dimensionless)

// Mass
SPECFEM_UNIT_TAG(gram, g, Grams)
SPECFEM_UNIT_TAG(kilogram, kg, Kilograms)

// Time
SPECFEM_UNIT_TAG(second, s, Seconds)

// Length
SPECFEM_UNIT_TAG(meter, m, Meters)
SPECFEM_UNIT_TAG(kilometer, km, Kilometers)

// Angle
SPECFEM_UNIT_TAG(radian, rad, Radians)

// Density
SPECFEM_UNIT_TAG(g_per_m3, gpm3, GramPerCubicMeter)
SPECFEM_UNIT_TAG(kg_per_m3, kgpm3, KilogramPerCubicMeter)

// Velocity
SPECFEM_UNIT_TAG(m_per_s, mps, MetersPerSecond)
SPECFEM_UNIT_TAG(km_per_s, kmps, KilometersPerSecond)

// Frequency
SPECFEM_UNIT_TAG(hertz, Hz, Hertz)
SPECFEM_UNIT_TAG(omega, w, Omega)

// Pressure
SPECFEM_UNIT_TAG(pascal_unit, Pa, Pascal)
SPECFEM_UNIT_TAG(megapascal, MPa, Megapascal)

// Torque / seismic moment
SPECFEM_UNIT_TAG(dyne_centimeter, dyn_cm, DyneCentimeter)
SPECFEM_UNIT_TAG(newton_meter, Nm, NewtonMeter)

#undef SPECFEM_UNIT_TAG

} // namespace specfem::units::unit_symbols
