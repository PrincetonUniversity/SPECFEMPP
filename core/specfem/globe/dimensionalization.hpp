#pragma once

#include "specfem/globe/planet_constants.hpp"
#include "specfem/units.hpp"

#include <type_traits>

namespace specfem::globe {

/**
 * @brief Convert an SI length to globe non-dimensional units.
 * @tparam Scale Length scale relative to meters.
 * @param value SI length.
 * @param constants Explicit planet scales.
 * @return The non-dimensional value.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename Scale>
specfem::units::Dimensionless nondimensionalize(
    const specfem::units::Quantity<specfem::units::SI::DimLength, Scale> value,
    const specfem::globe::PlanetConstants &constants) {
  const double value_si = value.raw() * specfem::units::ratio_value<Scale>;
  const auto &values = constants.values();
  return specfem::units::Dimensionless(value_si / values.r_planet);
}

/**
 * @brief Convert an SI density to globe non-dimensional units.
 * @tparam Scale Density scale relative to grams per cubic meter.
 * @param value SI density.
 * @param constants Explicit planet scales.
 * @return The non-dimensional value.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename Scale>
specfem::units::Dimensionless nondimensionalize(
    const specfem::units::Quantity<specfem::units::SI::DimDensity, Scale> value,
    const specfem::globe::PlanetConstants &constants) {
  const double value_base = value.raw() * specfem::units::ratio_value<Scale>;
  const auto &values = constants.values();
  // The units package uses grams as its mass base. Convert rhoav from kg/m^3
  // to the corresponding raw base value.
  return specfem::units::Dimensionless(value_base / (1000.0 * values.rhoav));
}

/**
 * @brief Convert a globe non-dimensional value to an SI length.
 * @tparam QuantityType `specfem::units::Meters` (scaled variants are also
 * accepted).
 * @param value Non-dimensional value.
 * @param constants Explicit planet scales.
 * @return The requested physical quantity.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename QuantityType>
  requires(std::is_same_v<typename QuantityType::dimension_type,
                          specfem::units::SI::DimLength>)
QuantityType dimensionalize(const specfem::units::Dimensionless value,
                            const specfem::globe::PlanetConstants &constants) {
  using Scale = typename QuantityType::scale_type;
  const auto &values = constants.values();
  const double value_base = value.raw() * values.r_planet;
  return QuantityType(value_base / specfem::units::ratio_value<Scale>);
}

/**
 * @brief Convert a globe non-dimensional value to an SI density.
 * @tparam QuantityType `specfem::units::KilogramPerCubicMeter` (scaled variants
 * are also accepted).
 * @param value Non-dimensional value.
 * @param constants Explicit planet scales.
 * @return The requested physical quantity.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename QuantityType>
  requires(std::is_same_v<typename QuantityType::dimension_type,
                          specfem::units::SI::DimDensity>)
QuantityType dimensionalize(const specfem::units::Dimensionless value,
                            const specfem::globe::PlanetConstants &constants) {
  using Scale = typename QuantityType::scale_type;
  const auto &values = constants.values();
  // The units package uses grams as its mass base. Convert rhoav from kg/m^3
  // to the corresponding raw base value.
  const double value_base = value.raw() * 1000.0 * values.rhoav;
  return QuantityType(value_base / specfem::units::ratio_value<Scale>);
}

} // namespace specfem::globe
