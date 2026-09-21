#pragma once

#include "specfem/globe/planet_constants.hpp"
#include "specfem/units.hpp"

#include <type_traits>

namespace specfem::globe {

/**
 * @brief Convert an SI length or density to globe non-dimensional units.
 * @param value SI length or density.
 * @param constants Explicit planet scales.
 * @return The non-dimensional value.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename Dimension, typename Scale>
  requires(std::is_same_v<Dimension, specfem::units::SI::DimLength> ||
           std::is_same_v<Dimension, specfem::units::SI::DimDensity>)
specfem::units::Dimensionless
nondimensionalize(const specfem::units::Quantity<Dimension, Scale> value,
                  const specfem::globe::PlanetConstants &constants) {
  const double value_si = value.raw() * specfem::units::ratio_value<Scale>;
  const auto &values = constants.values();
  if constexpr (std::is_same_v<Dimension, specfem::units::SI::DimLength>) {
    return specfem::units::Dimensionless(value_si / values.r_planet);
  } else {
    // The units package uses grams as its mass base. Convert kg/m^3 to the
    // corresponding raw base value before comparing with rhoav in kg/m^3.
    return specfem::units::Dimensionless(value_si / (1000.0 * values.rhoav));
  }
}

/**
 * @brief Convert a globe non-dimensional value to an SI length or density.
 * @tparam QuantityType `specfem::units::Meters` or
 * `specfem::units::KilogramPerCubicMeter` (scaled variants are also accepted).
 * @param value Non-dimensional value.
 * @param constants Explicit planet scales.
 * @return The requested physical quantity.
 *
 * @warning Calling this outside the globe model evaluator
 * is a units-contract bug. The database and assembly both store SI values.
 */
template <typename QuantityType>
  requires(std::is_same_v<typename QuantityType::dimension_type,
                          specfem::units::SI::DimLength> ||
           std::is_same_v<typename QuantityType::dimension_type,
                          specfem::units::SI::DimDensity>)
QuantityType dimensionalize(const specfem::units::Dimensionless value,
                            const specfem::globe::PlanetConstants &constants) {
  using Dimension = typename QuantityType::dimension_type;
  using Scale = typename QuantityType::scale_type;
  const auto &values = constants.values();
  double value_base = value.raw() * values.r_planet;
  if constexpr (std::is_same_v<Dimension, specfem::units::SI::DimDensity>) {
    value_base = value.raw() * 1000.0 * values.rhoav;
  }
  return QuantityType(value_base / specfem::units::ratio_value<Scale>);
}

} // namespace specfem::globe
