#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specfem::globe {

/** @brief Planet selection encoded by a globe database `PLANET_TYPE`. */
enum class Planet { earth = 1, mars = 2, moon = 3 };

/** @brief Convert a database `PLANET_TYPE` to a checked planet. */
inline constexpr Planet planet_from_type(const int planet_type) {
  switch (planet_type) {
  case 1:
    return Planet::earth;
  case 2:
    return Planet::mars;
  case 3:
    return Planet::moon;
  default:
    throw std::invalid_argument("Unknown globe PLANET_TYPE " +
                                std::to_string(planet_type));
  }
}

/**
 * @brief Planet-selected interpretation of an opaque database value array.
 *
 * The Globe database records the selected planet, a planet schema version, and
 * a counted array of values. Field positions belong to that planet schema and
 * are deliberately not part of the mesh-reader contract. The array is retained
 * only long enough to verify that the model catalog derives identical values.
 */
class PlanetConstants {
public:
  /** @brief Current database schema emitted for a selected planet. */
  [[nodiscard]] static constexpr int current_schema_version(Planet planet) {
    switch (planet) {
    case Planet::earth:
    case Planet::mars:
    case Planet::moon:
      return 1;
    }
    throw std::invalid_argument("Unknown Globe planet");
  }

  /**
   * @brief Decode and validate values selected by a database planet schema.
   * @param planet Planet selected by `PLANET_TYPE`.
   * @param schema_version Version of that planet's value schema.
   * @param values Opaque schema-ordered values.
   * @return Validated transient planet constants.
   */
  [[nodiscard]] static PlanetConstants
  from_database(Planet planet, int schema_version, std::vector<double> values);

  /** @brief Selected planet. */
  [[nodiscard]] Planet planet() const noexcept { return planet_; }

  /** @brief Selected planet schema version. */
  [[nodiscard]] int schema_version() const noexcept { return schema_version_; }

  /** @brief Opaque values in the selected planet schema's canonical order. */
  [[nodiscard]] const std::vector<double> &values() const noexcept {
    return values_;
  }

  /**
   * @brief Validate database values against values derived by the model
   * catalog.
   * @param catalog_values Values in the same planet schema and canonical order.
   */
  void check_catalog_values(const std::vector<double> &catalog_values) const;

private:
  PlanetConstants(const Planet planet, const int schema_version,
                  std::vector<double> values)
      : planet_(planet), schema_version_(schema_version),
        values_(std::move(values)) {}

  static void validate_schema(Planet planet, int schema_version,
                              const std::vector<double> &values);

  Planet planet_;
  int schema_version_ = 0;
  std::vector<double> values_;
};

inline PlanetConstants
PlanetConstants::from_database(const Planet planet, const int schema_version,
                               std::vector<double> values) {
  validate_schema(planet, schema_version, values);
  return PlanetConstants(planet, schema_version, std::move(values));
}

inline void
PlanetConstants::validate_schema(const Planet planet, const int schema_version,
                                 const std::vector<double> &values) {
  if (schema_version != current_schema_version(planet)) {
    throw std::runtime_error("Unsupported planet schema version " +
                             std::to_string(schema_version));
  }

  std::size_t expected_value_count = 0;
  switch (planet) {
  case Planet::earth:
  case Planet::mars:
  case Planet::moon:
    expected_value_count = 15;
    break;
  }
  if (values.size() != expected_value_count) {
    throw std::runtime_error(
        "Planet schema " + std::to_string(schema_version) + " requires " +
        std::to_string(expected_value_count) +
        " values, but the database contains " + std::to_string(values.size()));
  }

  if (!std::all_of(values.begin(), values.end(),
                   [](const double value) { return std::isfinite(value); })) {
    throw std::runtime_error("Planet schema contains a non-finite value");
  }

  // The first six quantities are common to the current schemas. Remaining
  // positions are planet-owned catalog verification values with no global
  // geological meaning.
  if (values[0] <= 0.0 || values[1] <= 0.0 || values[2] <= 0.0 ||
      values[2] > 1.0 || values[3] <= 0.0 || values[4] <= 0.0 ||
      values[5] < 0.0) {
    throw std::runtime_error("Invalid values in planet schema");
  }
}

inline void PlanetConstants::check_catalog_values(
    const std::vector<double> &catalog_values) const {
  if (catalog_values.size() != values_.size()) {
    throw std::runtime_error(
        "Globe database planet value count disagrees with the model catalog");
  }

  constexpr double relative_tolerance = 1.0e-12;
  for (std::size_t index = 0; index < values_.size(); ++index) {
    const double expected = values_[index];
    const double actual = catalog_values[index];
    if (!std::isfinite(actual) ||
        std::abs(expected - actual) >
            relative_tolerance * std::max(1.0, std::abs(expected))) {
      std::ostringstream message;
      message << "Globe database planet value " << index << '=' << expected
              << " disagrees with model catalog value " << actual;
      throw std::runtime_error(message.str());
    }
  }
}

} // namespace specfem::globe
