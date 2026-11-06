#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::globe {

/** @brief Planet selection encoded by SPECFEM3D_GLOBE `PLANET_TYPE`. */
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
 * @brief Fixed planet constants read from the mesh database.
 *
 * The database preserves the fixed planet values and dimensional scales used by
 * the mesher. Model-dependent discontinuity radii are not stored here.
 */
class PlanetConstants {
public:
  /** @brief Current database schema emitted for a selected planet. */
  [[nodiscard]] static constexpr int current_schema_version(Planet) {
    return 2;
  }

  /**
   * @brief Decode and validate planet constants from a globe database.
   * @param planet Planet selected by `PLANET_TYPE`.
   * @param schema_version Version of the planet value schema.
   * @param values Values in the order `R_PLANET`, `RHOAV`,
   * `ONE_MINUS_F_SQUARED`, `HOURS_PER_DAY`, `SECONDS_PER_HOUR`,
   * `TOPO_MAXIMUM`.
   */
  [[nodiscard]] static PlanetConstants
  from_database(Planet planet, int schema_version, std::vector<double> values);

  /** @brief Selected planet. */
  [[nodiscard]] Planet planet() const noexcept { return planet_; }

  /** @brief Selected planet schema version. */
  [[nodiscard]] int schema_version() const noexcept { return schema_version_; }

  /** @brief Resolved model radius used to dimensionalize the database, m. */
  [[nodiscard]] double r_planet() const noexcept { return r_planet_; }

  /** @brief Average density used by the model catalog, kg/m^3. */
  [[nodiscard]] double rhoav() const noexcept { return rhoav_; }

  /** @brief Planet figure parameter \f$(1-f)^2\f$. */
  [[nodiscard]] double one_minus_f_squared() const noexcept {
    return one_minus_f_squared_;
  }

  /** @brief Rotation period component in hours per day. */
  [[nodiscard]] double hours_per_day() const noexcept { return hours_per_day_; }

  /** @brief Rotation period component in seconds per hour. */
  [[nodiscard]] double seconds_per_hour() const noexcept {
    return seconds_per_hour_;
  }

  /** @brief Maximum topographic elevation represented by the planet, m. */
  [[nodiscard]] double topo_maximum() const noexcept { return topo_maximum_; }

private:
  PlanetConstants(Planet planet, int schema_version, double r_planet,
                  double rhoav, double one_minus_f_squared,
                  double hours_per_day, double seconds_per_hour,
                  double topo_maximum)
      : planet_(planet), schema_version_(schema_version), r_planet_(r_planet),
        rhoav_(rhoav), one_minus_f_squared_(one_minus_f_squared),
        hours_per_day_(hours_per_day), seconds_per_hour_(seconds_per_hour),
        topo_maximum_(topo_maximum) {}

  Planet planet_;
  int schema_version_ = 0;
  double r_planet_ = 0.0;
  double rhoav_ = 0.0;
  double one_minus_f_squared_ = 0.0;
  double hours_per_day_ = 0.0;
  double seconds_per_hour_ = 0.0;
  double topo_maximum_ = 0.0;
};

inline PlanetConstants
PlanetConstants::from_database(const Planet planet, const int schema_version,
                               std::vector<double> values) {
  if (schema_version != current_schema_version(planet)) {
    throw std::runtime_error("Unsupported planet schema version " +
                             std::to_string(schema_version));
  }
  if (values.size() != 6) {
    throw std::runtime_error("Planet schema " + std::to_string(schema_version) +
                             " requires 6 values, but the database contains " +
                             std::to_string(values.size()));
  }
  if (!std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value) && value > 0.0;
      })) {
    throw std::runtime_error("Invalid constants in planet schema");
  }
  if (values[2] > 1.0) {
    throw std::runtime_error("Invalid ONE_MINUS_F_SQUARED in planet schema");
  }

  return PlanetConstants(planet, schema_version, values[0], values[1],
                         values[2], values[3], values[4], values[5]);
}

} // namespace specfem::globe
