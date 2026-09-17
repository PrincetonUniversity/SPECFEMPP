#pragma once

#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::constants {

/** @brief Planet selection encoded by a globe database `PLANET_TYPE`. */
enum class Planet { earth = 1, mars = 2, moon = 3 };

/** @brief Immutable SI constants derived solely from a planet selection. */
struct PlanetConstantSet {
  double r_planet;            ///< Mean planet radius, m.
  double rhoav;               ///< Mean density, kg/m^3.
  double one_minus_f_squared; ///< Geographic/geocentric datum.
  double hours_per_day;       ///< Rotation period, hours per day.
  double seconds_per_hour;    ///< Seconds per planet hour.
  double topo_maximum;        ///< Maximum supported topography, m.
};

inline constexpr PlanetConstantSet earth{
  .r_planet = 6371000.0,
  .rhoav = 5514.3,
  .one_minus_f_squared = (1.0 - 1.0 / 299.8) * (1.0 - 1.0 / 299.8),
  .hours_per_day = 24.0,
  .seconds_per_hour = 3600.0,
  .topo_maximum = 9000.0,
};

inline constexpr PlanetConstantSet mars{
  .r_planet = 3390000.0,
  .rhoav = 3393.0,
  .one_minus_f_squared = (1.0 - 1.0 / 169.8) * (1.0 - 1.0 / 169.8),
  .hours_per_day = 24.658,
  .seconds_per_hour = 3600.0,
  .topo_maximum = 23200.0,
};

inline constexpr PlanetConstantSet moon{
  .r_planet = 1737100.0,
  .rhoav = 3344.0,
  .one_minus_f_squared = (1.0 - 1.0 / 901.0) * (1.0 - 1.0 / 901.0),
  .hours_per_day = 27.322,
  .seconds_per_hour = 3600.0,
  .topo_maximum = 10780.0,
};

/** @brief Return the immutable constant table for @p planet. */
inline constexpr const PlanetConstantSet &
planet_constant_set(const Planet planet) {
  switch (planet) {
  case Planet::earth:
    return earth;
  case Planet::mars:
    return mars;
  case Planet::moon:
    return moon;
  default:
    throw std::invalid_argument("Unknown planet " +
                                std::to_string(static_cast<int>(planet)));
  }
}

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
 * @brief Selected planet constants and guarded model-dependent radii in SI.
 *
 * Planet-wide values are available immediately after database-header parsing.
 * The model oracle populates the discontinuity radii after replaying
 * `MODEL_CONFIG`.
 */
class PlanetConstants {
public:
  /** @brief Model-dependent discontinuity radii in SI metres. */
  struct Radii {
    double r_icb = 0.0;
    double r_cmb = 0.0;
    double r_moho = 0.0;
    double r_80 = 0.0;
    double r_220 = 0.0;
    double r_400 = 0.0;
    double r_670 = 0.0;
    double r_771 = 0.0;
    double r_ocean = 0.0;

    /** @brief Validate the fundamental discontinuity ordering. */
    void validate(double r_planet) const;
  };

  /** @brief Construct Earth's constants without model-dependent radii. */
  PlanetConstants() : PlanetConstants(Planet::earth) {}

  /** @brief Select a planet's deterministic constant table. */
  explicit PlanetConstants(const Planet planet)
      : planet_(planet), values_(planet_constant_set(planet)) {}

  /** @brief Selected planet. */
  [[nodiscard]] Planet planet() const noexcept { return planet_; }

  /** @brief Immutable constants derived from the selected planet. */
  [[nodiscard]] const PlanetConstantSet &values() const noexcept {
    return values_;
  }

  /** @brief True only after the model oracle has populated the radii. */
  [[nodiscard]] bool has_radii() const noexcept { return radii_.has_value(); }

  /**
   * @brief Return model-dependent radii.
   * @throws std::logic_error before oracle initialization.
   */
  [[nodiscard]] const Radii &radii() const {
    if (!radii_) {
      throw std::logic_error(
          "PlanetConstants radii are unavailable before oracle initialization");
    }
    return *radii_;
  }

  /** @brief Validate and store radii reported by the model oracle. */
  void set_radii(Radii radii) {
    radii.validate(values_.r_planet);
    radii_ = radii;
  }

private:
  Planet planet_;
  PlanetConstantSet values_;
  std::optional<Radii> radii_;
};

inline void PlanetConstants::Radii::validate(const double r_planet) const {
  if (!(0.0 < r_icb && r_icb < r_cmb && r_cmb < r_moho && r_moho < r_planet)) {
    std::ostringstream message;
    message << "Invalid planet radii ordering: r_icb=" << r_icb
            << ", r_cmb=" << r_cmb << ", r_moho=" << r_moho
            << ", r_planet=" << r_planet;
    throw std::runtime_error(message.str());
  }
}

/**
 * @brief Check redundant database scales against a selected planet table.
 * @throws std::runtime_error naming both values on disagreement.
 */
inline void check_database_values(const PlanetConstants &constants,
                                  const double database_r_planet,
                                  const double database_rhoav) {
  constexpr double relative_tolerance = 1.0e-12;
  const auto &values = constants.values();
  if (!std::isfinite(database_r_planet) ||
      std::abs(database_r_planet - values.r_planet) >
          relative_tolerance * values.r_planet) {
    std::ostringstream message;
    message << "Globe database R_PLANET=" << database_r_planet
            << " disagrees with planet table R_PLANET=" << values.r_planet;
    throw std::runtime_error(message.str());
  }
  if (!std::isfinite(database_rhoav) ||
      std::abs(database_rhoav - values.rhoav) >
          relative_tolerance * values.rhoav) {
    std::ostringstream message;
    message << "Globe database RHOAV=" << database_rhoav
            << " disagrees with planet table RHOAV=" << values.rhoav;
    throw std::runtime_error(message.str());
  }
}

} // namespace specfem::constants
