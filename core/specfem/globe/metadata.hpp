#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::globe {

/** @brief Planet selection encoded by a globe database `PLANET_TYPE`. */
enum class Planet { earth = 1, mars = 2, moon = 3 };

/** @brief Resolved SI constants written by the globe mesher. */
struct PlanetConstantSet {
  double r_planet = 0.0;            ///< Mean planet radius, m.
  double rhoav = 0.0;               ///< Mean density, kg/m^3.
  double one_minus_f_squared = 0.0; ///< Geographic/geocentric datum.
  double hours_per_day = 0.0;       ///< Rotation period, hours per day.
  double seconds_per_hour = 0.0;    ///< Seconds per planet hour.
  double topo_maximum = 0.0;        ///< Maximum supported topography, m.

  /** @brief Validate values read from the globe database. */
  void validate() const;
};

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
 * All values are available immediately after database-header parsing. The
 * model oracle later verifies the scales and radii after replaying
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

  /** @brief Construct an empty placeholder for a non-globe mesh. */
  PlanetConstants() = default;

  /** @brief Construct constants resolved and written by the globe mesher. */
  PlanetConstants(const Planet planet, PlanetConstantSet values)
      : planet_(planet), values_(values) {
    values_.validate();
  }

  /** @brief Selected planet. */
  [[nodiscard]] Planet planet() const noexcept { return planet_; }

  /** @brief Resolved constants read from the globe database. */
  [[nodiscard]] const PlanetConstantSet &values() const noexcept {
    return values_;
  }

  /** @brief True when the globe database has populated the radii. */
  [[nodiscard]] bool has_radii() const noexcept { return radii_.has_value(); }

  /**
   * @brief Return model-dependent radii.
   * @throws std::logic_error before database initialization.
   */
  [[nodiscard]] const Radii &radii() const {
    if (!radii_) {
      throw std::logic_error("PlanetConstants radii are unavailable before "
                             "database initialization");
    }
    return *radii_;
  }

  /** @brief Validate radii and check an existing database value. */
  void set_radii(Radii radii) {
    radii.validate(values_.r_planet);
    if (radii_) {
      constexpr double relative_tolerance = 1.0e-12;
      const auto agrees = [relative_tolerance](const double expected,
                                               const double actual) {
        return std::abs(expected - actual) <=
               relative_tolerance * std::max(1.0, std::abs(expected));
      };
      if (!agrees(radii_->r_icb, radii.r_icb) ||
          !agrees(radii_->r_cmb, radii.r_cmb) ||
          !agrees(radii_->r_moho, radii.r_moho) ||
          !agrees(radii_->r_80, radii.r_80) ||
          !agrees(radii_->r_220, radii.r_220) ||
          !agrees(radii_->r_400, radii.r_400) ||
          !agrees(radii_->r_670, radii.r_670) ||
          !agrees(radii_->r_771, radii.r_771) ||
          !agrees(radii_->r_ocean, radii.r_ocean)) {
        throw std::runtime_error(
            "Globe database radii disagree with the model evaluator");
      }
    }
    radii_ = radii;
  }

private:
  Planet planet_ = Planet::earth;
  PlanetConstantSet values_;
  std::optional<Radii> radii_;
};

inline void PlanetConstantSet::validate() const {
  if (!std::isfinite(r_planet) || r_planet <= 0.0 || !std::isfinite(rhoav) ||
      rhoav <= 0.0 || !std::isfinite(one_minus_f_squared) ||
      one_minus_f_squared <= 0.0 || one_minus_f_squared > 1.0 ||
      !std::isfinite(hours_per_day) || hours_per_day <= 0.0 ||
      !std::isfinite(seconds_per_hour) || seconds_per_hour <= 0.0 ||
      !std::isfinite(topo_maximum) || topo_maximum < 0.0) {
    throw std::runtime_error("Invalid planet constants in globe database");
  }
}

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
 * @brief Check database scales against the initialized model evaluator.
 * @throws std::runtime_error naming both values on disagreement.
 */
inline void check_database_values(const PlanetConstants &constants,
                                  const double evaluator_r_planet,
                                  const double evaluator_rhoav) {
  constexpr double relative_tolerance = 1.0e-12;
  const auto &values = constants.values();
  if (!std::isfinite(evaluator_r_planet) ||
      std::abs(evaluator_r_planet - values.r_planet) >
          relative_tolerance * values.r_planet) {
    std::ostringstream message;
    message << "Globe database R_PLANET=" << values.r_planet
            << " disagrees with model evaluator R_PLANET="
            << evaluator_r_planet;
    throw std::runtime_error(message.str());
  }
  if (!std::isfinite(evaluator_rhoav) ||
      std::abs(evaluator_rhoav - values.rhoav) >
          relative_tolerance * values.rhoav) {
    std::ostringstream message;
    message << "Globe database RHOAV=" << values.rhoav
            << " disagrees with model evaluator RHOAV=" << evaluator_rhoav;
    throw std::runtime_error(message.str());
  }
}

} // namespace specfem::globe
