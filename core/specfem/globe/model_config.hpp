#pragma once

#include <stdexcept>
#include <string>

namespace specfem::globe {

/**
 * @brief Mesher-resolved model selection replayed by the globe evaluator.
 *
 * Every field is read from the database's `MODEL_CONFIG` records. Defaults are
 * deliberately invalid so an incomplete read cannot silently select a
 * plausible but different model.
 */
struct ModelConfig {
  std::string model_name; ///< Database `MODEL` value.

  int planet_type = 0; ///< Database `PLANET_TYPE` value.
  int nchunks = 0;     ///< Number of globe chunks.
  int nex_xi = 0;      ///< Elements along the chunk xi direction.
  int nex_eta = 0;     ///< Elements along the chunk eta direction.

  bool ellipticity = false;
  bool topography = false;
  bool oceans = false;
  bool attenuation = false;
  bool gravity = false;
  bool rotation = false;

  double min_attenuation_period = 0.0; ///< Minimum period, s.
  double max_attenuation_period = 0.0; ///< Maximum period, s.

  /** @brief Reject a configuration that was not fully populated. */
  void validate() const {
    if (model_name.empty()) {
      throw std::invalid_argument(
          "specfem::globe::ModelConfig: model_name is empty; it must come "
          "from the MODEL record of the mesh database");
    }
    require_positive(planet_type, "planet_type");
    require_positive(nchunks, "nchunks");
    require_positive(nex_xi, "nex_xi");
    require_positive(nex_eta, "nex_eta");

    if (attenuation && (min_attenuation_period <= 0.0 ||
                        max_attenuation_period <= min_attenuation_period)) {
      throw std::invalid_argument(
          "specfem::globe::ModelConfig: attenuation is enabled but the "
          "period band [" +
          std::to_string(min_attenuation_period) + ", " +
          std::to_string(max_attenuation_period) +
          "] is not a positive, increasing interval");
    }
  }

private:
  static void require_positive(const int value, const std::string &field) {
    if (value <= 0) {
      throw std::invalid_argument(
          "specfem::globe::ModelConfig: " + field + " is " +
          std::to_string(value) +
          "; it must be read from the mesh database, which is the only place "
          "the mesher's value exists");
    }
  }
};

} // namespace specfem::globe
