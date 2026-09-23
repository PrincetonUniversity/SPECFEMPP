#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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

  /** @brief Opaque mesher-side catalog codes used only for skew detection. */
  std::vector<int> catalog_codes;

  /** @brief Opaque mesher-side catalog flags used only for skew detection. */
  std::vector<bool> catalog_flags;

  /** @brief Mesher attenuation center frequency used as a round-trip check. */
  double attenuation_source_frequency = 0.0;

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

    const bool has_catalog_metadata =
        !catalog_codes.empty() || !catalog_flags.empty();
    if (has_catalog_metadata &&
        (catalog_codes.size() != 5 || catalog_flags.size() != 16)) {
      throw std::invalid_argument(
          "specfem::globe::ModelConfig: opaque catalog metadata must contain "
          "exactly 5 codes and 16 flags");
    }

    if (attenuation && (min_attenuation_period <= 0.0 ||
                        max_attenuation_period <= min_attenuation_period)) {
      throw std::invalid_argument(
          "specfem::globe::ModelConfig: attenuation is enabled but the "
          "period band [" +
          std::to_string(min_attenuation_period) + ", " +
          std::to_string(max_attenuation_period) +
          "] is not a positive, increasing interval");
    }
    if (attenuation && attenuation_source_frequency > 0.0) {
      const double expected =
          1.0 / std::sqrt(min_attenuation_period * max_attenuation_period);
      // Avoid pulling the Kokkos-backed utilities::is_close into this header.
      if (std::abs(attenuation_source_frequency - expected) >
          1.0e-12 * std::abs(expected)) {
        throw std::invalid_argument(
            "specfem::globe::ModelConfig: attenuation source frequency "
            "disagrees with the stored period band");
      }
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
