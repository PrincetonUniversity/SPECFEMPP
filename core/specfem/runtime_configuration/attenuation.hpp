#pragma once

#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "yaml-cpp/yaml.h"
#include <optional>

namespace specfem::runtime_configuration {

/**
 * @brief Attenuation configuration defines the parameters for attenuation in
 * the simulation
 *
 */

class Attenuation {
public:
  /**
   * @brief Construct a new attenuation configuration object with default values
   *
   */
  Attenuation();

  /**
   * @brief Construct a new attenuation configuration object
   *
   * @param attenuation_enabled whether to include attenuation in the simulation
   * @param reference_frequency reference frequency for attenuation
   * @param maximum_attenuation_frequency maximum attenuation frequency
   * @param minimum_attenuation_frequency minimum attenuation frequency
   */
  Attenuation(const bool attenuation_enabled,
              const type_real reference_frequency,
              const type_real maximum_attenuation_frequency,
              const type_real minimum_attenuation_frequency)
      : reference_frequency(specfem::units::Hertz(reference_frequency)),
        attenuation_frequency_band(
            specfem::units::Hertz(minimum_attenuation_frequency),
            specfem::units::Hertz(maximum_attenuation_frequency)) {};

  /**
   * @brief Construct a new attenuation configuration object
   *
   * @param Node YAML node describing the run configuration
   */
  Attenuation(const YAML::Node &Node);

  /**
   * @brief Whether reference frequency was explicitly set in the configuration
   * @return true if reference-frequency was provided
   */
  bool has_reference_frequency() const {
    return this->reference_frequency.has_value();
  }

  /**
   * @brief Return reference frequency for attenuation.
   * @throws std::bad_optional_access if reference-frequency was not set
   */
  specfem::units::Hertz get_reference_frequency() const {
    return this->reference_frequency.value();
  }

  specfem::utilities::Band<specfem::units::Hertz>
  get_attenuation_frequency_band() const {
    return this->attenuation_frequency_band;
  }

private:
  std::optional<specfem::units::Hertz>
      reference_frequency; ///< reference frequency for attenuation (optional —
                           ///< may be read from mesh database instead)
  specfem::utilities::Band<specfem::units::Hertz>
      attenuation_frequency_band; ///< frequency band for attenuation in the
                                  ///< simulation
};

} // namespace specfem::runtime_configuration
