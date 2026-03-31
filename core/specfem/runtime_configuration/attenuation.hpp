#pragma once

#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include "yaml-cpp/yaml.h"

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
  Attenuation()
      : attenuation_enabled(false), reference_frequency(-1.0),
        attenuation_frequency_band(specfem::units::Hertz(-10.0),
                                   specfem::units::Hertz(-1.0)) {};

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
      : attenuation_enabled(attenuation_enabled),
        reference_frequency(reference_frequency),
        attenuation_frequency_band(
            specfem::units::Hertz(minimum_attenuation_frequency),
            specfem::units::Hertz(maximum_attenuation_frequency)) {};

  /**
   * @brief Construct a new attenuation configuration object
   *
   * @param Node YAML node describing the run configuration
   */
  Attenuation(const YAML::Node &Node);

  bool is_attenuation_enabled() const { return this->attenuation_enabled; }
  specfem::units::Hertz get_reference_frequency() const {
    return specfem::units::Hertz(this->reference_frequency);
  }
  specfem::utilities::Band<specfem::units::Hertz>
  get_attenuation_frequency_band() const {
    return this->attenuation_frequency_band;
  }

private:
  bool attenuation_enabled; ///< whether to include attenuation in the
                            ///< simulation
  specfem::units::Hertz reference_frequency; ///< whether to include reference
                                             ///< frequency in the simulation
  specfem::utilities::Band<specfem::units::Hertz>
      attenuation_frequency_band; ///< frequency band for attenuation in the
                                  ///< simulation
};

} // namespace specfem::runtime_configuration
