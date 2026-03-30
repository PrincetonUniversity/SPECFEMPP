#pragma once

#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
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
        maximum_attenuation_frequency(-1.0),
        minimum_attenuation_frequency(-1.0) {};

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
        maximum_attenuation_frequency(maximum_attenuation_frequency),
        minimum_attenuation_frequency(minimum_attenuation_frequency) {};

  /**
   * @brief Construct a new attenuation configuration object
   *
   * @param Node YAML node describing the run configuration
   */
  Attenuation(const YAML::Node &Node);

  bool is_attenuation_enabled() const { return this->attenuation_enabled; }
  type_real get_reference_frequency() const {
    return this->reference_frequency;
  }
  type_real get_maximum_attenuation_frequency() const {
    return this->maximum_attenuation_frequency;
  }
  type_real get_minimum_attenuation_frequency() const {
    return this->minimum_attenuation_frequency;
  }

private:
  bool attenuation_enabled;      ///< whether to include attenuation in the
                                 ///< simulation
  type_real reference_frequency; ///< whether to include reference frequency in
                                 ///< the simulation
  type_real maximum_attenuation_frequency; ///< whether to include maximum
                                           ///< attenuation frequency in the
                                           ///< simulation
  type_real minimum_attenuation_frequency; ///< whether to include minimum
                                           ///< attenuation frequency in the
                                           ///< simulation
};

} // namespace specfem::runtime_configuration
