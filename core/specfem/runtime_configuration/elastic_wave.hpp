#pragma once

#include "enumerations/specfem_enums.hpp"
#include "specfem/utilities.hpp"
#include <string>
#include <yaml-cpp/yaml.h>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Elastic wave class to instantiate the correct elastic wave based on
 * the simulation parameters
 *
 */
class elastic_wave {
public:
  /**
   * @brief Construct a new elastic wave object
   *
   * @param elastic_wave_type Type of the elastic wave (P_SV or SH)
   */
  elastic_wave(const std::string &elastic_wave_type)
      : elastic_wave_type(elastic_wave_type) {}

  elastic_wave(const YAML::Node &Node);
  /**
   * @brief Get the type of the elastic wave
   *
   * @return std::string Type of the elastic wave (P_SV or SH)
   */
  inline specfem::enums::elastic_wave get_elastic_wave_type() const {
    if (specfem::utilities::is_psv_string(this->elastic_wave_type)) {
      return specfem::enums::elastic_wave::psv;
    } else if (specfem::utilities::is_sh_string(this->elastic_wave_type)) {
      return specfem::enums::elastic_wave::sh;
    } else {
      throw std::runtime_error("Invalid elastic wave type: " +
                               this->elastic_wave_type);
    }
  }

private:
  std::string elastic_wave_type; ///< Type of the elastic wave (P_SV or SH)
};

} // namespace runtime_configuration
} // namespace specfem
