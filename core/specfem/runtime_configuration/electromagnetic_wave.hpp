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
class electromagnetic_wave {
public:
  /**
   * @brief Construct a new elastic wave object
   *
   * @param electromagnetic_wave_type Type of the electromagnetic wave (TE or
   * TM)
   */
  electromagnetic_wave(const std::string &electromagnetic_wave_type)
      : electromagnetic_wave_type(electromagnetic_wave_type) {}

  electromagnetic_wave(const YAML::Node &Node);
  /**
   * @brief Get the type of the elastic wave
   *
   * @return std::string Type of the elastic wave (P_SV or SH)
   */
  inline specfem::enums::electromagnetic_wave
  get_electromagnetic_wave_type() const {
    if (specfem::utilities::is_te_string(this->electromagnetic_wave_type)) {
      return specfem::enums::electromagnetic_wave::te;
    } else {
      throw std::runtime_error("Invalid electromagnetic wave type: " +
                               this->electromagnetic_wave_type);
    }
  }

private:
  std::string electromagnetic_wave_type; ///< Type of the electromagnetic wave
                                         ///< (TE or TM)
};

} // namespace runtime_configuration
} // namespace specfem
