#pragma once

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "enumerations/wavefield.hpp"
#include "yaml-cpp/yaml.h"
#include <string>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief class to read receiver information
 *
 */
class receivers {
public:
  receivers(const YAML::Node &Node) : receivers_node(Node) {
    assert(this->receivers_node["stations"].IsDefined());
    assert(this->receivers_node["angle"].IsDefined());
    assert(this->receivers_node["nstep_between_samples"].IsDefined());
    assert(this->receivers_node["seismogram-type"].IsDefined());
  };

  /**
   * @brief Get the path of stations file
   *
   * @return std::string describing the locations of stations file
   */
  YAML::Node get_stations() const { return this->receivers_node; }
  /**
   * @brief Get the angle of the receiver
   *
   * @return type_real describing the angle of the receiver
   */
  type_real get_angle() const {
    return this->receivers_node["angle"].as<type_real>();
  };
  /**
   * @brief Get the number of time steps between seismogram sampling
   *
   * @return int descibing seismogram sampling frequency
   */
  int get_nstep_between_samples() const {
    return this->receivers_node["nstep_between_samples"].as<int>();
  }

  /**
   * @brief Get the types of seismogram requested
   *
   * @return std::vector<specfem::wavefield::type> vector seismogram types
   */
  std::vector<specfem::wavefield::type> get_seismogram_types() const;

private:
  YAML::Node receivers_node; /// Node that contains receiver information
};
} // namespace runtime_configuration
} // namespace specfem
