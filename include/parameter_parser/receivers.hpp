#ifndef _RUNTIME_CONFIGURATION_RECEIVERS_HPP
#define _RUNTIME_CONFIGURATION_RECEIVERS_HPP

#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
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
  receivers(const YAML::Node &stations_node, const type_real angle,
            const int nstep_between_samples)
      : stations_node(stations_node), angle(angle),
        nstep_between_samples(nstep_between_samples){};

  receivers(const std::string stations_file, const type_real angle,
            const int nstep_between_samples);

  receivers(const YAML::Node &Node);

  /**
   * @brief Get the path of stations file
   *
   * @return std::string describing the locations of stations file
   */
  YAML::Node get_stations() const { return this->stations_node; }
  /**
   * @brief Get the angle of the receiver
   *
   * @return type_real describing the angle of the receiver
   */
  type_real get_angle() const { return this->angle; };
  /**
   * @brief Get the number of time steps between seismogram sampling
   *
   * @return int descibing seismogram sampling frequency
   */
  int get_nstep_between_samples() const { return this->nstep_between_samples; }
  /**
   * @brief Get the types of seismogram requested
   *
   * @return std::vector<specfem::seismogram::type> vector seismogram types
   */
  std::vector<specfem::enums::seismogram::type> get_seismogram_types() const {
    return stypes;
  }

private:
  YAML::Node stations_node;  ///< YAML node containing receiver information
  type_real angle;           ///< Angle of the receiver
  int nstep_between_samples; ///< Seismogram sampling frequency
  std::vector<specfem::enums::seismogram::type> stypes; ///< std::vector
                                                        ///< containing type of
                                                        ///< seismograms to be
                                                        ///< written
};
} // namespace runtime_configuration
} // namespace specfem

#endif
