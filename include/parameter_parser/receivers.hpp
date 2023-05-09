#ifndef _RUNTIME_CONFIGURATION_RECIEVERS_HPP
#define _RUNTIME_CONFIGURATION_RECIEVERS_HPP

#include "constants.hpp"
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
  receivers(const std::string stations_file, const int angle,
            const int nstep_between_samples)
      : stations_file(stations_file), angle(angle),
        nstep_between_samples(nstep_between_samples){};

  receivers(const YAML::Node &Node);

  /**
   * @brief Get the path of stations file
   *
   * @return std::string describing the locations of stations file
   */
  std::string get_stations_file() const { return this->stations_file; }
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
  std::vector<specfem::seismogram::type> get_seismogram_types() const {
    return stypes;
  }

private:
  std::string stations_file; ///< path to stations file
  type_real angle;           ///< Angle of the receiver
  int nstep_between_samples; ///< Seismogram sampling frequency
  std::vector<specfem::seismogram::type> stypes; ///< std::vector containing
                                                 ///< type of seismograms to be
                                                 ///< written
};
} // namespace runtime_configuration
} // namespace specfem

#endif
