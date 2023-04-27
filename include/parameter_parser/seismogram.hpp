#ifndef _PARAMETER_SEISMOGRAM_HPP
#define _PARAMETER_SEISMOGRAM_HPP

#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "writer.h"
#include "yaml-cpp/yaml.h"
#include <tuple>

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Seismogram class is used to instantiate seismogram writer
 *
 */
class seismogram {

public:
  /**
   * @brief Construct a new seismogram object
   *
   * @param stations_file Path to stations file
   * @param angle Angle of the station
   * @param nstep_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   * @param seismogram_type Type of seismogram
   * @param output_folder Path to folder location where seismogram will be
   * stored
   */
  seismogram(const std::string stations_file, const type_real angle,
             const int nstep_between_samples,
             const std::string seismogram_format,
             const std::string output_folder)
      : stations_file(stations_file), angle(angle),
        nstep_between_samples(nstep_between_samples),
        seismogram_format(seismogram_format), output_folder(output_folder){};
  /**
   * @brief Construct a new seismogram object
   *
   * @param Node YAML node describing the seismogram writer
   */
  seismogram(const YAML::Node &Node);
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

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param receivers Vector of pointers to receiver objects used to instantiate
   * the writer
   * @param compute_receivers Pointer to specfem::compute::receivers struct used
   * to instantiate the writer
   * @param dt Time interval between timesteps
   * @param t0 Starting time of simulation
   * @return specfem::writer::writer* Pointer to an instantiated writer object
   */
  specfem::writer::writer *instantiate_seismogram_writer(
      std::vector<specfem::receivers::receiver *> &receivers,
      specfem::compute::receivers *compute_receivers, const type_real dt,
      const type_real t0) const;

private:
  std::string stations_file; ///< path to stations file
  type_real angle;           ///< Angle of the receiver
  int nstep_between_samples; ///< Seismogram sampling frequency
  std::vector<specfem::seismogram::type> stypes; ///< std::vector containing
                                                 ///< type of seismograms to be
                                                 ///< written
  std::string seismogram_format;                 ///< format of output file
  std::string output_folder;                     ///< Path to output folder
};

} // namespace runtime_configuration
} // namespace specfem

#endif
