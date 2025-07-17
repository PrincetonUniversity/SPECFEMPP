#ifndef _PARAMETER_SEISMOGRAM_HPP
#define _PARAMETER_SEISMOGRAM_HPP

#include "io/seismogram/writer.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
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
   * @param output_format Outpul seismogram file format
   * @param output_folder Path to folder location where seismogram will be
   * stored
   */
  seismogram(const std::string output_format, const std::string output_folder)
      : output_format(output_format), output_folder(output_folder) {};
  /**
   * @brief Construct a new seismogram object
   *
   * @param Node YAML node describing the seismogram writer
   */
  seismogram(const YAML::Node &Node);

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param wave_type Type of wavefield (Writes .BXY for SH waves and .BXX, .BXZ
   * for P-SV waves)
   * @param dt Time interval between subsequent timesteps
   * @param t0 Solver start time
   * @param nsteps_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   * @return std::shared_ptr<specfem::io::writer> Pointer to an instantiated
   * writer object
   */
  std::shared_ptr<specfem::io::writer> instantiate_seismogram_writer(
      const specfem::enums::elastic_wave wave_type,
      const specfem::enums::electromagnetic_wave electromagnetic_wave,
      const type_real dt, const type_real t0,
      const int nsteps_between_samples) const;

private:
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
};

} // namespace runtime_configuration
} // namespace specfem

#endif
