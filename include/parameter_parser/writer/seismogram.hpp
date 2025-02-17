#ifndef _PARAMETER_SEISMOGRAM_HPP
#define _PARAMETER_SEISMOGRAM_HPP

#include "IO/seismogram/writer.hpp"
#include "receiver/interface.hpp"
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
      : output_format(output_format), output_folder(output_folder){};
  /**
   * @brief Construct a new seismogram object
   *
   * @param Node YAML node describing the seismogram writer
   */
  seismogram(const YAML::Node &Node);

  /**
   * @brief Instantiate a seismogram writer object
   *
   * @param receivers Vector of pointers to receiver objects used to instantiate
   * the writer
   * @param compute_receivers Pointer to specfem::compute::receivers struct used
   * to instantiate the writer
   * @param dt Time interval between timesteps
   * @param t0 Starting time of simulation
   * @return specfem::IO::writer* Pointer to an instantiated writer object
   */
  std::shared_ptr<specfem::IO::writer>
  instantiate_seismogram_writer(const type_real dt, const type_real t0,
                                const int nsteps_between_samples) const;

private:
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
};

} // namespace runtime_configuration
} // namespace specfem

#endif
