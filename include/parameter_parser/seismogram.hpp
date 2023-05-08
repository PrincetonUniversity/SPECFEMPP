#ifndef _PARAMETER_SEISMOGRAM_HPP
#define _PARAMETER_SEISMOGRAM_HPP

#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "writer/interface.hpp"
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
   * @param seismogram_format Outpul seismogram file format
   * @param output_folder Path to folder location where seismogram will be
   * stored
   */
  seismogram(const std::string seismogram_format,
             const std::string output_folder)
      : seismogram_format(seismogram_format), output_folder(output_folder){};
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
   * @return specfem::writer::writer* Pointer to an instantiated writer object
   */
  specfem::writer::writer *instantiate_seismogram_writer(
      std::vector<specfem::receivers::receiver *> &receivers,
      specfem::compute::receivers *compute_receivers, const type_real dt,
      const type_real t0, const int nsteps_between_samples) const;

private:
  std::string seismogram_format; ///< format of output file
  std::string output_folder;     ///< Path to output folder
};

} // namespace runtime_configuration
} // namespace specfem

#endif
