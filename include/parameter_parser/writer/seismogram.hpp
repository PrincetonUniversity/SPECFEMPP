#ifndef _PARAMETER_SEISMOGRAM_HPP
#define _PARAMETER_SEISMOGRAM_HPP

#include "receiver/interface.hpp"
#include "specfem_setup.hpp"
#include "writer/interface.hpp"
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
   * @param assembly Assembly object where recievers and seismograms are stored
   */
  std::shared_ptr<specfem::writer::writer> instantiate_seismogram_writer(
      const specfem::compute::assembly &assembly) const;

private:
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
};

} // namespace runtime_configuration
} // namespace specfem

#endif
