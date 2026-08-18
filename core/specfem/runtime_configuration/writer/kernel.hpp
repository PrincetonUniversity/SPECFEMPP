#pragma once

#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {

/**
 * @brief Configuration for misfit kernel writer
 *
 * Manages output format, location, and simulation type settings for
 * kernel file generation. Creates appropriate writer instances based
 * on configuration parameters.
 */
class kernel {
public:
  /**
   * @brief Construct kernel configuration from explicit parameters.
   *
   * @param output_format File format for output (e.g., "npy", "hdf5")
   * @param output_folder Directory path for output files
   * @param type Simulation type (forward/adjoint)
   */
  kernel(const std::string &output_format, const std::string &output_folder,
         const specfem::simulation::type type)
      : output_format(output_format), output_folder(output_folder),
        simulation_type(type) {}

  /**
   * @brief Construct kernel configuration from YAML node.
   *
   * @param Node YAML configuration node containing output settings
   * @param type Simulation type (2D/3D, forward/adjoint)
   */
  kernel(const YAML::Node &Node, const specfem::simulation::type type);

  /**
   * @brief Create appropriate writer instance based on configuration.
   *
   * @return Shared pointer to instantiated kernel writer
   */
  std::shared_ptr<specfem::io::writer> instantiate_kernel_writer() const;

private:
  std::string output_format; ///< Output file format (binary/ascii)
  std::string output_folder; ///< Output directory path
  specfem::simulation::type simulation_type; ///< Simulation type configuration
};
} // namespace runtime_configuration
} // namespace specfem
