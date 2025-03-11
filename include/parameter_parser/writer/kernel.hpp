#pragma once

#include "IO/reader.hpp"
#include "IO/writer.hpp"
#include "enumerations/simulation.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {
class kernel {
public:
  kernel(const std::string output_format, const std::string output_folder,
         const specfem::simulation::type type)
      : output_format(output_format), output_folder(output_folder),
        simulation_type(type) {}

  kernel(const YAML::Node &Node, const specfem::simulation::type type);

  std::shared_ptr<specfem::IO::writer> instantiate_kernel_writer() const;

  inline specfem::simulation::type get_simulation_type() const {
    return this->simulation_type;
  }

private:
  std::string output_format;                 ///< format of output file
  std::string output_folder;                 ///< Path to output folder
  specfem::simulation::type simulation_type; ///< Type of simulation
};
} // namespace runtime_configuration
} // namespace specfem
