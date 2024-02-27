#ifndef _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP
#define _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP

#include "compute/compute_assembly.hpp"
#include "writer/writer.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {
class wavefield {

public:
  wavefield(const std::string wavefield_format, const std::string output_folder)
      : wavefield_format(wavefield_format), output_folder(output_folder){};

  wavefield(const YAML::Node &Node);

  std::shared_ptr<specfem::writer::writer> instantiate_wavefield_writer(
      const specfem::compute::assembly &assembly) const;

private:
  std::string wavefield_format; ///< format of output file
  std::string output_folder;    ///< Path to output folder
};
} // namespace runtime_configuration
} // namespace specfem

#endif /* _SPECFEM_RUNTIME_CONFIGURATION_WAVEFIELD_HPP */
