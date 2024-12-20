#pragma once

#include "compute/assembly/assembly.hpp"
#include "reader/reader.hpp"
#include "writer/writer.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {
class property {
public:
  property(const std::string output_format, const std::string output_folder)
      : output_format(output_format), output_folder(output_folder) {}

  property(const YAML::Node &Node);

  std::shared_ptr<specfem::writer::writer>
  instantiate_property_writer(const specfem::compute::assembly &assembly) const;

private:
  std::string output_format;                 ///< format of output file
  std::string output_folder;                 ///< Path to output folder
};
} // namespace runtime_configuration
} // namespace specfem
