#pragma once

#include "compute/assembly/assembly.hpp"
#include "reader/reader.hpp"
#include "writer/writer.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace runtime_configuration {
class property {
public:
  property(const std::string output_format, const std::string output_folder,
           const bool write_mode)
      : output_format(output_format), output_folder(output_folder),
        write_mode(write_mode) {}

  property(const YAML::Node &Node, const bool write_mode);

  std::shared_ptr<specfem::writer::writer>
  instantiate_property_writer(const specfem::compute::assembly &assembly) const;

  std::shared_ptr<specfem::reader::reader> instantiate_property_reader() const;

  bool has_gll_model() const;

private:
  bool write_mode;           ///< True if writing, false if reading
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
};
} // namespace runtime_configuration
} // namespace specfem
