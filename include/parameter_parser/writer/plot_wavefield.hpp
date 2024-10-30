#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "writer/writer.hpp"
#include "yaml-cpp/yaml.h"
#include <string>

namespace specfem {
namespace runtime_configuration {
class plot_wavefield {

public:
  plot_wavefield(const std::string output_format,
                 const std::string output_folder,
                 const std::string wavefield_type)
      : output_format(output_format), output_folder(output_folder),
        wavefield_type(wavefield_type) {}

  plot_wavefield(const YAML::Node &Node);

  std::shared_ptr<specfem::writer::writer> instantiate_wavefield_plotter(
      const specfem::compute::assembly &assembly) const;

private:
  std::string output_format;  ///< format of output file
  std::string output_folder;  ///< Path to output folder
  std::string wavefield_type; ///< Type of wavefield to plot
};
} // namespace runtime_configuration
} // namespace specfem
