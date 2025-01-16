#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "plotter/plotter.hpp"
#include "yaml-cpp/yaml.h"
#include <string>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief Runtime configuration class for instantiating wavefield plotter
 *
 */
class plot_wavefield {

public:
  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new plotter configuration object
   *
   * @param output_format output format for the resulting plot (PNG, JPG)
   * @param output_folder path to the folder where the plot will be stored
   * @param wavefield_type type of wavefield to plot (displacement, velocity,
   * acceleration)
   */
  plot_wavefield(const std::string output_format,
                 const std::string output_folder, const std::string component,
                 const std::string wavefield_type, const int time_interval)
      : output_format(output_format), output_folder(output_folder),
        component(component), wavefield_type(wavefield_type),
        time_interval(time_interval) {}

  /**
   * @brief Construct a new plotter configuration object from YAML node
   *
   * @param Node YAML node describing the plotter configuration
   */
  plot_wavefield(const YAML::Node &Node);
  ///@}

  /**
   * @brief Instantiate a wavefield plotter object
   *
   * @param assembly SPECFEM++ assembly object
   * @return std::shared_ptr<specfem::IO::writer> Pointer to an instantiated
   * plotter object
   */
  std::shared_ptr<specfem::plotter::plotter> instantiate_wavefield_plotter(
      const specfem::compute::assembly &assembly) const;

private:
  std::string output_format;  ///< format of output file
  std::string output_folder;  ///< Path to output folder
  std::string component;      ///< Component of the wavefield to plot
  std::string wavefield_type; ///< Type of wavefield to plot
  int time_interval;          ///< Time interval for plotting
};
} // namespace runtime_configuration
} // namespace specfem
