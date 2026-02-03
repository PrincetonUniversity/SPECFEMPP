#pragma once

#include "enumerations/display.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem/assembly.hpp"
#include "specfem/periodic_tasks.hpp"

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
   * @param simulation_wavefield_type type of simulation wavefield to plot
   * (forward, adjoint)
   * @param component component of the wavefield to plot (x, y, z, magnitude)
   * @param time_interval time interval between subsequent plots
   * @param elastic_wave type of elastic wave simulation, ignored for 3D
   * @param electromagnetic_wave type of electromagnetic wave simulation ignored
   * for 3D
   */
  plot_wavefield(
      const std::string &output_format, const std::string &output_folder,
      const std::string &field_type,
      const std::string &simulation_wavefield_type,
      const std::string &component, const int time_interval,
      const specfem::enums::elastic_wave elastic_wave,
      const specfem::enums::electromagnetic_wave electromagnetic_wave)
      : output_format(output_format), output_folder(output_folder),
        field_type(field_type),
        simulation_wavefield_type(simulation_wavefield_type),
        component(component), time_interval(time_interval),
        elastic_wave(elastic_wave), electromagnetic_wave(electromagnetic_wave) {
  }

  /**
   * @brief Construct a new plotter configuration object from YAML node
   *
   * @param Node YAML node describing the plotter configuration
   */
  plot_wavefield(
      const YAML::Node &Node, const specfem::enums::elastic_wave elastic_wave,
      const specfem::enums::electromagnetic_wave electromagnetic_wave);
  ///@}

  /**
   * @brief Instantiate a wavefield plotter object
   *
   * @param assembly SPECFEM++ assembly object
   * @return std::shared_ptr<specfem::io::writer> Pointer to an instantiated
   * plotter object
   */
  template <specfem::dimension::type DimensionTag>
  std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> >
  instantiate_wavefield_plotter(
      const specfem::assembly::assembly<DimensionTag> &assembly,
      const type_real &dt) const;

private:
  std::string output_format; ///< format of output file
  std::string output_folder; ///< Path to output folder
  std::string field_type; ///< Component of the wavefield to plot (displacement,
                          ///< velocity, etc.)
  std::string simulation_wavefield_type; ///< Type of simulation wavefield to
                                         ///< plot ( forward, adjoint)
  std::string component;                 ///< Component of the wavefield to plot
                                         ///< (x,y,z,magnitude)
  type_real dt;                          ///< Time step
  int time_interval;                     ///< Time interval for plotting
  specfem::enums::elastic_wave elastic_wave; ///< Type of elastic wave
                                             ///< simulation
  specfem::enums::electromagnetic_wave
      electromagnetic_wave; ///< Type of
                            ///< electromagnetic
                            ///< wave
                            ///< simulation
};
} // namespace runtime_configuration
} // namespace specfem
