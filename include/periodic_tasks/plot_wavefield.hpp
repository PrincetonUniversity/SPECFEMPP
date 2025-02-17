#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "plotter.hpp"
#include <boost/filesystem.hpp>
#ifdef NO_VTK
#include <sstream>
#endif

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Writer to plot the wavefield
 */
class plot_wavefield : public plotter {
public:
  /**
   * @brief Construct a new plotter object
   *
   * @param assembly SPECFFEM++ assembly object
   * @param output_format Output format of the plot (PNG, JPG, etc.)
   * @param component Component of the wavefield to plot (displacement,
   * velocity, etc.)
   * @param wavefield Type of wavefield to plot (forward, adjoint, etc.)
   * @param time_interval Time interval between subsequent plots
   * @param output_folder Path to output folder where plots will be stored
   */
  plot_wavefield(const specfem::compute::assembly &assembly,
                 const specfem::display::format &output_format,
                 const specfem::display::wavefield &component,
                 const specfem::wavefield::simulation_field &wavefield,
                 const int &time_interval,
                 const boost::filesystem::path &output_folder)
      : plotter(time_interval), output_format(output_format),
        component(component), output_folder(output_folder),
        wavefield(wavefield), assembly(assembly) {
#ifdef NO_VTK
    std::ostringstream message;
    message << "Display section is not enabled, since SPECFEM++ was built "
               "without VTK\n"
            << "Please install VTK and rebuild SPECFEM++ with "
               "-DVTK_DIR=/path/to/vtk";
    throw std::runtime_error(message.str());
#endif
  }

  /**
   * @brief Plot the wavefield
   *
   */
  void run() override;

private:
  const specfem::display::format output_format; ///< Output format of the plot
  const specfem::display::wavefield component;  ///< Component of the wavefield
  const specfem::wavefield::simulation_field wavefield; ///< Type of wavefield
                                                        ///< to plot
  const boost::filesystem::path output_folder; ///< Path to output folder
  specfem::compute::assembly assembly;         ///< Assembly object
};
} // namespace periodic_tasks
} // namespace specfem
