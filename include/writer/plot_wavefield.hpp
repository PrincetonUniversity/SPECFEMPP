#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "writer.hpp"
#include <boost/filesystem.hpp>
#ifdef NO_VTK
#include <sstream>
#endif

namespace specfem {
namespace writer {
/**
 * @brief Writer to plot the wavefield
 */
class plot_wavefield : public writer {
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
      : output_format(output_format), component(component),
        output_folder(output_folder), wavefield(wavefield),
        time_interval(time_interval), assembly(assembly) {
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
  void write() override;

  /**
   * @brief Returns true if the wavefield should be plotted at the current
   * timestep. Updates the internal timestep counter
   *
   * @param istep Current timestep
   * @return true if the wavefield should be plotted at the current timestep
   */
  bool compute_plotting(const int istep) {
    if (istep % time_interval == 0) {
      this->m_istep = istep;
      return true;
    }
    return false;
  }

private:
  const specfem::display::format output_format; ///< Output format of the plot
  const specfem::display::wavefield component;  ///< Component of the wavefield
  const specfem::wavefield::simulation_field wavefield; ///< Type of wavefield
                                                        ///< to plot
  const boost::filesystem::path output_folder; ///< Path to output folder
  const int time_interval;                     ///< Time interval between plots
  int m_istep = 0;                             ///< Current timestep
  specfem::compute::assembly assembly;         ///< Assembly object
};
} // namespace writer
} // namespace specfem
