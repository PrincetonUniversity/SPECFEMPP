#pragma once

#include "plotter.hpp"
#include "specfem/enums.hpp"
#include <boost/filesystem.hpp>

namespace specfem::periodic_tasks {
/**
 * @brief Periodic task to plot the wavefield during simulation.
 *
 * The plot_wavefield class is responsible for visualizing the wavefield
 * (e.g., displacement, velocity, acceleration) at regular intervals
 * during the time-stepping loop. It supports different output formats
 * and can handle 2D and 3D simulations through template specialization.
 *
 * This class typically interfaces with visualization libraries (like VTK)
 * to generate image files or data files for post-processing.
 *
 * @tparam DimensionTag The dimension of the simulation (dim2 or dim3).
 *
 * @see specfem::periodic_tasks::plotter
 */
template <specfem::element::dimension_tag DimensionTag> class plot_wavefield;

} // namespace specfem::periodic_tasks

#include "plot_wavefield/dim2/plot_wavefield.hpp"
#include "plot_wavefield/dim3/plot_wavefield.hpp"
