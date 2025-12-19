#pragma once

#include "enumerations/display.hpp"
#include "enumerations/wavefield.hpp"
#include "plotter.hpp"
#include "specfem/assembly.hpp"
#include "specfem_mpi/interface.hpp"
#include <boost/filesystem.hpp>

namespace specfem::periodic_tasks {
/**
 * @brief Writer to plot the wavefield
 */
template <specfem::dimension::type DimensionTag> class plot_wavefield;

} // namespace specfem::periodic_tasks

#include "plot_wavefield/dim2/plot_wavefield.hpp"
#include "plot_wavefield/dim3/plot_wavefield.hpp"
