#pragma once

#include "specfem/enums.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/runtime_configuration.hpp"
#include <chrono>
#include <ctime>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

/**
 * @namespace specfem::program
 * @brief Core program execution and lifecycle management
 *
 * Provides unified execution interface for 2D and 3D simulations, runtime
 * context management (Kokkos/MPI initialization), and program abort utilities.
 */
namespace specfem::program {

/**
 * @brief Print simulation header with dimension, title, and start time
 */
template <specfem::element::dimension_tag DimensionTag>
std::string
print_header(const specfem::runtime_configuration::setup &setup,
             const std::chrono::time_point<std::chrono::system_clock> now);

/**
 * @brief Print simulation end message with timing information
 */
std::string
print_end_message(std::chrono::time_point<std::chrono::system_clock> start_time,
                  std::chrono::duration<double> solver_time);

/**
 * @brief Execute complete SPECFEM simulation with runtime dimension selection
 *
 * Main entry point for the SPECFEM executable that dispatches to the
 * appropriate dimension-specific implementation (2D or 3D), which orchestrate
 * the full simulation workflow: mesh reading, database generation,
 * source/receiver configuration, assembly, time integration, and seismogram and
 * wavefield/kernel outputs.
 *
 * @param dimension Simulation dimension: "2d" or "3d"
 * @param parameter_dict User-provided YAML configuration overriding defaults
 * @param default_dict YAML default values for all simulation parameters
 * @return true on successful completion, false on failure
 *
 * @throws std::runtime_error if dimension is invalid or simulation encounters
 * fatal error
 */
bool execute(const std::string &dimension, const YAML::Node &parameter_dict);

/**
 * @brief Generate Q plot data for given Q and frequency range
 *
 * This function computes the Maxwell solid parameters (tau_sigma and tau_eps)
 * for a specified quality factor Q and frequency range, then generates data for
 * plotting the achieved Q^-1 vs frequency. The output is saved to a text file
 * in the specified output directory, which can be used with external plotting
 * tools.
 *
 * It is important to note that this executable is intended to reproduce what
 * specfem++ is internally computing in the attenuation assembly, that is the
 * maxwell solid real and imaginary factors \f$M_1\f$ and \f$M_2\f$ and the
 * resulting \f$Q^{-1}\f$, so that users can visualize the frequency dependence
 * of \f$Q^{-1}\f$ for their chosen parameters. The number of standard linear
 * solids (SLS) used in the computation is fixed at 3, which is a common choice
 * for seismic applications to capture the frequency dependence of attenuation
 * without excessive computational cost.
 *
 * The output file will contain columns for frequency, \f$Q^{-1}\f$, the real
 * modulus \f$M_1\f$, and the contributions from each standard linear solid
 * (SLS) to \f$Q^{-1}\f$ and \f$M_1\f$.
 *
 * These columns can then be used reproduce the plots of Savage et al. 2010
 * (relative phase velocity should be flipped along x axis, and \f$Q^{-1}\f$
 * should be plotted on a log scale) and Dahlen & Tromp 1998, Figure 6.8. Note
 * that for Dahlen & Tromp 1998, Figure 6.8, we need many more SLS than the
 * standard 3 to reproduce the smooth curve.
 *
 * @param Q Quality factor to compute the Maxwell solid parameters for
 * @param minfreq Minimum frequency for the inversion of tau_sigma and tau_eps
 * @param maxfreq Maximum frequency for the inversion of tau_sigma and tau_eps
 * @param min_plot_freq Minimum frequency for the output plot data
 * @param max_plot_freq Maximum frequency for the output plot data
 * @param output_dir Directory to save the output plot data file
 * @return true on successful generation of plot data, false on failure
 */
bool qplots(const type_real Q, const type_real minfreq, const type_real maxfreq,
            const type_real min_plot_freq, const type_real max_plot_freq,
            const std::string &output_dir);

} // namespace specfem::program

#include "program/abort.hpp"
#include "program/context.hpp"
#include "specfem/program.tpp"
