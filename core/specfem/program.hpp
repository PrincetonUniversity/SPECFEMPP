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
bool execute(const std::string &dimension, const YAML::Node &parameter_dict,
             const YAML::Node &default_dict);

} // namespace specfem::program

#include "program/abort.hpp"
#include "program/context.hpp"
#include "specfem/program.tpp"
