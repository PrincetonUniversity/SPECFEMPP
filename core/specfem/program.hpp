#pragma once

#include "specfem/enums.hpp"
#include "specfem/periodic_tasks.hpp"
#include <chrono>
#include <ctime>
#include <memory>
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
 * @brief Execute complete SPECFEM simulation with runtime dimension selection
 *
 * Main entry point for the SPECFEM executable that dispatches to the
 * appropriate dimension-specific implementation (2D or 3D), which orchestrate
 * the full simulation workflow: mesh reading, database generation,
 * source/receiver configuration, assembly, time integration, and seismogram and
 * wavefield/kernel outputs.
 *
 * @param dimension Simulation dimension: "2d" or "3d"
 * @param parameter_dict User-provided YAML configuration
 * @return true on successful completion, false on failure
 *
 * @throws std::runtime_error if dimension is invalid or simulation encounters
 * fatal error
 */
bool execute(const std::string &dimension, const YAML::Node &parameter_dict);

namespace detail {
// Internal dimension-specific program implementations
void program_2d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim2> > >
        tasks);

void program_3d(
    const YAML::Node &parameter_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim3> > >
        tasks);
} // namespace detail

} // namespace specfem::program

#include "program/abort.hpp"
#include "program/context.hpp"
