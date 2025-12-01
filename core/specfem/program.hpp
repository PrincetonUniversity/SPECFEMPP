#pragma once
/**
 * @file program.hpp
 * @brief Unified SPECFEM++ program interface
 *
 * This header defines the unified program function for SPECFEM++ simulations,
 * supporting both 2D and 3D simulations through dimension-templated execution.
 *
 * The program function initializes and runs the simulation based on the
 * provided parameters and periodic tasks.
 */
#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks.hpp"

#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::program {

/**
 * @brief Execute SPECFEM program with runtime dimension selection
 *
 * @param dimension Dimension string ("2d" or "3d")
 * @param parameter_dict YAML parameter configuration
 * @param default_dict YAML default configuration
 * @param tasks Vector of periodic tasks
 * @return true if execution successful, false otherwise
 */
bool execute(
    const std::string &dimension, const YAML::Node &parameter_dict,
    const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks);

} // namespace specfem::program

#include "program/context.hpp"
