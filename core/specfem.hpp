
#pragma once

/**
 * @file specfem.hpp
 * @brief Unified SPECFEM++ header providing access to core functionality
 *
 * This header provides a clean interface to the unified SPECFEM++ architecture,
 * supporting both 2D and 3D simulations through dimension-templated execution.
 *
 * Example usage:
 *   specfem 2d -p config.yaml
 *   specfem 3d -p config.yaml
 */
#include <boost/program_options.hpp>

/**
 * @brief Define command line argument options for the unified executable
 *
 * @return Boost program options description
 */
boost::program_options::options_description define_args();

/**
 * @brief Parse command line arguments and validate input
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @param vm Variables map to store parsed arguments
 * @param dimension Output parameter for the dimension (2d/3d)
 * @return 1 if successful, 0 if help requested, -1 if error
 */
int parse_args(int argc, char **argv, boost::program_options::variables_map &vm,
               std::string &dimension);
