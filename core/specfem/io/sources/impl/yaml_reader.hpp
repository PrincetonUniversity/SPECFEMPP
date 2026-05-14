#pragma once

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>
#include <string>
#include <vector>

namespace specfem {
namespace io {

/**
 * @brief Read sources from a YAML source file.
 *
 * Parses a YAML file containing source definitions (force, moment-tensor, etc.)
 * and returns constructed source objects. Does NOT perform timing adjustment
 * or simulation-type validation -- those are handled by the caller.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param file_path Path to the YAML source file
 * @param nsteps Number of time steps
 * @param dt Time step
 * @param wavefield_type Source wavefield type (forward/backward)
 * @return Vector of constructed source objects
 */
template <specfem::element::dimension_tag DimensionTag>
std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
read_yaml_sources(const std::string &file_path, int nsteps, type_real dt,
                  specfem::simulation::field_type wavefield_type);

/**
 * @brief Read sources from a YAML node directly.
 *
 * Overload that accepts a YAML::Node instead of a file path. The node should
 * contain the top-level source dictionary with "sources" and
 * "number-of-sources" keys.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param source_node YAML node containing source definitions
 * @param nsteps Number of time steps
 * @param dt Time step
 * @param wavefield_type Source wavefield type (forward/backward)
 * @return Vector of constructed source objects
 */
template <specfem::element::dimension_tag DimensionTag>
std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
read_yaml_sources(const YAML::Node &source_node, int nsteps, type_real dt,
                  specfem::simulation::field_type wavefield_type);

} // namespace io
} // namespace specfem
