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
namespace sources_impl {

/**
 * @brief Read sources from a file in the specified format.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam Format Source file format (YAML, CMTSOLUTION, FORCESOLUTION)
 * @param file_path Path to the source file
 * @param nsteps Number of time steps
 * @param dt Time step
 * @param wavefield_type Source wavefield type (forward/backward)
 * @return Vector of constructed source objects
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::enums::source_format Format>
std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
read(const std::string &file_path, int nsteps, type_real dt,
     specfem::simulation::field_type wavefield_type);

/**
 * @brief Read sources from a YAML node directly.
 *
 * Only valid for YAML format.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam Format Source file format (must be YAML)
 * @param source_node YAML node containing source definitions
 * @param nsteps Number of time steps
 * @param dt Time step
 * @param wavefield_type Source wavefield type (forward/backward)
 * @return Vector of constructed source objects
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::enums::source_format Format>
std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
read(const YAML::Node &source_node, int nsteps, type_real dt,
     specfem::simulation::field_type wavefield_type);

} // namespace sources_impl
} // namespace io
} // namespace specfem
