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
/**
 * @brief Convert simulation type to source wavefield type.
 *
 * @param simulation_type The simulation type (forward or combined)
 * @return Corresponding field type for the source wavefield
 */
specfem::simulation::field_type
wavefield_type_from_simulation(specfem::simulation::type simulation_type);

template <specfem::element::dimension_tag DimensionTag,
          specfem::enums::source_format Format>
std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
read(const std::string &file_path, int nsteps, type_real dt,
     specfem::simulation::field_type wavefield_type);

// Explicit specialization declarations for the file-path overload. These must
// be visible before read_sources.tpp implicitly instantiates read<...>;
// otherwise the compiler instantiates the primary template first and the
// out-of-line specializations in impl/{dim2,dim3}/*.cpp become "explicit
// specialization after instantiation" errors (exposed by unity builds / LTO).
template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
read<specfem::element::dimension_tag::dim2,
     specfem::enums::source_format::CMTSOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
read<specfem::element::dimension_tag::dim2,
     specfem::enums::source_format::FORCESOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
read<specfem::element::dimension_tag::dim2,
     specfem::enums::source_format::YAML>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
read<specfem::element::dimension_tag::dim3,
     specfem::enums::source_format::CMTSOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
read<specfem::element::dimension_tag::dim3,
     specfem::enums::source_format::FORCESOLUTION>(
    const std::string &file_path, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
read<specfem::element::dimension_tag::dim3,
     specfem::enums::source_format::YAML>(
    const std::string &file_path, int nsteps, type_real dt,
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

// Explicit specialization declarations for the YAML-node overload (only the
// YAML format is valid here). Declared before any implicit instantiation for
// the same reason as the file-path overload above.
template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim2>>>
read<specfem::element::dimension_tag::dim2,
     specfem::enums::source_format::YAML>(
    const YAML::Node &source_node, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

template <>
std::vector<std::shared_ptr<
    specfem::sources::source<specfem::element::dimension_tag::dim3>>>
read<specfem::element::dimension_tag::dim3,
     specfem::enums::source_format::YAML>(
    const YAML::Node &source_node, int nsteps, type_real dt,
    specfem::simulation::field_type wavefield_type);

} // namespace sources_impl
} // namespace io
} // namespace specfem
