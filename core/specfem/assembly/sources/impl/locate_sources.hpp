#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

#include <memory>
#include <vector>

namespace specfem::assembly::sources_impl {

/**
 * @brief Locate seismic sources within the finite element mesh
 *
 * Maps source global coordinates to local element coordinates and assigns
 * medium tags based on element classification. Sources given generic
 * coordinates must already have been resolved to global coordinates (via
 * @ref specfem::assembly::to) before this call.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 *
 * @param element_types Element classification data (medium, property, boundary
 * types)
 * @param mesh Finite element mesh with coordinates and connectivity
 * @param sources [in,out] Source objects to locate. Input: global coordinates
 * and time functions. Output: assigned element indices and medium tags.
 *
 * @throws std::runtime_error If source cannot be located within mesh domain
 * @throws std::invalid_argument If coordinates are invalid or mesh is malformed
 *
 * @note This function is an implementation detail and should be only called
 * within @ref specfem::assembly::sources construction.
 */
template <specfem::element::dimension_tag DimensionTag>
void locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag>>>
        &sources);

} // namespace specfem::assembly::sources_impl
