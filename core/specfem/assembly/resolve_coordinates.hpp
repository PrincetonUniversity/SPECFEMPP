#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/input_coordinates.hpp"
#include "specfem/point.hpp"

namespace specfem {
namespace assembly {

/**
 * @brief Convert a generic coordinate to mesh-space global coordinates.
 *
 * Uses dynamic dispatch to determine the concrete coordinate type and
 * perform the appropriate conversion. Trivial types (e.g., cartesian_3d)
 * copy values directly; depth-based types will use mesh topography when
 * available.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param coords The generic coordinate to resolve
 * @param mesh The assembled mesh (provides topography, geometry context)
 * @return Resolved global coordinates in mesh space
 *
 * @throws std::runtime_error if the coordinate type is unknown or
 *         if a required conversion is not yet implemented
 */
template <specfem::element::dimension_tag DimensionTag>
specfem::point::global_coordinates<DimensionTag> resolve_coordinates(
    const specfem::coordinate_systems::input_coordinates<DimensionTag> &coords,
    const specfem::assembly::mesh<DimensionTag> &mesh);

} // namespace assembly
} // namespace specfem
