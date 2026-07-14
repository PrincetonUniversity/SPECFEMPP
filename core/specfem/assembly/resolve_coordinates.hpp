#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/coordinate_systems/coordinate_resolution_result.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/utm.hpp"
#include "specfem/mesh.hpp"
#include "specfem/point.hpp"

#include <optional>

namespace specfem {
namespace assembly {

/**
 * @brief Convert a generic coordinate to mesh-space global coordinates.
 *
 * Uses dynamic dispatch to determine the concrete coordinate type and
 * perform the appropriate conversion:
 *
 * - **cartesian** with origin set: `global = stored + origin`
 * - **cartesian** with origin nullopt: queries the topographic surface above
 *   (x, y) to set the origin elevation, then resolves as above. The found
 *   elevation is reported in
 *   @ref specfem::coordinate_systems::CoordinateResolutionResult::topography.
 * - **geographic**: projects via UTM (requires @p utm_config), then resolves
 *   depth as cartesian-with-depth.
 * - **geocentric**: not yet implemented (Globe3D future).
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param coords The generic coordinate to resolve (non-const: may set origin)
 * @param mesh The assembled mesh (provides geometry context)
 * @param surface Free-surface faces for topographic depth resolution (dim3
 *        only)
 * @param utm_config Optional UTM projection config (required for geographic)
 * @return Resolved global coordinates plus the found topography (if any)
 *
 * @throws std::runtime_error if the coordinate type is unknown,
 *         geographic coords are used without utm_config, or
 *         geocentric coords are used (not yet implemented)
 */
template <specfem::element::dimension_tag DimensionTag>
specfem::coordinate_systems::CoordinateResolutionResult<DimensionTag>
resolve_coordinates(
    specfem::coordinate_systems::coordinates<DimensionTag> &coords,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::mesh::acoustic_free_surface<DimensionTag> &surface,
    const std::optional<specfem::coordinate_systems::utm_projection_config>
        &utm_config = std::nullopt);

} // namespace assembly
} // namespace specfem
