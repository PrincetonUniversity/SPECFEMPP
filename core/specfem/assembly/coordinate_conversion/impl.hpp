#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/point/global_coordinates.hpp"

namespace specfem {
namespace assembly {

// Forward declaration of the assembled mesh (used only as a reference
// parameter below; the full definition is pulled in by the implementation
// files that place a coordinate against the surface).
template <specfem::element::dimension_tag Dimension> struct mesh;

namespace coordinate_conversion_impl {

/**
 * @brief Recover a concrete coordinate type from the polymorphic base.
 *
 * The single runtime step shared by every conversion: the stored coordinate's
 * concrete type is only known at runtime.
 *
 * @tparam Concrete Concrete coordinate type to attempt to recover
 * @tparam DimensionTag Spatial dimension of the coordinate
 * @param coordinate Polymorphic coordinate to inspect
 * @return Pointer to the concrete coordinate, or @c nullptr if the dynamic type
 * differs
 */
template <typename Concrete, specfem::element::dimension_tag DimensionTag>
const Concrete *
as(const specfem::coordinate_systems::coordinates<DimensionTag> &coordinate) {
  return dynamic_cast<const Concrete *>(&coordinate);
}

/**
 * @brief Place a dim3 Cartesian coordinate in mesh space.
 *
 * Absolute coordinates (origin set) pass through; depth-based coordinates
 * (origin unset) are placed against the topographic surface above (x, y), or
 * the flat fallback (z = 0) when the mesh has no free surface. Used for raw
 * Cartesian inputs, whose depth is unknown until resolved.
 *
 * This function serves as a type bridge from
 * coordinate_systems::cartesian_coordinates to point::global_coordinates, and
 * is used by the public to_cartesian() entry points.
 *
 * @param cartesian Cartesian coordinate to place
 * @param mesh Assembled mesh geometry (used to find the surface above x, y)
 * @param surface Free-surface faces of the raw mesh
 * @return Mesh-space point
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
resolve_cartesian(
    const specfem::coordinate_systems::cartesian_coordinates<
        specfem::element::dimension_tag::dim3> &cartesian,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface);

} // namespace coordinate_conversion_impl
} // namespace assembly
} // namespace specfem
