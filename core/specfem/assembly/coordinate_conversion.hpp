#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/mesh/mesh_base.hpp"
#include "specfem/point/global_coordinates.hpp"

namespace specfem {
namespace assembly {

// Forward declaration of the assembled mesh (reference parameter only; the full
// definition lives in specfem/assembly/mesh.hpp and is pulled in by the
// conversion implementation files, not by call sites).
template <specfem::element::dimension_tag Dimension> struct mesh;

/**
 * @brief Resolve a generic coordinate into a mesh-space Cartesian point.
 *
 * The placement entry point used by the source/receiver constructors. The input
 * coordinate (Cartesian, geographic, or -- on a globe -- geocentric) is
 * projected into mesh space and, when depth-based, placed against the
 * topographic surface. The projection is chosen by the raw-mesh type: a plain
 * Cartesian mesh passes through, a regional Cartesian3D mesh uses the UTM
 * projection, a Globe3D mesh composes the geographic -> geocentric -> Cartesian
 * transforms on the sphere.
 *
 * @code
 * auto gcoord = specfem::assembly::to_cartesian(read_coords, mesh, raw_mesh);
 * @endcode
 *
 * @param input Input coordinate to place
 * @param mesh Assembled mesh geometry (used to place a depth against the
 * surface)
 * @param raw_mesh Raw mesh carrying the projection config and free-surface
 * faces
 * @return Mesh-space point (@ref specfem::point::global_coordinates)
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::mesh::cartesian2d_mesh &raw_mesh);

/// @copydoc to_cartesian
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh);

/// @copydoc to_cartesian
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh);

/**
 * @brief Convert a mesh-space Cartesian coordinate to a geographic coordinate.
 *
 * The inverse of the placement projection: a regional Cartesian3D mesh inverts
 * the UTM projection; a Globe3D mesh composes the Cartesian -> geocentric ->
 * geographic transforms. The assembled @p mesh is unused (the input is already
 * a mesh-space point) and kept only for a uniform signature.
 *
 * @param input Cartesian input coordinate
 * @param mesh Assembled mesh geometry (unused)
 * @param raw_mesh Raw mesh carrying the projection config
 * @return Geographic coordinate
 */
specfem::coordinate_systems::geographic_coordinates to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::cartesian3d_mesh &raw_mesh);

/// @copydoc to_geographic
specfem::coordinate_systems::geographic_coordinates to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh);

/**
 * @brief Convert a mesh-space Cartesian coordinate to a geocentric coordinate
 * (Globe3D only).
 *
 * @param input Cartesian input coordinate
 * @param mesh Assembled mesh geometry (unused)
 * @param raw_mesh Raw globe mesh
 * @return Geocentric coordinate
 */
specfem::coordinate_systems::geocentric_coordinates to_geocentric(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::globe3d_mesh &raw_mesh);

} // namespace assembly
} // namespace specfem
