#pragma once

#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/coordinate_systems/spherical_basis.hpp"
#include "specfem/element/tags.hpp"
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
 * topographic surface. The projection is read from the assembled mesh: a plain
 * Cartesian mesh passes through, a regional mesh uses the UTM projection, a
 * globe mesh composes the geographic -> geocentric -> Cartesian transforms on
 * the sphere.
 *
 * @code
 * auto gcoord = specfem::assembly::to_cartesian(read_coords, mesh);
 * @endcode
 *
 * @param input Input coordinate to place
 * @param mesh Assembled mesh (carries the projection and free-surface geometry
 * used to place a depth against the surface)
 * @return Mesh-space point (@ref specfem::point::global_coordinates)
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim2> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh);

/// @copydoc to_cartesian
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
to_cartesian(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Convert a mesh-space Cartesian coordinate to a geographic coordinate.
 *
 * The inverse of the placement projection: a regional mesh inverts the UTM
 * projection; a globe mesh composes the Cartesian -> geocentric -> geographic
 * transforms. The assembled @p mesh supplies the projection.
 *
 * @param input Cartesian input coordinate
 * @param mesh Assembled mesh (carries the projection)
 * @return Geographic coordinate
 */
specfem::coordinate_systems::geographic_coordinates to_geographic(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Convert a mesh-space Cartesian coordinate to a geocentric coordinate
 * (globe only).
 *
 * @param input Cartesian input coordinate
 * @param mesh Assembled mesh (carries the projection)
 * @return Geocentric coordinate
 */
specfem::coordinate_systems::geocentric_coordinates to_geocentric(
    const specfem::coordinate_systems::coordinates<
        specfem::element::dimension_tag::dim3> &input,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Local East/North/Up basis at a mesh-space point, in mesh Cartesian
 * components.
 *
 * The receiver-side counterpart of the source frame resolution: on a globe mesh
 * the basis is the local geographic frame at the point's colatitude/longitude
 * (recovered from its planet-centred Cartesian position through
 * @ref to_geocentric, so the ellipticity/topography guard is owned there); on a
 * regional UTM or plain Cartesian mesh the local axes coincide with the mesh
 * axes (x = East, y = North, z = Up), so the basis is the identity. UTM
 * meridian convergence is deliberately ignored, as for sources.
 *
 * @param point Mesh-space Cartesian point
 * @param mesh Assembled mesh (carries the projection)
 * @return The East, North and Up unit vectors in mesh Cartesian components
 */
specfem::coordinate_systems::GeographicBasis local_geographic_basis(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &point,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

} // namespace assembly
} // namespace specfem
