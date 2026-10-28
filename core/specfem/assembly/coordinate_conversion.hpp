#pragma once

#include "specfem/coordinate_systems/basis_rotation.hpp"
#include "specfem/coordinate_systems/cartesian.hpp"
#include "specfem/coordinate_systems/coordinates.hpp"
#include "specfem/coordinate_systems/geocentric.hpp"
#include "specfem/coordinate_systems/geographic.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/point/global_coordinates.hpp"

#include <array>

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
 * @brief Rotation mapping local geocentric @f$ (\theta, \phi, r) @f$ components
 * to mesh Cartesian @f$ (x, y, z) @f$ components.
 *
 * Columns are the local unit vectors @f$ \hat{e}_\theta, \hat{e}_\phi,
 * \hat{e}_r @f$ in mesh components (see
 * @ref specfem::coordinate_systems::rotation_matrix).
 */
using LocalFrameRotation = specfem::coordinate_systems::BasisMatrix<
    specfem::coordinate_systems::CartesianFrame,
    specfem::coordinate_systems::GeocentricFrame, double>;

/**
 * @brief Local-to-Cartesian rotation on a flat regional mesh.
 *
 * On a UTM or plain Cartesian mesh the local axes coincide with the mesh axes:
 * x = East = @f$ \hat{e}_\phi @f$, y = North = @f$ -\hat{e}_\theta @f$,
 * z = Up = @f$ \hat{e}_r @f$. That is the globe rotation evaluated at
 * @f$ \theta = 0, \phi = -\pi/2 @f$ (it reproduces the regional CMTSOLUTION
 * relabel Mxx = Mpp, Myy = Mtt, Mzz = Mrr, Mxy = -Mtp, Mxz = Mrp, Myz = -Mrt
 * and passes an East/North/Up force through unchanged). The matrix is built
 * from exact 0/±1 entries rather than from trigonometric functions so that
 * regional results are unaffected by rounding.
 *
 * @return Rotation with columns (0,-1,0), (1,0,0), (0,0,1)
 */
LocalFrameRotation flat_mesh_rotation();

/**
 * @brief Rotation from the local geocentric frame at a mesh-space point into
 * the mesh Cartesian frame.
 *
 * The single owner of "which local frame does this point have": sources rotate
 * their moment tensor / force with it and receivers derive their
 * East/North/Up rows from it (@ref local_east_north_up_basis). On a globe mesh
 * the frame rotates with the point's colatitude/longitude, recovered from its
 * planet-centred Cartesian position through @ref to_geocentric (which owns the
 * ellipticity/topography guard). On a regional UTM or plain Cartesian mesh it
 * is @ref flat_mesh_rotation; UTM meridian convergence is deliberately ignored.
 *
 * @param point Mesh-space Cartesian point
 * @param mesh Assembled mesh (carries the projection)
 * @return Rotation mapping (theta, phi, r) components to mesh (x, y, z)
 */
LocalFrameRotation local_frame_rotation(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &point,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

/**
 * @brief Local East/North/Up basis at a mesh-space point, as the rows of the
 * receiver rotation matrix.
 *
 * The receiver-side view of @ref local_frame_rotation: the rows are the East,
 * North and Up unit vectors in mesh Cartesian components, i.e. the matrix that
 * maps a mesh-frame vector to (East, North, Up) components and is handed to
 * `SeismogramIterator::set_rotation_matrix`. The relabel
 * @f$ (\theta, \phi, r) \to (E, N, U) @f$ is taken from
 * @ref specfem::coordinate_systems::to_east_north_up_components, so no sign
 * convention is spelled here. The identity on a flat mesh.
 *
 * @param point Mesh-space Cartesian point
 * @param mesh Assembled mesh (carries the projection)
 * @return Rows East, North, Up in mesh Cartesian components
 */
std::array<std::array<double, 3>, 3> local_east_north_up_basis(
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &point,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh);

} // namespace assembly
} // namespace specfem
