#pragma once

#include "specfem/coordinate_systems/spherical_basis.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace sources {

/**
 * @brief Force vector in the local East-North-Up (geographic) frame.
 *
 * This is the as-read FORCESOLUTION frame on a globe: the direction components
 * are given along East (@f$ +\phi @f$), North (decreasing colatitude, i.e.
 * @f$ -\theta @f$) and Up (radially outward, @f$ +r @f$). It is rotated into
 * the Cartesian mesh frame at the source location.
 */
struct GeographicForce {
  type_real east;  ///< eastward component (toward +phi)
  type_real north; ///< northward component (toward -theta)
  type_real up;    ///< upward component (radially outward)
};

/**
 * @brief Force vector in the global Cartesian @f$ (x, y, z) @f$ frame.
 */
struct CartesianForce {
  type_real fx; ///< x component
  type_real fy; ///< y component
  type_real fz; ///< z component
};

/**
 * @brief Rotate a force from the local East-North-Up frame to global Cartesian.
 *
 * Applies the rank-1 rotation
 * @f$ \mathbf{f} = f_E\,\mathbf{e}_\phi + f_N\,(-\mathbf{e}_\theta)
 * + f_U\,\mathbf{e}_r @f$, with the local basis from
 * @ref specfem::coordinate_systems::local_spherical_basis at @p theta, @p phi.
 *
 * @param theta Geocentric colatitude in radians (0 at the North Pole).
 * @param phi Longitude in radians.
 * @param f Force in the local East-North-Up frame.
 * @return The same force expressed in the global Cartesian frame.
 */
CartesianForce rotate_force_geographic_to_cartesian(double theta, double phi,
                                                    const GeographicForce &f);

/**
 * @brief Relabel a local East-North-Up force onto a flat regional mesh.
 *
 * On a flat regional (UTM or Cartesian) mesh the local tangent axes coincide
 * with the mesh axes (@f$ x = \text{East},\ y = \text{North},\ z = \text{Up}
 * @f$), so a vector passes through unchanged. Kept as a named entry point so
 * the regional convention lives in one place (and documents that, unlike the
 * moment tensor, the vector relabel carries no sign flips).
 *
 * @param f Force in the local East-North-Up frame.
 * @return The force with components taken as the Cartesian mesh axes.
 */
CartesianForce geographic_to_enu(const GeographicForce &f);

} // namespace sources
} // namespace specfem
