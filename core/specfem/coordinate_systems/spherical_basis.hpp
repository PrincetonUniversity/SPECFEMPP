#pragma once

#include <array>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Local spherical orthonormal basis at a point on the sphere.
 *
 * The three unit vectors are expressed in global Cartesian components:
 * \f[
 *   \mathbf{e}_r = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta),\quad
 *   \mathbf{e}_\theta =
 * (\cos\theta\cos\phi,\ \cos\theta\sin\phi,\ -\sin\theta),
 *   \quad
 *   \mathbf{e}_\phi = (-\sin\phi,\ \cos\phi,\ 0).
 * \f]
 * This is the single owner of the spherical-basis trigonometry. Callers that
 * need a local frame build their rotation from it rather than re-deriving the
 * trigonometry: the moment-tensor rotation (rank-2, @f$ R M R^{\mathsf{T}} @f$)
 * and the seismogram component rotation (rank-1, into a local vertical/north/
 * east frame) both consume these vectors, each assembling its own matrix in its
 * own sign and component-order convention.
 */
struct SphericalBasis {
  std::array<double, 3> e_r;     ///< radial unit vector (outward)
  std::array<double, 3> e_theta; ///< colatitude unit vector (toward +theta)
  std::array<double, 3> e_phi;   ///< longitude unit vector (toward +phi)
};

/**
 * @brief Construct the local spherical basis at colatitude @p theta and
 * longitude @p phi.
 *
 * @param theta Geocentric colatitude in radians (0 at the North Pole).
 * @param phi Longitude in radians.
 * @return The three orthonormal basis vectors in global Cartesian components.
 */
SphericalBasis local_spherical_basis(double theta, double phi);

} // namespace coordinate_systems
} // namespace specfem
