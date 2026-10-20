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
 * This is the single owner of the spherical-basis trigonometry. The moment
 * tensor rotation (rank-2, @f$ R M R^{\mathsf{T}} @f$) builds on these vectors
 * directly; rank-1 consumers use @ref GeographicBasis.
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

/**
 * @brief Local geographic (East, North, Up) orthonormal basis at a point on
 * the sphere.
 *
 * Built from @ref SphericalBasis with the convention East = +e_phi,
 * North = -e_theta (toward decreasing colatitude), Up = +e_r (radially
 * outward). This is the single owner of that sign convention: force sources
 * rotate from this frame into the mesh and receivers rotate seismograms from
 * the mesh into it. The frame is right-handed (east x north = up, determinant
 * +1), so axial vectors (rotation, curl) rotate like polar vectors. Matches
 * SPECFEM3D_GLOBE's receiver orientation vectors (nu_rec).
 *
 * At the poles the longitude is undefined; the Cartesian->geocentric inverse
 * nudges both angles by ~1e-7 rad, so at the North Pole East ~ +y, North ~ -x,
 * Up ~ +z, and at the South Pole East ~ +y, North ~ +x, Up ~ -z (no NaN, still
 * right-handed).
 */
struct GeographicBasis {
  std::array<double, 3> east;  ///< eastward unit vector (+e_phi)
  std::array<double, 3> north; ///< northward unit vector (-e_theta)
  std::array<double, 3> up;    ///< upward unit vector (+e_r, outward)
};

/**
 * @brief Construct the local East-North-Up basis at colatitude @p theta and
 * longitude @p phi.
 *
 * @param theta Geocentric colatitude in radians (0 at the North Pole).
 * @param phi Longitude in radians.
 * @return The three orthonormal basis vectors in global Cartesian components.
 */
GeographicBasis local_geographic_basis(double theta, double phi);

} // namespace coordinate_systems
} // namespace specfem
