#pragma once

#include "specfem/coordinate_systems/spherical_basis.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace sources {

/**
 * @brief Moment tensor in the geographic spherical @f$ (r, \theta, \phi) @f$
 * basis.
 *
 * This is the as-read CMTSOLUTION / Harvard-CMT frame: @f$ \theta @f$ is
 * geocentric colatitude (0 at the North Pole) and @f$ \phi @f$ is longitude.
 * On a globe it is rotated into the Cartesian mesh frame at the source
 * location (assuming a perfect sphere, i.e. geographic latitude taken as the
 * geocentric colatitude; ellipticity is deferred). The tensor is symmetric, so
 * the six stored components fully describe it:
 *
 * \f[
 * \mathbf{M}_{(r\theta\phi)} = \begin{pmatrix}
 *   M_{rr} & M_{rt} & M_{rp} \\
 *   M_{rt} & M_{tt} & M_{tp} \\
 *   M_{rp} & M_{tp} & M_{pp}
 * \end{pmatrix}
 * \f]
 */
struct GeographicMomentTensor {
  type_real Mrr; ///< radial-radial
  type_real Mtt; ///< colatitude-colatitude
  type_real Mpp; ///< longitude-longitude
  type_real Mrt; ///< radial-colatitude
  type_real Mrp; ///< radial-longitude
  type_real Mtp; ///< colatitude-longitude
};

/**
 * @brief Moment tensor in the global Cartesian @f$ (x, y, z) @f$ basis.
 *
 * \f[
 * \mathbf{M}_{(xyz)} = \begin{pmatrix}
 *   M_{xx} & M_{xy} & M_{xz} \\
 *   M_{xy} & M_{yy} & M_{yz} \\
 *   M_{xz} & M_{yz} & M_{zz}
 * \end{pmatrix}
 * \f]
 */
struct CartesianMomentTensor {
  type_real Mxx; ///< x-x
  type_real Myy; ///< y-y
  type_real Mzz; ///< z-z
  type_real Mxy; ///< x-y
  type_real Mxz; ///< x-z
  type_real Myz; ///< y-z
};

/**
 * @brief Rotate a moment tensor from the spherical @f$ (r,\theta,\phi) @f$
 * basis to the global Cartesian basis.
 *
 * Applies the rank-2 tensor rotation @f$ \mathbf{M}_{xyz} = R\,
 * \mathbf{M}_{r\theta\phi}\, R^{\mathsf{T}} @f$, where the columns of @f$ R @f$
 * are the local basis vectors @f$ [\mathbf{e}_r\,|\,\mathbf{e}_\theta\,|\,
 * \mathbf{e}_\phi] @f$ from
 * @ref specfem::coordinate_systems::local_spherical_basis. This reproduces the
 * closed-form rotation in SPECFEM3D_GLOBE's `locate_sources.f90`.
 *
 * @param theta Geocentric colatitude in radians (0 at the North Pole).
 * @param phi Longitude in radians.
 * @param m Moment tensor in the spherical basis.
 * @return The same tensor expressed in the global Cartesian basis.
 */
CartesianMomentTensor
rotate_moment_tensor_geographic_to_cartesian(double theta, double phi,
                                             const GeographicMomentTensor &m);

/**
 * @brief Relabel a spherical moment tensor into a local East-North-Up frame.
 *
 * On a flat regional (UTM or Cartesian) mesh the source's local tangent frame
 * coincides with the mesh axes, so the rotation degenerates to the standard
 * relabel @f$ x = \text{East}(\phi),\ y = \text{North}(-\theta),\ z =
 * \text{Up}(r) @f$:
 * \f[
 *   M_{xx} = M_{pp},\ M_{yy} = M_{tt},\ M_{zz} = M_{rr},\
 *   M_{xy} = -M_{tp},\ M_{xz} = M_{rp},\ M_{yz} = -M_{rt}.
 * \f]
 * This is the convention used by regional CMTSOLUTION sources; it is the flat
 * limit of @ref rotate_moment_tensor_geographic_to_cartesian, kept as a
 * separate entry point so the sign convention lives in one place.
 *
 * @param m Moment tensor in the spherical basis.
 * @return The tensor in the local East-North-Up Cartesian frame.
 */
CartesianMomentTensor geographic_to_enu(const GeographicMomentTensor &m);

} // namespace sources
} // namespace specfem
