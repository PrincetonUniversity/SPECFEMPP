#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Components of a vector in the local East/North/Up frame.
 *
 * This is the frame FORCESOLUTION directions are read in and E/N/Z seismogram
 * channels are written in. It is the @ref GeocentricFrame (theta, phi, r)
 * basis with its axes relabelled: East = @f$ +\mathbf{e}_\phi @f$,
 * North = @f$ -\mathbf{e}_\theta @f$ (toward decreasing colatitude), Up =
 * @f$ +\mathbf{e}_r @f$. The two conversions below are the single owner of
 * that relabel, so no caller spells the North sign flip itself.
 *
 * @tparam ValueType Scalar component type.
 */
template <typename ValueType> struct EastNorthUp {
  ValueType east;  ///< component along +phi
  ValueType north; ///< component along -theta (toward the North Pole)
  ValueType up;    ///< component along +r (radially outward)
};

/**
 * @brief Reorder East/North/Up components into GeocentricFrame order.
 *
 * @tparam ValueType Scalar component type (deduced).
 * @param v Components in the local East/North/Up frame.
 * @return The same vector's components ordered (theta, phi, r), i.e.
 * @f$ (-N, E, U) @f$.
 */
template <typename ValueType>
KOKKOS_INLINE_FUNCTION Kokkos::Array<ValueType, 3>
to_geocentric_components(const EastNorthUp<ValueType> &v) {
  return { -v.north, v.east, v.up };
}

/**
 * @brief Reorder GeocentricFrame components into East/North/Up. Inverse of
 * @ref to_geocentric_components.
 *
 * @tparam ValueType Scalar component type (deduced).
 * @param theta_phi_r Components ordered (theta, phi, r).
 * @return The same vector's components in the local East/North/Up frame.
 */
template <typename ValueType>
KOKKOS_INLINE_FUNCTION EastNorthUp<ValueType>
to_east_north_up_components(const Kokkos::Array<ValueType, 3> &theta_phi_r) {
  return { theta_phi_r[1], -theta_phi_r[0], theta_phi_r[2] };
}

} // namespace coordinate_systems
} // namespace specfem
