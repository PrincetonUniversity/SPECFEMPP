#pragma once

#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/damping.hpp"
#include "specfem/point.hpp"
#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @brief Compute damping force for wave attenuation.
 *
 * Generic damping force computation interface that adds viscous damping
 * to wave propagation equations. Provides compile-time dispatch to
 * medium-specific implementations based on element attributes.
 *
 * @note Only medium types with damping force support will modify the
 * acceleration. Medium types without damping force support will result a no-op.
 *
 * @tparam T Scalar type for damping factor
 * @tparam PointPropertiesType Point-wise material properties
 * @tparam PointVelocityType Point-wise velocity field
 * @tparam PointAccelerationType Point-wise acceleration field
 * @param factor Integration factor (e.g., product of quadrature weight(s) and
 * Jacobian determinant) \f$ J \, w_q \f$
 * @param point_properties Material properties at point
 * @param velocity Velocity field at point
 * @param acceleration[in,out] Acceleration field (modified by damping)
 */
template <typename T, typename Tags>
KOKKOS_INLINE_FUNCTION void
compute_damping_force(const T factor,
                      const specfem::point::properties<Tags> &point_properties,
                      const specfem::point::velocity<Tags> &velocity,
                      specfem::point::acceleration<Tags> &acceleration) {

  if constexpr (specfem::element::attributes<
                    Tags::dimension_tag, Tags::medium_tag>::has_damping_force) {
    impl_compute_damping_force<T, Tags>(factor, point_properties, velocity,
                                        acceleration);
  }
}

} // namespace medium_physics
} // namespace specfem
