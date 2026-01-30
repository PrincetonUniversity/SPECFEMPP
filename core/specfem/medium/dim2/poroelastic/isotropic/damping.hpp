#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_damping_computation_dim2_poroelastic_isotropic
 *
 */

/**
 * @ingroup specfem_damping_computation_dim2_poroelastic_isotropic
 * @brief Compute viscous damping force for 2D poroelastic isotropic media.
 *
 * Implements Darcy viscous damping for fluid-solid coupling in porous media.
 * Damping arises from viscous flow resistance through pore networks,
 * proportional to relative fluid-solid velocity and permeability.
 *
 * **Viscous force equations:**
 * \f$ \mathbf{F}_{\mathrm{visc}} = \eta_{\mathrm{f}} \mathbf{K}^{-1} \mathbf{w}
 * \f$
 *
 * where:
 * - \f$ \eta_{\mathrm{f}} \f$: fluid viscosity
 * - \f$ \mathbf{K}^{-1} \f$: inverse permeability tensor
 * - \f$ \mathbf{w} \f$: relative fluid velocity
 *
 * **Acceleration updates:**
 * \f$ \ddot{\mathbf{u}} += \frac{\phi}{\alpha} \mathbf{F}_{\mathrm{visc}} \f$
 * (solid coupling)
 * \f$ \ddot{\mathbf{w}} -= \mathbf{F}_{\mathrm{visc}} \f$ (fluid damping)
 *
 * @tparam T Scalar type for damping factor
 * @param factor Time step scaling factor
 * @param point_properties Poroelastic properties (\f$\eta_{\mathrm{f}},
 * \mathbf{K}^{-1}, \phi, \alpha\f$)
 * @param velocity Velocity field [u_x, u_z, w_x, w_z]
 * @param acceleration[in,out] Acceleration field (modified by damping)
 */
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_compute_damping_force(
    const std::true_type,
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  // viscous damping
  const auto viscx =
      factor * ((point_properties.eta_f() * point_properties.inverse_permxx() *
                 velocity(2)) +
                (point_properties.eta_f() * point_properties.inverse_permxz() *
                 velocity(3)));

  const auto viscz =
      factor * ((point_properties.eta_f() * point_properties.inverse_permxz() *
                 velocity(2)) +
                (point_properties.eta_f() * point_properties.inverse_permzz() *
                 velocity(3)));

  acceleration(0) +=
      point_properties.phi() / point_properties.tortuosity() * viscx;
  acceleration(1) +=
      point_properties.phi() / point_properties.tortuosity() * viscz;

  acceleration(2) -= viscx;
  acceleration(3) -= viscz;
}
} // namespace medium
} // namespace specfem
