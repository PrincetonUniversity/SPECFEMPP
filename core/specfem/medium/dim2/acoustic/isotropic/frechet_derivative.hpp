#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_frechet_derivative_dim2_acoustic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim2_acoustic
 * @brief Compute Fréchet derivatives for 2D acoustic isotropic media.
 *
 * Calculates sensitivity kernels for density and bulk modulus in
 * 2D acoustic wave propagation. Returns kernels \f$ K_{rho}\f$ and
 * \f$ K_{kappa}\f$ used in seismic inversion.
 *
 * The kernels are computed using the following equations:
 *
 * Density kernel:
 * \f[
 *  \Delta K_{rho} = \left( \frac{\partial u^{\dagger}}{\partial x}
 * \frac{\partial u^{b}}{\partial x} +
 *                    \frac{\partial u^{\dagger}}{\partial z} \frac{\partial
 * u^{b}}{\partial z} \right)
 *             \frac{1}{\rho} \Delta t
 * \f]
 *
 * Bulk modulus kernel:
 * \f[
 *  \Delta K_{kappa} = \ddot{u}^{\dagger} \cdot u^{b} \frac{1}{\kappa} \Delta t
 * \f]
 *
 * where \f$u^{\dagger}\f$ is the adjoint field, \f$u^{b}\f$ is the backward
 * field,
 * \f$\ddot{u}^{\dagger}\f$ is the adjoint acceleration, and \f$\Delta t\f$ is
 * the time step.
 *
 * @tparam PointPropertiesType Acoustic material properties
 * @tparam AdjointPointVelocityType Adjoint velocity field
 * @tparam AdjointPointAccelerationType Adjoint acceleration field
 * @tparam BackwardPointDisplacementType Backward displacement field
 * @tparam PointFieldDerivativesType Spatial field derivatives
 *
 * @param properties Acoustic material properties (density, bulk modulus)
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing density and bulk modulus sensitivities
 *
 * @note This is the specialized implementation for 2D acoustic isotropic media
 */
template <typename PointPropertiesType, typename AdjointPointVelocityType,
          typename AdjointPointAccelerationType,
          typename BackwardPointDisplacementType,
          typename PointFieldDerivativesType>
KOKKOS_FUNCTION specfem::point::kernels<
    PointPropertiesType::dimension_tag, PointPropertiesType::medium_tag,
    PointPropertiesType::property_tag, PointPropertiesType::simd::using_simd>
impl_compute_frechet_derivatives(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  const auto rho_kl =
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1)) *
      properties.rho_inverse() * dt;

  const auto kappa_kl =
      (adjoint_acceleration.get_data() * backward_displacement.get_data()) *
      static_cast<type_real>(1.0) / properties.kappa() * dt;

  return { rho_kl, kappa_kl };
}

} // namespace medium
} // namespace specfem
