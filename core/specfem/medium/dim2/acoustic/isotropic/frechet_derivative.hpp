#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>
#include <utility>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_frechet_derivative_dim2_acoustic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim2_acoustic
 * @brief Compute Fréchet derivatives for acoustic isotropic media (2D and 3D).
 *
 * Calculates sensitivity kernels for density and bulk modulus in
 * acoustic wave propagation. Returns kernels \f$ K_{rho}\f$ and
 * \f$ K_{kappa}\f$ used in seismic inversion. Applies to both 2D and 3D
 * via a compile-time loop over the number of spatial dimensions.
 *
 * The kernels are computed using the following equations
 * (Luo et al. 2013, eqs. A-27, A-28):
 *
 * Density kernel:
 * \f[
 *  \Delta K_{rho} = \sum_{i=1}^{n_{dim}}
 *    \frac{\partial u^{\dagger}}{\partial x_i}
 *    \frac{\partial u^{b}}{\partial x_i}
 *    \frac{1}{\rho} \Delta t
 * \f]
 *
 * Bulk modulus kernel:
 * \f[
 *  \Delta K_{kappa} = \ddot{u}^{\dagger} \cdot u^{b} \frac{1}{\kappa} \Delta t
 * \f]
 *
 * where \f$u^{\dagger}\f$ is the adjoint acoustic potential, \f$u^{b}\f$ is
 * the backward acoustic potential, \f$\ddot{u}^{\dagger}\f$ is the adjoint
 * potential second time derivative, and \f$\Delta t\f$ is the time step.
 *
 * @tparam Tags Compile-time tag bundle (dimension, medium, property, SIMD)
 *
 * @param properties Acoustic material properties (density, bulk modulus)
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing density and bulk modulus sensitivities
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::medium_tag == specfem::element::medium_tag::acoustic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_FUNCTION specfem::point::kernels<Tags> compute_frechet_derivatives(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::velocity<Tags> &adjoint_velocity,
    const specfem::point::acceleration<Tags> &adjoint_acceleration,
    const specfem::point::displacement<Tags> &backward_displacement,
    const specfem::point::field_derivatives<Tags> &adjoint_derivatives,
    const specfem::point::field_derivatives<Tags> &backward_derivatives,
    const type_real &dt) {

  // Number of spatial dimensions: 2 for dim2, 3 for dim3
  constexpr int ndim = specfem::element::dimension<Tags::dimension_tag>::dim;

  // Density kernel: 1/rho * (grad(phi^adj) · grad(phi)) * dt
  typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype
      rho_kl = 0;
  for (int i = 0; i < ndim; ++i)
    rho_kl += adjoint_derivatives.du(0, i) * backward_derivatives.du(0, i);
  rho_kl *= properties.rho_inverse() * dt;

  // Bulk modulus kernel: ddot(phi^adj) * phi / kappa * dt
  const auto kappa_kl =
      (adjoint_acceleration.get_data() * backward_displacement.get_data()) *
      static_cast<type_real>(1.0) / properties.kappa() * dt;

  return { rho_kl, kappa_kl };
}

} // namespace medium_physics
} // namespace specfem
