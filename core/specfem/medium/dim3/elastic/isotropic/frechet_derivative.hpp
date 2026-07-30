#pragma once

#include "specfem/element.hpp"
#include "specfem/medium/dim3/elastic/isotropic/strain.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_frechet_derivative_dim3_elastic_isotropic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim3_elastic_isotropic
 * @brief Compute Fréchet derivatives for 3D elastic isotropic media.
 *
 * Calculates density, shear modulus, bulk modulus, and equivalent
 * \f$(\rho, \alpha, \beta)\f$ sensitivity kernels. The implementation follows
 * the isotropic elastic kernel expressions used by SPECFEM3D:
 *
 * \f[
 *  \Delta K_{\rho} = -\rho \Delta t\,
 *    \ddot{u}^{\dagger} \cdot u^{b}
 * \f]
 *
 * \f[
 *  \Delta K_{\mu} = -2\mu \Delta t\,
 *    \varepsilon_{dev}^{\dagger} : \varepsilon_{dev}^{b}
 * \f]
 *
 * \f[
 *  \Delta K_{\kappa} = -\kappa \Delta t\,
 *    \nabla \cdot u^{\dagger}\, \nabla \cdot u^{b}
 * \f]
 *
 * where \f$u^{\dagger}\f$ is the adjoint displacement field and \f$u^b\f$ is
 * the backward/reconstructed displacement field.
 *
 * @tparam Tags Compile-time tag bundle (dimension, medium, property, SIMD)
 *
 * @param properties Elastic material properties (\f$\rho\f$, \f$\mu\f$,
 * \f$\kappa\f$)
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing elastic parameter sensitivities
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
           Tags::medium_tag == specfem::element::medium_tag::elastic &&
           Tags::property_tag == specfem::element::property_tag::isotropic)
KOKKOS_FUNCTION specfem::point::kernels<Tags> compute_frechet_derivatives(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::velocity<Tags> &adjoint_velocity,
    const specfem::point::acceleration<Tags> &adjoint_acceleration,
    const specfem::point::displacement<Tags> &backward_displacement,
    const specfem::point::field_derivatives<Tags> &adjoint_derivatives,
    const specfem::point::field_derivatives<Tags> &backward_derivatives,
    const type_real &dt) {

  (void)adjoint_velocity;

  using datatype =
      typename specfem::datatype::simd<type_real, Tags::using_simd>::datatype;

  const auto adjoint_deviatoric_strain =
      impl_compute_deviatoric_strain(adjoint_derivatives);
  const auto backward_deviatoric_strain =
      impl_compute_deviatoric_strain(backward_derivatives);

  datatype mu_kl = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      mu_kl +=
          adjoint_deviatoric_strain(i, j) * backward_deviatoric_strain(i, j);
    }
  }

  const auto adjoint_trace = adjoint_derivatives.du(0, 0) +
                             adjoint_derivatives.du(1, 1) +
                             adjoint_derivatives.du(2, 2);
  const auto backward_trace = backward_derivatives.du(0, 0) +
                              backward_derivatives.du(1, 1) +
                              backward_derivatives.du(2, 2);

  auto kappa_kl = adjoint_trace * backward_trace;
  auto rho_kl =
      adjoint_acceleration.get_data() * backward_displacement.get_data();

  rho_kl = static_cast<type_real>(-1.0) * properties.rho() * dt * rho_kl;
  mu_kl = static_cast<type_real>(-2.0) * properties.mu() * dt * mu_kl;
  kappa_kl = static_cast<type_real>(-1.0) * properties.kappa() * dt * kappa_kl;

  const auto rhop_kl = rho_kl + kappa_kl + mu_kl;

  const auto beta_kl =
      static_cast<type_real>(2.0) *
      (mu_kl - static_cast<type_real>(4.0 / 3.0) * properties.mu() /
                   properties.kappa() * kappa_kl);

  const auto alpha_kl =
      static_cast<type_real>(2.0) *
      (static_cast<type_real>(1.0) + static_cast<type_real>(4.0 / 3.0) *
                                         properties.mu() / properties.kappa()) *
      kappa_kl;

  return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
}

} // namespace medium_physics
} // namespace specfem
