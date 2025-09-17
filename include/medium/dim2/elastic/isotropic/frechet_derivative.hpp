#pragma once

#include "algorithms/gradient.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"

#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

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
                                 specfem::element::medium_tag::elastic_psv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  const auto kappa = properties.lambdaplus2mu() - properties.mu();

  /*
  In the following the explanation for the SH wave kernels is given.

  Computing the deviatoric strain tensor for SH waves:

                         strain epsilon
    D = [    dux_dx         1/2(dux_dy+duy_dx) 1/2(dux_dz+duz_dx),
        1/2(duy_dx+dux_dy)     duy_dy         1/2(duy_dz+duz_dy),
        1/2(duz_dx+dux_dz) 1/2(duz_dy+duy_dz)     duz_dz   ]

                              trace of strain diagonalized
         [ 1/3 (dux_dx + duy_dy + duz_dz)       0                  0,
      -    0             1/3 (dux_dx + duy_dy + duz_dz)            0,
           0                    0     1/3 (dux_dx + duy_dy + duz_dx)  ]

  We use `s` as the "standard" wavefield and `s#` as the adjoint wavefield.
  We use `ad_` as prefix for the adjoint strainfield, and `b_` as prefix for
  the "standard" strainfield.
  */

  // Compute the gradient of the adjoint field
  // ad_dsxx = 0.5 * (ds#x/dx + ds#x/dx) = ds#x/dx
  const auto ad_dsxx = adjoint_derivatives.du(0, 0);

  // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
  const auto ad_dsxz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));

  // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz) = ds#z/dz
  const auto ad_dszz = adjoint_derivatives.du(1, 1);

  // Compute the gradient of the backward field
  // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
  const auto b_dsxx = backward_derivatives.du(0, 0);

  // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
  const auto b_dsxz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));

  // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
  const auto b_dszz = backward_derivatives.du(1, 1);

  // what's this?
  // --------------------------------------
  // const type_real kappa_kl =
  //     -1.0 * kappa * dt * ((ad_dsxx + ad_dszz) * (b_dsxx + b_dszz));
  // const type_real mu_kl = -2.0 * properties.mu * dt *
  //                         (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
  //                          2.0 * ad_dsxz * b_dsxz - 1.0 / 3.0 * kappa_kl);
  // const type_real rho_kl =
  //     -1.0 * properties.rho * dt *
  //     (adjoint_field.acceleration * backward_field.displacement);
  // --------------------------------------

  // In the papers we use dagger for the notation of the adjoint wavefield
  // here I'm using #

  // Part of Tromp et al. 2005, Eq 18
  // div(s#) * div(s)
  auto kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);

  // Part of Tromp et al. 2005, Eq 17
  // [eps+ : eps] - 1/3 [div (s#) * div(s)]
  // I am not clear on how we get to the following form but from the
  // GPU cuda code from the fortran code I assume that there is an
  // assumption being made that eps#_i * eps_j = eps#_j * eps_i in the
  // isotropic case due to the symmetry of the voigt notation stiffness
  // matrix. Since x
  auto mu_kl = (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                static_cast<type_real>(2.0) * ad_dsxz * b_dsxz -
                static_cast<type_real>(1.0 / 3.0) * kappa_kl);

  // This notation/naming is confusing with respect to the physics.
  // Should be forward.acceleration dotted with adjoint displacement
  // See Tromp et al. 2005, Equation 14.
  auto rho_kl =
      adjoint_acceleration.get_data() * backward_displacement.get_data();

  // Finishing the kernels
  kappa_kl = static_cast<type_real>(-1.0) * kappa * dt * kappa_kl;
  mu_kl = static_cast<type_real>(-2.0) * properties.mu() * dt * mu_kl;
  rho_kl = static_cast<type_real>(-1.0) * properties.rho() * dt * rho_kl;

  // rho' kernel, first term in Equation 20
  const auto rhop_kl = rho_kl + kappa_kl + mu_kl;

  // beta (shear wave) kernel, second term in Equation 20
  const auto beta_kl = static_cast<type_real>(2.0) *
                       (mu_kl - static_cast<type_real>(4.0 / 3.0) *
                                    properties.mu() / kappa * kappa_kl);

  // alpha (compressional wave) kernel, third and last term in Eq. 20
  // of Tromp et al 2005.
  const auto alpha_kl =
      static_cast<type_real>(2.0) *
      (static_cast<type_real>(1.0) +
       static_cast<type_real>(4.0 / 3.0) * properties.mu() / kappa) *
      kappa_kl;

  return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
}

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
                                 specfem::element::medium_tag::elastic_sh>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  /*
    SH (membrane) waves
    -------------------

    The deviatroic strain tensor for SH waves in and isotropic elastic medium
    is given by:

    SH-waves: plane strain assumption ux==uz==0 and d/dy==0
      D = [   0             1/2 duy_dx       0,                   [0 0 0,
            1/2 duy_dx       0             1/2 duy_dz,       -     0 0 0,
              0             1/2 duy_dz       0          ]          0 0 0]

    Resulting in the following kernels D# : D
    D# : D = sum_i sum_j D#_ij * D_ij
              = 1/2du#y_dx * 1/2duy_dx + 1/2du#y_dx * 1/2duy_dx
                  + 1/2du#y_dz * 1/2duy_dz + 1/2du#y_dz * 1/2duy_dz
              = 1/2 ( du#y_dx * duy_dx) + 1/2 (du#y_dz * duy_dz)
              = 1/2 ( du#y_dx * duy_dx + du#y_dz * duy_dz )

    */
  const auto mu_kl =
      static_cast<type_real>(-2.0) * properties.mu() * dt *
      static_cast<type_real>(0.5) *
      // du#y_dx * duy_dx +
      (adjoint_derivatives.du(0, 0) * backward_derivatives.du(0, 0) +
       // du#y_dz * duy_dz
       adjoint_derivatives.du(0, 1) * backward_derivatives.du(0, 1));
  const auto rho_kl =
      static_cast<type_real>(-1.0) * properties.rho() * dt *
      (adjoint_acceleration.get_data() * backward_displacement.get_data());
  const auto kappa_kl = decltype(mu_kl)(0.0);

  const auto rhop_kl = rho_kl + kappa_kl + mu_kl;
  const auto alpha_kl = 0.0;
  const auto beta_kl = static_cast<type_real>(2.0) * mu_kl;

  return { rho_kl, mu_kl, kappa_kl, rhop_kl, alpha_kl, beta_kl };
}
} // namespace medium
} // namespace specfem
