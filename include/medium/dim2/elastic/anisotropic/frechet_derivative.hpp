#pragma once

#include "algorithms/gradient.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"

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
                                 specfem::element::property_tag::anisotropic>,
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  /*
  Note: Using # as adjoint modifier for the comments, so that `s` is the
  "standard" strainfield and `s#` is the adjoint strainfield. We use `ad_` as
  prefix for the adjoint wavefield and its derivatives, and `b_` as prefix
  for the "standard" backward wavefield and its derivatives.
  */

  // ad_dsxx = 0.5 * (ds#x/dx + ds#x/dx)
  const auto ad_dsxx = adjoint_derivatives.du(0, 0);

  // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
  const auto ad_dsxz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));

  // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz)
  const auto ad_dszz = adjoint_derivatives.du(1, 1);

  // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
  const auto b_dsxx = backward_derivatives.du(0, 0);

  // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
  const auto b_dsxz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));

  // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
  const auto b_dszz = backward_derivatives.du(1, 1);

  // inner part of rho kernel equation 14
  // rho_kl = s#''_i * s_j
  auto rho_kl =
      adjoint_acceleration.get_data() * backward_displacement.get_data();

  // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
  // That is \eps_{jk} \eps_{lm}
  auto c11_kl = ad_dsxx * b_dsxx;
  auto c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
  auto c15_kl = 2 * ad_dsxx * b_dsxz + ad_dsxz * b_dsxx;
  auto c33_kl = ad_dszz * b_dszz;
  auto c35_kl = 2 * b_dsxz * ad_dszz + ad_dsxz * b_dszz;
  auto c55_kl = 4 * ad_dsxz * b_dsxz;

  // Computing the rest of the integral.
  // rho from equation 14
  rho_kl = static_cast<type_real>(-1.0) * properties.rho() * dt * rho_kl;
  c11_kl = static_cast<type_real>(-1.0) * c11_kl * properties.c11() * dt;
  c13_kl = static_cast<type_real>(-1.0) * c13_kl * properties.c13() * dt;
  c15_kl = static_cast<type_real>(-1.0) * c15_kl * properties.c15() * dt;
  c33_kl = static_cast<type_real>(-1.0) * c33_kl * properties.c33() * dt;
  c35_kl = static_cast<type_real>(-1.0) * c35_kl * properties.c35() * dt;
  c55_kl = static_cast<type_real>(-1.0) * c55_kl * properties.c55() * dt;

  return { rho_kl, c11_kl, c13_kl, c15_kl, c33_kl, c35_kl, c55_kl };
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
                                 specfem::element::property_tag::anisotropic>,
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



  Note: Using # as adjoint modifier for the comments, so that `s` is the
  "standard" strainfield and `s#` is the adjoint strainfield. We use `ad_` as
  prefix for the adjoint wavefield and its derivatives, and `b_` as prefix for
  the "standard" backward wavefield and its derivatives.
  */

  // // ad_dsyx = 0.5 * (ds#y/dx + ds#x/dy) = 0.5 * (ds#y/dx)
  // const datatype ad_dsyx = static_cast<type_real>(0.5) *
  // adjoint_derivatives.du(0, 0);

  // // ad_dsyz = 0.5 * (ds#y/dz + ds#z/dy) = 0.5 * (ds#y/dz)
  // const datatype ad_dszz = static_cast<type_real>(0.5) *
  // adjoint_derivatives.du(1, 0);

  // // b_dsyx = 0.5 * (dsy/dx + dsx/dy) = 0.5 * dsy/dx
  // const datatype b_dsyx = static_cast<type_real>(0.5) *
  // backward_derivatives.du(0, 0);

  // // b_dsyz = 0.5 * (dsy/dz + dsz/dy) = 0.5 * dsz/dx
  // const datatype b_dsyz = static_cast<type_real>(0.5) *
  // backward_derivatives.du(1, 0));

  // // Inner part of the 2-D version of Equation 15 in Tromp et al. 2005
  // // That is \eps_{jk} \eps_{lm}
  // datatype c11_kl = 0; // ad_dsxx * b_dsxx
  // datatype c13_kl = ad_dsxx * b_dszz + ad_dszz * b_dsxx;
  // datatype c15_kl = 2 * ad_dsxx * b_dsxz + ad_dsxz * b_dsxx;
  // datatype c33_kl = ad_dszz * b_dszz;
  // datatype c35_kl = 2 * b_dsxz * ad_dszz + ad_dsxz * b_dszz;
  // datatype c55_kl = 4 * ad_dsxz * b_dsxz;

  // // Computing the rest of the integral.
  // // rho from equation 14
  // rho_kl = static_cast<type_real>(-1.0) * properties.rho * dt * rho_kl;
  // c11_kl = static_cast<type_real>(-1.0) * c11_kl * properties.c11 * dt;
  // c13_kl = static_cast<type_real>(-1.0) * c13_kl * properties.c13 * dt;
  // c15_kl = static_cast<type_real>(-1.0) * c15_kl * properties.c15 * dt;
  // c33_kl = static_cast<type_real>(-1.0) * c33_kl * properties.c33 * dt;
  // c35_kl = static_cast<type_real>(-1.0) * c35_kl * properties.c35 * dt;
  // c55_kl = static_cast<type_real>(-1.0) * c55_kl * properties.c55 * dt;

  /*
  I realized that we need the rest of the stiffness matrix for the SH wave,
  which is probably why anisotropic sh kernels aren't really supported in the
  specfem2d fortran code. That would require a larger update to the
  anisotropic properties. I will leave this as a placeholder for now.
  Specifically, we need the following additional properties:
  - c44
  - c45/c54 (symmetric)
  */

  Kokkos::abort("SH anisotropic kernels not implemented yet");

  return { 0, 0, 0, 0, 0, 0, 0 };
}

} // namespace medium
} // namespace specfem
