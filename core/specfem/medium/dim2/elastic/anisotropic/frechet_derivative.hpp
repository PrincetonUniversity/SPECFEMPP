#pragma once

#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/algorithms.hpp"
#include "specfem/point.hpp"

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_frechet_derivative_dim2_elastic_anisotropic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim2_elastic_anisotropic
 * @brief Compute Fréchet derivatives for 2D elastic PSV anisotropic media.
 *
 * Calculates sensitivity kernels for anisotropic elastic stiffness parameters
 * in PSV wave propagation. Computes kernels for density and six independent
 * stiffness coefficients (c11, c13, c15, c33, c35, c55) using strain tensor
 * formulation.
 *
 * Based on Tromp et al. 2005, Equation 15 for anisotropic kernels:
 * \f[
 * \Delta K_{c_{ijkl}} = -\varepsilon_{ij}^{\dagger} \varepsilon_{kl}^b \,
 * c_{ijkl} \,
 * \Delta t
 * \f]
 * \f[
 * \Delta K_{rho} = -\rho \Delta t \, \ddot{u}^{\dagger} \cdot u^b
 * \f]
 *
 * @tparam PointPropertiesType Anisotropic material properties
 * @tparam AdjointPointVelocityType Adjoint velocity field
 * @tparam AdjointPointAccelerationType Adjoint acceleration field
 * @tparam BackwardPointDisplacementType Backward displacement field
 * @tparam PointFieldDerivativesType Spatial field derivatives
 *
 * @param properties Anisotropic elastic properties (ρ, c11, c13, c15, c33, c35,
 * c55)
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing density and stiffness parameter
 * sensitivities
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

/**
 * @ingroup specfem_medium_frechet_derivative_dim2_elastic_anisotropic
 * @brief Compute Fréchet derivatives for 2D elastic SH anisotropic media.
 *
 * Placeholder implementation for SH anisotropic kernels. Currently not
 * implemented due to missing stiffness matrix components (c44, c45/c54) in the
 * anisotropic properties structure.
 *
 * @tparam PointPropertiesType Anisotropic material properties
 * @tparam AdjointPointVelocityType Adjoint velocity field
 * @tparam AdjointPointAccelerationType Adjoint acceleration field
 * @tparam BackwardPointDisplacementType Backward displacement field
 * @tparam PointFieldDerivativesType Spatial field derivatives
 *
 * @param properties Anisotropic elastic properties
 * @param adjoint_velocity Adjoint velocity field
 * @param adjoint_acceleration Adjoint acceleration field
 * @param backward_displacement Backward displacement field
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Zero kernels (not implemented)
 *
 * @warning This function currently calls Kokkos::abort() - not implemented
 * @note Requires additional stiffness coefficients (c44, c45) for full
 * implementation
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
