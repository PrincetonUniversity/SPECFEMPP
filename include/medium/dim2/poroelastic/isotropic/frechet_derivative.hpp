#pragma once

#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/algorithms.hpp"
#include "specfem/point.hpp"

#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_frechet_derivative_dim2_poroelastic
 *
 */

/**
 * @ingroup specfem_medium_frechet_derivative_dim2_poroelastic
 * @brief Compute Fréchet derivatives for 2D poroelastic isotropic media.
 *
 * Calculates sensitivity kernels for poroelastic properties based on Biot's
 * theory. Computes kernels for both solid and fluid phases, including
 * density, elastic moduli, Biot parameters, and wave speed kernels.
 *
 * Based on Morency et al. 2009 formulation, computes 15 different kernels:
 * - Primary kernels: ρₜ, ρf, η, s/m, μfr, B, C, M
 * - Wave speed kernels: cpI, cpII, cs, ratio
 * - Density normalized kernels: ρb, ρfb, φ
 *
 * @tparam PointPropertiesType Poroelastic material properties
 * @tparam AdjointPointVelocityType Adjoint velocity field (solid+fluid)
 * @tparam AdjointPointAccelerationType Adjoint acceleration field (solid+fluid)
 * @tparam BackwardPointDisplacementType Backward displacement field
 * (solid+fluid)
 * @tparam PointFieldDerivativesType Spatial field derivatives
 *
 * @param properties Poroelastic properties (ρ, μ, φ, η, permeability, Biot
 * coefficients)
 * @param adjoint_velocity Adjoint velocity field for solid and fluid phases
 * @param adjoint_acceleration Adjoint acceleration field for solid and fluid
 * phases
 * @param backward_displacement Backward displacement field for solid and fluid
 * phases
 * @param adjoint_derivatives Spatial derivatives of adjoint field
 * @param backward_derivatives Spatial derivatives of backward field
 * @param dt Time step size
 * @return Point kernels containing all 15 poroelastic parameter sensitivities
 *
 * @warning This implementation has NOT been tested. Please create a GitHub
 * issue if you encounter bugs or unexpected behavior in the kernel
 * computations.
 * @note For complete technical details and theoretical derivations, please
 * refer to: <a href="https://doi.org/10.1111/j.1365-246X.2009.04332.x">Morency,
 * C., Luo, Y., & Tromp, J. (2009).</a>
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
                                 specfem::element::medium_tag::poroelastic>,
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
  In the following the explanation for the poroelastic kernels is given.

  Computing the solid deviatoric strain tensor:

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
  // solid phase
  // ad_dsxx = 0.5 * (ds#x/dx + ds#x/dx) = ds#x/dx
  const auto ad_dsxx = adjoint_derivatives.du(0, 0);

  // ad_dsxz = 0.5 * (ds#x/dz + ds#z/dx)
  const auto ad_dsxz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(1, 0) + adjoint_derivatives.du(0, 1));

  // ad_dszz = 0.5 * (ds#z/dz + ds#z/dz) = ds#z/dz
  const auto ad_dszz = adjoint_derivatives.du(1, 1);

  // fluid phase
  // ad_dwxx = 0.5 * (dw#x/dx + dw#x/dx) = dw#x/dx
  const auto ad_dwxx = adjoint_derivatives.du(2, 0);

  // ad_dwxz = 0.5 * (dw#x/dz + dw#z/dx)
  const auto ad_dwxz =
      static_cast<type_real>(0.5) *
      (adjoint_derivatives.du(2, 1) + adjoint_derivatives.du(3, 0));

  // ad_dwzz = 0.5 * (dw#z/dz + dw#z/dz) = dw#z/dz
  const auto ad_dwzz = adjoint_derivatives.du(3, 1);

  // Compute the gradient of the backward field
  // solid phase
  // b_dsxx = 0.5 * (dsx/dx + dsx/dx) = dsx/dx
  const auto b_dsxx = backward_derivatives.du(0, 0);

  // b_dsxz = 0.5 * (dsx/dz + dsz/dx)
  const auto b_dsxz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(1, 0) + backward_derivatives.du(0, 1));

  // b_dszz = 0.5 * (dsz/dz + dsz/dz) = dsz/dz
  const auto b_dszz = backward_derivatives.du(1, 1);

  // fluid phase
  // b_dwxx = 0.5 * (dwx/dx + dwx/dx) = dwx/dx
  const auto b_dwxx = backward_derivatives.du(2, 0);

  // b_dwxz = 0.5 * (dwx/dz + dwz/dx)
  const auto b_dwxz =
      static_cast<type_real>(0.5) *
      (backward_derivatives.du(2, 1) + backward_derivatives.du(3, 0));

  // b_dwzz = 0.5 * (dwz/dz + dwz/dz) = dwz/dz
  const auto b_dwzz = backward_derivatives.du(3, 1);

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

  // Will be used in several expressions
  // div(s#) * div(s)
  auto kappa_kl = (ad_dsxx + ad_dszz) * (b_dsxx + b_dszz);

  // Part of Morency et al. 2009, Eq 46
  // [eps+ : eps] - 1/3 [div (s#) * div(s)]
  // I am not clear on how we get to the following form but from the
  // GPU cuda code from the fortran code I assume that there is an
  // assumption being made that eps#_i * eps_j = eps#_j * eps_i in the
  // isotropic case due to the symmetry of the voigt notation stiffness
  // matrix.
  auto mufr_kl = (ad_dsxx * b_dsxx + ad_dszz * b_dszz +
                  static_cast<type_real>(2.0) * ad_dsxz * b_dsxz -
                  static_cast<type_real>(1.0 / 3.0) * kappa_kl);

  // This notation/naming is confusing with respect to the physics.
  // Should be forward.acceleration dotted with adjoint displacement
  // See Morency et al. 2009, Equation 39
  auto rhot_kl = adjoint_acceleration(0) * backward_displacement(0) +
                 adjoint_acceleration(1) * backward_displacement(1);
  // See Morency et al. 2009, Equation 40
  auto rhof_kl = adjoint_acceleration(2) * backward_displacement(0) +
                 adjoint_acceleration(3) * backward_displacement(1) +
                 adjoint_acceleration(0) * backward_displacement(2) +
                 adjoint_acceleration(1) * backward_displacement(3);
  // See Morency et al. 2009, Equation 41
  auto sm_kl = adjoint_acceleration(2) * backward_displacement(2) +
               adjoint_acceleration(3) * backward_displacement(3);

  // Viscous term, see Morency et al. 2009, Equation 42
  auto eta_kl = adjoint_velocity(2) * backward_displacement(2) +
                adjoint_velocity(3) * backward_displacement(3);

  // Biot bulk moduli
  // B based on Morency et al. 2009, Equation 43
  auto B_kl = kappa_kl;
  // C based on Morency et al. 2009, Equation 44
  auto C_kl = (ad_dsxx + ad_dszz) * (b_dwxx + b_dwzz) +
              (ad_dwxx + ad_dwzz) * (b_dsxx + b_dszz);
  // M based on Morency et al. 2009, Equation 45
  auto M_kl = (ad_dwxx + ad_dwzz) * (b_dwxx + b_dwzz);

  //

  // Finishing the kernels
  mufr_kl = static_cast<type_real>(-2.0) * properties.mu_G() * dt * mufr_kl;
  rhot_kl = static_cast<type_real>(-1.0) * properties.rho_bar() * dt * rhot_kl;
  rhof_kl = static_cast<type_real>(-1.0) * properties.rho_f() * dt * rhof_kl;
  sm_kl = static_cast<type_real>(-1.0) *
          (properties.rho_f() * properties.tortuosity() / properties.phi()) *
          dt * sm_kl;
  eta_kl = static_cast<type_real>(-1.0) *
           (properties.eta_f() / properties.permxx()) * dt * eta_kl;
  B_kl = static_cast<type_real>(-1.0) *
         (properties.H_Biot() -
          static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
         dt * B_kl;
  C_kl = static_cast<type_real>(-1.0) * properties.C_Biot() * dt * C_kl;
  M_kl = static_cast<type_real>(-1.0) * properties.M_Biot() * dt * M_kl;

  // Density normalized kernels - directly defined in point/kernels.hpp but
  // needed below rhob: First term in Equation 50
  const auto rhob_kl = rhot_kl + B_kl + mufr_kl;
  // rhofb: Second term in Equation 50
  const auto rhofb_kl = rhof_kl + C_kl + M_kl + sm_kl;
  // phi: Third term in Equation 50
  const auto phi_kl = static_cast<type_real>(-1.0) * (M_kl + sm_kl);
  const auto Bb_kl = B_kl;
  const auto Cb_kl = C_kl;
  const auto Mb_kl = M_kl;
  const auto mufrb_kl = mufr_kl;

  // Wavespeed kernels
  // Some parameters to simplify kernels expressions
  // Approximated velocities (no viscous dissipation)
  const auto phi_over_tort = properties.phi() / properties.tortuosity();

  const auto afactor =
      properties.rho_bar() - phi_over_tort * properties.rho_f();
  const auto bfactor =
      properties.H_Biot() +
      phi_over_tort * properties.rho_bar() / properties.rho_f() *
          properties.M_Biot() -
      static_cast<type_real>(2.0) * phi_over_tort * properties.C_Biot();
  const auto cfactor = phi_over_tort / properties.rho_f() *
                       (properties.H_Biot() * properties.M_Biot() -
                        properties.C_Biot() * properties.C_Biot());

  const auto cpIsquare =
      (bfactor + Kokkos::sqrt(bfactor * bfactor - static_cast<type_real>(4.0) *
                                                      afactor * cfactor)) /
      (static_cast<type_real>(2.0) * afactor);
  const auto cpIIsquare =
      (bfactor - Kokkos::sqrt(bfactor * bfactor - static_cast<type_real>(4.0) *
                                                      afactor * cfactor)) /
      (static_cast<type_real>(2.0) * afactor);
  const auto cssquare = properties.mu_G() / afactor;

  // Approximated ratio r = amplitude "w" field/amplitude "s" field (no viscous
  // dissipation) used later for wavespeed kernels calculation, which are
  // presently implemented for inviscid case, contrary to primary and
  // density-normalized kernels, which are consistent with viscous fluid case.
  const auto gamma1 = properties.H_Biot() - phi_over_tort * properties.C_Biot();
  const auto gamma2 = properties.C_Biot() - phi_over_tort * properties.M_Biot();
  const auto gamma3 =
      phi_over_tort *
      (properties.M_Biot() * (afactor / properties.rho_f() + phi_over_tort) -
       properties.C_Biot());
  const auto gamma4 =
      phi_over_tort *
      (properties.C_Biot() * (afactor / properties.rho_f() + phi_over_tort) -
       properties.H_Biot());

  const auto ratio =
      static_cast<type_real>(0.5) * (gamma1 - gamma3) / gamma4 +
      static_cast<type_real>(0.5) *
          Kokkos::sqrt((gamma1 - gamma3) * (gamma1 - gamma3) /
                           (gamma4 * gamma4) +
                       static_cast<type_real>(4.0) * gamma2 / gamma4);

  const auto phi_over_tort_ratio = phi_over_tort * ratio;
  const auto ratio_square = ratio * ratio;

  // wavespeed kernels
  const auto dd1 = (static_cast<type_real>(1.0) +
                    properties.rho_bar() / properties.rho_f() - phi_over_tort) *
                       ratio_square +
                   static_cast<type_real>(2.0) * ratio +
                   static_cast<type_real>(1.0) / phi_over_tort;

  const auto rhobb_kl =
      rhob_kl -
      phi_over_tort * properties.rho_f() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (cpIIsquare +
           (cpIsquare - cpIIsquare) *
               ((phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 +
                (properties.rho_bar() * properties.rho_bar() * ratio_square /
                 (properties.rho_f() * properties.rho_f()) *
                 (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
                 (phi_over_tort_ratio +
                  phi_over_tort * (static_cast<type_real>(1.0) +
                                   properties.rho_f() / properties.rho_bar()) -
                  static_cast<type_real>(1.0))) /
                    (dd1 * dd1)) -
           static_cast<type_real>(4.0 / 3.0) * cssquare) *
          B_kl -
      properties.rho_bar() * ratio_square / properties.M_Biot() *
          (cpIsquare - cpIIsquare) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) / (dd1 * dd1) *
          M_kl +
      properties.rho_bar() * ratio / properties.C_Biot() *
          (cpIsquare - cpIIsquare) *
          ((phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 -
           phi_over_tort_ratio *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (static_cast<type_real>(1.0) +
                properties.rho_bar() * ratio / properties.rho_f()) /
               (dd1 * dd1)) *
          C_kl +
      phi_over_tort * properties.rho_f() * cssquare / properties.mu_G() *
          mufrb_kl;

  const auto rhofbb_kl =
      rhofb_kl +
      phi_over_tort * properties.rho_f() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (cpIIsquare +
           (cpIsquare - cpIIsquare) *
               ((phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 +
                (properties.rho_bar() * properties.rho_bar() * ratio_square /
                 (properties.rho_f() * properties.rho_f()) *
                 (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
                 (phi_over_tort_ratio +
                  phi_over_tort * (static_cast<type_real>(1.0) +
                                   properties.rho_f() / properties.rho_bar()) -
                  static_cast<type_real>(1.0))) /
                    (dd1 * dd1)) -
           static_cast<type_real>(4.0 / 3.0) * cssquare) *
          B_kl +
      properties.rho_bar() * ratio_square / properties.M_Biot() *
          (cpIsquare - cpIIsquare) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) / (dd1 * dd1) *
          M_kl -
      properties.rho_bar() * ratio / properties.C_Biot() *
          (cpIsquare - cpIIsquare) *
          ((phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 -
           phi_over_tort_ratio *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (static_cast<type_real>(1.0) +
                properties.rho_bar() * ratio / properties.rho_f()) /
               (dd1 * dd1)) *
          C_kl -
      phi_over_tort * properties.rho_f() * cssquare / properties.mu_G() *
          mufrb_kl;

  const auto phib_kl =
      phi_kl -
      phi_over_tort * properties.rho_bar() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (cpIsquare - properties.rho_f() / properties.rho_bar() * cpIIsquare -
           (cpIsquare - cpIIsquare) *
               ((static_cast<type_real>(2.0) * ratio_square * phi_over_tort +
                 (static_cast<type_real>(1.0) +
                  properties.rho_f() / properties.rho_bar()) *
                     (static_cast<type_real>(2.0) * ratio * phi_over_tort +
                      static_cast<type_real>(1.0))) /
                    dd1 +
                (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
                    (phi_over_tort_ratio +
                     phi_over_tort *
                         (static_cast<type_real>(1.0) +
                          properties.rho_f() / properties.rho_bar()) -
                     static_cast<type_real>(1.0)) *
                    ((static_cast<type_real>(1.0) +
                      properties.rho_bar() / properties.rho_f() -
                      static_cast<type_real>(2.0) * phi_over_tort) *
                         ratio_square +
                     static_cast<type_real>(2.0) * ratio) /
                    (dd1 * dd1)) -
           static_cast<type_real>(4.0 / 3.0) * properties.rho_f() * cssquare /
               properties.rho_bar()) *
          B_kl +
      properties.rho_f() / properties.M_Biot() * (cpIsquare - cpIIsquare) *
          (static_cast<type_real>(2.0) * ratio *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               ((static_cast<type_real>(1.0) +
                 properties.rho_bar() / properties.rho_f() -
                 static_cast<type_real>(2.0) * phi_over_tort) *
                    ratio_square +
                static_cast<type_real>(2.0) * ratio) /
               (dd1 * dd1)) *
          M_kl +
      phi_over_tort * properties.rho_f() / properties.C_Biot() *
          (cpIsquare - cpIIsquare) * ratio *
          ((static_cast<type_real>(1.0) +
            properties.rho_f() / properties.rho_bar() * ratio) /
               dd1 -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (static_cast<type_real>(1.0) +
                properties.rho_bar() / properties.rho_f() * ratio) *
               ((static_cast<type_real>(1.0) +
                 properties.rho_bar() / properties.rho_f() -
                 static_cast<type_real>(2.0) * phi_over_tort) *
                    ratio +
                static_cast<type_real>(2.0)) /
               (dd1 * dd1)) *
          C_kl -
      phi_over_tort * properties.rho_f() * cssquare / properties.mu_G() *
          mufrb_kl;

  const auto cpI_kl =
      static_cast<type_real>(2.0) * cpIsquare * properties.rho_bar() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (static_cast<type_real>(1.0) - phi_over_tort +
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio +
                phi_over_tort * (static_cast<type_real>(1.0) +
                                 properties.rho_f() / properties.rho_bar()) -
                static_cast<type_real>(1.0)) /
               dd1) *
          B_kl +
      static_cast<type_real>(2.0) * cpIsquare * properties.rho_f() *
          properties.tortuosity() / (properties.phi() * properties.M_Biot()) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 * M_kl +
      static_cast<type_real>(2.0) * cpIsquare * properties.rho_f() /
          properties.C_Biot() *
          (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
          (static_cast<type_real>(1.0) +
           properties.rho_bar() / properties.rho_f() * ratio) /
          dd1 * C_kl;

  const auto cpII_kl =
      static_cast<type_real>(2.0) * cpIIsquare * properties.rho_bar() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (phi_over_tort * properties.rho_f() / properties.rho_bar() -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio +
                phi_over_tort * (static_cast<type_real>(1.0) +
                                 properties.rho_f() / properties.rho_bar()) -
                static_cast<type_real>(1.0)) /
               dd1) *
          B_kl +
      static_cast<type_real>(2.0) * cpIIsquare * properties.rho_f() *
          properties.tortuosity() / (properties.phi() * properties.M_Biot()) *
          (static_cast<type_real>(1.0) -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1) *
          M_kl +
      static_cast<type_real>(2.0) * cpIIsquare * properties.rho_f() /
          properties.C_Biot() *
          (static_cast<type_real>(1.0) -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (static_cast<type_real>(1.0) +
                properties.rho_bar() / properties.rho_f() * ratio) /
               dd1) *
          C_kl;

  const auto cs_kl =
      static_cast<type_real>(-8.0 / 3.0) * cssquare * properties.rho_bar() /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (static_cast<type_real>(1.0) -
           phi_over_tort * properties.rho_f() / properties.rho_bar()) *
          B_kl +
      static_cast<type_real>(2.0) *
          (properties.rho_bar() - properties.rho_f() * phi_over_tort) /
          properties.mu_G() * cssquare * mufrb_kl;

  const auto ratio_kl =
      ratio * properties.rho_bar() * phi_over_tort /
          (properties.H_Biot() -
           static_cast<type_real>(4.0 / 3.0) * properties.mu_G()) *
          (cpIsquare - cpIIsquare) *
          (phi_over_tort *
               (static_cast<type_real>(2.0) * ratio +
                static_cast<type_real>(1.0) +
                properties.rho_f() / properties.rho_bar()) /
               dd1 -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio +
                phi_over_tort * (static_cast<type_real>(1.0) +
                                 properties.rho_f() / properties.rho_bar()) -
                static_cast<type_real>(1.0)) *
               (static_cast<type_real>(2.0) * ratio *
                    (static_cast<type_real>(1.0) +
                     properties.rho_bar() / properties.rho_f() -
                     phi_over_tort) +
                static_cast<type_real>(2.0)) /
               (dd1 * dd1)) *
          B_kl +
      ratio * properties.rho_f() * properties.tortuosity() /
          (properties.phi() * properties.M_Biot()) * (cpIsquare - cpIIsquare) *
          static_cast<type_real>(2.0) * phi_over_tort *
          ((phi_over_tort_ratio + static_cast<type_real>(1.0)) / dd1 -
           (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               ((static_cast<type_real>(1.0) +
                 properties.rho_bar() / properties.rho_f() - phi_over_tort) *
                    ratio +
                static_cast<type_real>(1.0)) /
               (dd1 * dd1)) *
          M_kl +
      ratio * properties.rho_f() / properties.C_Biot() *
          (cpIsquare - cpIIsquare) *
          ((static_cast<type_real>(2.0) * phi_over_tort_ratio *
                properties.rho_bar() / properties.rho_f() +
            phi_over_tort + properties.rho_bar() / properties.rho_f()) /
               dd1 -
           static_cast<type_real>(2.0) * phi_over_tort *
               (phi_over_tort_ratio + static_cast<type_real>(1.0)) *
               (static_cast<type_real>(1.0) +
                properties.rho_bar() / properties.rho_f() * ratio) *
               ((static_cast<type_real>(1.0) +
                 properties.rho_bar() / properties.rho_f() - phi_over_tort) *
                    ratio +
                static_cast<type_real>(1.0)) /
               (dd1 * dd1)) *
          C_kl;

  return { rhot_kl, rhof_kl,  eta_kl,    sm_kl,    mufr_kl,
           B_kl,    C_kl,     M_kl,      cpI_kl,   cpII_kl,
           cs_kl,   rhobb_kl, rhofbb_kl, ratio_kl, phib_kl };
}

} // namespace medium
} // namespace specfem
