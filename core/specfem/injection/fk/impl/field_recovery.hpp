#pragma once

#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/impl/medium_interface_coupler.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Spectral field components at a single evaluation point for one
 *        frequency.
 *
 * Five complex components matching the @c field_f layout of the reference
 * implementation (fk_core.cpp).  For elastic points the layout is:
 * @f$(u_x\text{-term},\, u_z,\, T_{xx},\, T_{xz},\, T_{zz})@f$.
 * For acoustic points all three stress slots carry the pressure.
 */
struct PointSpectrum {
  Kokkos::complex<double> values[5] = {}; ///< [0]=u_x, [1]=u_z,
                                          ///< [2..4]=stress/pressure
};

/**
 * @brief Recover the elastic spectral field at one evaluation point.
 *
 * Faithful device port of fk_core.cpp lines 427-477.  The 4-component working
 * vector is propagated from the half-space eigenmatrix @p e_mat through the
 * appropriate chain and partial-layer propagator, then projected onto the five
 * output field components.
 *
 * @param e_mat             Half-space eigenmatrix @f$E@f$ (4×4 complex).
 * @param chain_above       Elastic chain matrix for the layer above the point
 *                          (@c el_chain[ilayer+1]); pass identity when
 *                          @p above_all_layers is true or when the point is in
 *                          the half-space.
 * @param point_layer       Layer record containing the evaluation point; used
 *                          for the partial propagator and @c elastic_gamma0.
 * @param halfspace         Bottom elastic half-space layer; supplies the
 *                          vertical slownesses for the half-space exponentials.
 * @param omega             Complex angular frequency (rad/s).
 * @param ray_parameter     Horizontal slowness @f$p@f$ (s/m).
 * @param in_halfspace      True when the point is inside the lower half-space.
 * @param above_all_layers  True when the point is above all finite layers
 *                          (@c ilayer == number_of_elastic_layers-1 in the
 *                          reference).
 * @param zz                Height of the point measured from the top of the
 *                          half-space (used only when @p in_halfspace is true).
 * @param height            Height of the point within @p point_layer (used
 *                          when @p in_halfspace and @p above_all_layers are
 *                          both false).
 * @param bot_vec           Bottom state vector assembled by the driver.
 * @param stf_coeff         Source-time-function apodization × phase delay
 *                          (driver-computed scalar).
 * @param xi1               Traction rotation material factor at the point.
 * @param xim               Traction rotation material factor at the point.
 * @param compute_stress    When true, populate stress components
 *                          @c values[2..4]; otherwise leave them zero.
 * @return PointSpectrum with the five complex field components.
 */
KOKKOS_INLINE_FUNCTION PointSpectrum recover_elastic_point(
    const specfem::utilities::ComplexMatrix<4> &e_mat,
    const specfem::utilities::ComplexMatrix<4> &chain_above,
    const specfem::injection::fk::ElasticIsotropicLayer &point_layer,
    const specfem::injection::fk::ElasticIsotropicLayer &halfspace,
    Kokkos::complex<double> omega, type_real ray_parameter, bool in_halfspace,
    bool above_all_layers, type_real zz, type_real height,
    const specfem::utilities::ComplexVector<4> &bot_vec,
    Kokkos::complex<double> stf_coeff, type_real xi1, type_real xim,
    bool compute_stress) {

  using Cplx = Kokkos::complex<double>;

  PointSpectrum out;

  specfem::utilities::ComplexVector<4> working;

  if (in_halfspace) {
    // -------------------------------------------------------------------------
    // Point in the lower half-space (ref lines 428-440):
    //   working(i) = sum_k E(i,k) * G[k] * bot_vec[k]
    // where G = diag{exp(om*eta_beta*zz), exp(-om*eta_beta*zz),
    //               exp(om*eta_alpha*zz), exp(-om*eta_alpha*zz)}.
    // -------------------------------------------------------------------------
    const auto slowness = vertical_slowness(halfspace, ray_parameter);
    const Cplx eta_alpha = slowness[0];
    const Cplx eta_beta = slowness[1];
    const double zz_d = static_cast<double>(zz);

    const Cplx G0 = Kokkos::exp(omega * eta_beta * zz_d);
    const Cplx G1 = Kokkos::exp(-omega * eta_beta * zz_d);
    const Cplx G2 = Kokkos::exp(omega * eta_alpha * zz_d);
    const Cplx G3 = Kokkos::exp(-omega * eta_alpha * zz_d);

    for (int i = 0; i < 4; ++i) {
      working[i] =
          e_mat(i, 0) * G0 * bot_vec[0] + e_mat(i, 1) * G1 * bot_vec[1] +
          e_mat(i, 2) * G2 * bot_vec[2] + e_mat(i, 3) * G3 * bot_vec[3];
    }
  } else {
    // -------------------------------------------------------------------------
    // Point in the finite layer stack (ref lines 441-456):
    //   tmp = E * bot_vec
    //   if above_all_layers: working = tmp
    //   else: N = g0 * layer_propagator(point_layer, omega, height, p) *
    //   chain_above
    //         working = N * tmp
    // -------------------------------------------------------------------------
    const auto tmp = e_mat * bot_vec;

    if (above_all_layers) {
      working = tmp;
    } else {
      const double g0 = elastic_gamma0(point_layer, ray_parameter);
      specfem::utilities::ComplexMatrix<4> P =
          layer_propagator(point_layer, omega, height, ray_parameter);
      specfem::utilities::ComplexMatrix<4> N = P * chain_above;
      // scale every entry by g0 (ref: scale(N, gamma0[ilayer-1]))
      for (int k = 0; k < 16; ++k) {
        N.data[k] = N.data[k] * Cplx(g0, 0.0);
      }
      working = N * tmp;
    }
  }

  // ---------------------------------------------------------------------------
  // Project onto the five output field components (ref lines 460-476).
  // ---------------------------------------------------------------------------
  const Cplx dx_f = working[0]; // y_1
  const Cplx dz_f = working[1]; // y_3

  out.values[0] = stf_coeff * dx_f * Cplx(0.0, -1.0);
  out.values[1] = stf_coeff * dz_f;

  if (compute_stress) {
    const Cplx txz_f = working[2]; // tilde{y}_4
    const Cplx tzz_f = working[3]; // tilde{y}_6
    const double p = static_cast<double>(ray_parameter);
    const double x1 = static_cast<double>(xi1);
    const double xm = static_cast<double>(xim);

    out.values[2] = stf_coeff * omega * Cplx(p, 0.0) *
                    (Cplx(x1, 0.0) * tzz_f - Cplx(4.0 * xm, 0.0) * dx_f);
    out.values[3] = stf_coeff * omega * Cplx(p, 0.0) * txz_f * Cplx(0.0, -1.0);
    out.values[4] = stf_coeff * omega * Cplx(p, 0.0) * tzz_f;
  }

  return out;
}

/**
 * @brief Recover the acoustic spectral field at one evaluation point.
 *
 * Faithful device port of fk_core.cpp lines 478-521.  The state vector is
 * propagated through the solid chain, coupled to the fluid domain, and then
 * advanced through the acoustic propagator chain to the point height.
 *
 * @param e_mat                      Half-space eigenmatrix @f$E@f$ (4×4).
 * @param elastic_chain_at_interface Elastic chain matrix
 * @f$\text{el\_chain}[\text{ilayer\_ac}]@f$; pass identity when @p
 * has_elastic_below is false.
 * @param has_elastic_below          True when there are elastic layers below
 * the fluid block (@c ilayer_ac <= nlayer-2 in the reference).
 * @param acoustic_chain_above       Acoustic chain matrix for the point's fluid
 *                                   layer (@c ac_chain[ilayer-1] in the
 *                                   reference).
 * @param point_layer                Acoustic layer containing the evaluation
 *                                   point; supplies density and p_velocity.
 * @param omega                      Complex angular frequency (rad/s).
 * @param ray_parameter              Horizontal slowness @f$p@f$ (s/m).
 * @param height                     Height of the point within @p point_layer.
 * @param top_fluid_thickness        Thickness @f$H@f$ of the topmost fluid
 * layer; used only when @p apply_free_surface_bc is true (free-surface check,
 * ref line 504).
 * @param apply_free_surface_bc      When true, zero the pressure component
 * after propagation to enforce the free-surface boundary condition.
 * @param bot_vec                    Bottom state vector from the driver.
 * @param stf_coeff                  Source-time-function apodization × phase
 *                                   delay.
 * @return PointSpectrum with five complex field components; all three stress
 *         slots carry the acoustic pressure.
 */
KOKKOS_INLINE_FUNCTION PointSpectrum recover_acoustic_point(
    const specfem::utilities::ComplexMatrix<4> &e_mat,
    const specfem::utilities::ComplexMatrix<4> &elastic_chain_at_interface,
    bool has_elastic_below,
    const specfem::utilities::ComplexMatrix<4> &acoustic_chain_above,
    const specfem::injection::fk::AcousticLayer &point_layer,
    Kokkos::complex<double> omega, type_real ray_parameter, type_real height,
    type_real top_fluid_thickness, bool apply_free_surface_bc,
    const specfem::utilities::ComplexVector<4> &bot_vec,
    Kokkos::complex<double> stf_coeff) {

  using Cplx = Kokkos::complex<double>;

  PointSpectrum out;

  // ---------------------------------------------------------------------------
  // Propagate through the elastic chain to the fluid/solid interface
  // (ref lines 482-487).
  // ---------------------------------------------------------------------------
  const auto tmp = e_mat * bot_vec;

  specfem::utilities::ComplexVector<4> working;
  if (has_elastic_below) {
    working = elastic_chain_at_interface * tmp;
  } else {
    working = tmp;
  }

  // ---------------------------------------------------------------------------
  // Couple the elastic state into the acoustic (fluid) domain (ref 488-490).
  // ---------------------------------------------------------------------------
  working = couple_solid_to_fluid(working);

  // ---------------------------------------------------------------------------
  // Apply the acoustic propagator chain to the point height (ref 494-501).
  // The 2×2 product Q = layer_propagator(point_layer, omega, height, p)
  //                       * acoustic_chain_above
  // is applied to the leading two components of working.
  // ---------------------------------------------------------------------------
  {
    specfem::utilities::ComplexMatrix<4> Q_partial =
        layer_propagator(point_layer, omega, height, ray_parameter);
    specfem::utilities::ComplexMatrix<4> Q = Q_partial * acoustic_chain_above;

    const Cplx b0 = working[0];
    const Cplx b1 = working[1];
    working[0] = Q(0, 0) * b0 + Q(0, 1) * b1;
    working[1] = Q(1, 0) * b0 + Q(1, 1) * b1;
  }

  // ---------------------------------------------------------------------------
  // Free-surface boundary condition: zero the pressure component when the
  // point is at the top of the fluid stack (ref line 504).
  // ---------------------------------------------------------------------------
  if (apply_free_surface_bc) {
    working[1] = Cplx(0.0, 0.0);
  }

  // ---------------------------------------------------------------------------
  // Displacement and pressure projections (ref lines 508-520).
  // ---------------------------------------------------------------------------
  const double p = static_cast<double>(ray_parameter);
  const double rho = static_cast<double>(point_layer.density);

  const Cplx dz_f = working[0]; // u_z
  const Cplx dx_f =
      Cplx(0.0, -1.0) * Cplx(p * p / rho, 0.0) * working[1]; // u_x

  out.values[0] = stf_coeff * dx_f;
  out.values[1] = stf_coeff * dz_f;

  const Cplx pres = working[1] * omega * Cplx(p, 0.0) * stf_coeff;
  out.values[2] = pres;
  out.values[3] = pres;
  out.values[4] = pres;

  return out;
}

} // namespace fk_impl
} // namespace injection
} // namespace specfem
