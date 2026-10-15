#pragma once

#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/injection/fk/impl/layer_operators.hpp"
#include "specfem/injection/fk/impl/medium_interface_coupler.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/series_layout.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

namespace specfem {
namespace injection {
namespace fk_impl {

// ===========================================================================
// Depth-grouped state-vector recovery API
//
// A plane wave in a laterally homogeneous model factorises as
// u(x,y,z,t) = U(z, t - tau(x,y)): horizontal position is only a time shift.
// These operators recover the depth-only spectral state once per depth group;
// the per-point source-time apodization, (i*omega)^n derivative, time delay
// tau, and Lame traction weights are applied later (host synthesis + device
// sampling), not here.
// ===========================================================================

// Series slot indices (@c series_horizontal .. @c number_of_series) live in
// series_layout.hpp — the layout contract shared with the device sampler.

/**
 * @brief Depth-only spectral series at one depth for one frequency.
 *
 * These carry NO source-time function, NO time delay, and NO per-point Lame
 * factors; those are applied per point downstream. Slot meanings follow the
 * @c series_* indices above.
 */
struct SeriesSpectrum {
  Kokkos::complex<double> values[number_of_series] = {};
};

/**
 * @brief Elastic state inside a finite layer: @f$ \gamma_0 \bar{P}(h)\, s @f$.
 *
 * Propagates the interface state at the bottom of @p layer up by @p height.
 * The @c gamma0 prefactor makes @f$ \gamma_0 \bar{P}(0) = I @f$, so the state
 * is continuous across interfaces (a point at height 0 returns @p
 * state_at_bottom).
 *
 * @param layer           Layer containing the point.
 * @param omega           Complex angular frequency (rad/s).
 * @param height          Height of the point above the layer bottom (m).
 * @param ray_parameter   Horizontal slowness p (s/m).
 * @param state_at_bottom State vector at the bottom of @p layer.
 * @return State vector @f$(u_x, u_z, \sigma_{xz}, \sigma_{zz})@f$ at the point.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexVector<4>
elastic_state_in_layer(
    const specfem::injection::fk::ElasticIsotropicLayer &layer,
    Kokkos::complex<double> omega, type_real height, type_real ray_parameter,
    const specfem::utilities::ComplexVector<4> &state_at_bottom) {
  const double g0 = elastic_gamma0(layer, ray_parameter);
  const specfem::utilities::ComplexMatrix<4> propagator =
      layer_propagator(layer, omega, height, ray_parameter);
  const specfem::utilities::ComplexVector<4> propagated =
      propagator * state_at_bottom;
  specfem::utilities::ComplexVector<4> out;
  for (int i = 0; i < 4; ++i) {
    out[i] = Kokkos::complex<double>(g0, 0.0) * propagated[i];
  }
  return out;
}

/**
 * @brief Elastic state inside the half-space: @f$ E\, G(zz)\, \text{modes} @f$.
 *
 * @param e_mat          Half-space eigenmatrix E.
 * @param halfspace      Half-space layer (supplies vertical slownesses).
 * @param omega          Complex angular frequency (rad/s).
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @param zz             Depth below the half-space top (<= 0 inside).
 * @param modes          Bottom mode vector b(omega).
 * @return State vector at the point in the half-space.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexVector<4>
elastic_state_in_halfspace(
    const specfem::utilities::ComplexMatrix<4> &e_mat,
    const specfem::injection::fk::ElasticIsotropicLayer &halfspace,
    Kokkos::complex<double> omega, type_real ray_parameter, double zz,
    const specfem::utilities::ComplexVector<4> &modes) {
  using Cplx = Kokkos::complex<double>;
  const auto slowness = vertical_slowness(halfspace, ray_parameter);
  const Cplx eta_alpha = slowness[0];
  const Cplx eta_beta = slowness[1];

  const Cplx g0 = Kokkos::exp(omega * eta_beta * zz);
  const Cplx g1 = Kokkos::exp(-omega * eta_beta * zz);
  const Cplx g2 = Kokkos::exp(omega * eta_alpha * zz);
  const Cplx g3 = Kokkos::exp(-omega * eta_alpha * zz);

  specfem::utilities::ComplexVector<4> out;
  for (int i = 0; i < 4; ++i) {
    out[i] = e_mat(i, 0) * g0 * modes[0] + e_mat(i, 1) * g1 * modes[1] +
             e_mat(i, 2) * g2 * modes[2] + e_mat(i, 3) * g3 * modes[3];
  }
  return out;
}

/**
 * @brief Acoustic state inside a fluid layer: @f$ \bar{Q}(h)\, \text{bottom}
 * @f$.
 *
 * @param layer          Fluid layer containing the point.
 * @param omega          Complex angular frequency (rad/s).
 * @param height         Height of the point above the layer bottom (m).
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @param bottom         Acoustic state at the bottom of @p layer.
 * @return Acoustic state @f$(u_z, P/k)@f$ at the point.
 */
KOKKOS_INLINE_FUNCTION AcousticState
acoustic_state_in_layer(const specfem::injection::fk::AcousticLayer &layer,
                        Kokkos::complex<double> omega, type_real height,
                        type_real ray_parameter, const AcousticState &bottom) {
  const specfem::utilities::ComplexMatrix<4> propagator =
      layer_propagator(layer, omega, height, ray_parameter);
  AcousticState out;
  out.u_z =
      propagator(0, 0) * bottom.u_z + propagator(0, 1) * bottom.pressure_over_k;
  out.pressure_over_k =
      propagator(1, 0) * bottom.u_z + propagator(1, 1) * bottom.pressure_over_k;
  return out;
}

/**
 * @brief Project an elastic state onto the depth-only series S0..S4.
 *
 * No source-time function and no per-point Lame factors: the driver forms
 * sigma_rr = xi1*S2 - 4*xim*S4 per point. S3/S4 and S2 are filled only when
 * @p compute_stress is true.
 *
 * @param state          State vector @f$(u_x, u_z, \sigma_{xz},
 * \sigma_{zz})@f$.
 * @param omega          Complex angular frequency (rad/s).
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @param compute_stress When true, fill the stress slots S2..S4.
 * @return SeriesSpectrum with the depth-only series values.
 */
KOKKOS_INLINE_FUNCTION SeriesSpectrum
project_elastic(const specfem::utilities::ComplexVector<4> &state,
                Kokkos::complex<double> omega, type_real ray_parameter,
                bool compute_stress) {
  using Cplx = Kokkos::complex<double>;
  const Cplx neg_i(0.0, -1.0);
  const Cplx dx = state[0];
  const Cplx dz = state[1];

  SeriesSpectrum out;
  out.values[series_horizontal] = neg_i * dx; // u_r
  out.values[series_vertical] = dz;           // u_z

  if (compute_stress) {
    const Cplx txz = state[2];
    const Cplx tzz = state[3];
    const Cplx omega_p = omega * Cplx(static_cast<double>(ray_parameter), 0.0);
    out.values[series_sigma_zz] = omega_p * tzz;         // S2
    out.values[series_sigma_rz] = neg_i * omega_p * txz; // S3
    out.values[series_dx] = omega_p * dx;                // S4
  }
  return out;
}

/**
 * @brief Project an acoustic state onto the depth-only series (S0, S1, S2).
 *
 * S0 = u_x, S1 = u_z (displacement), S2 = potential-rate χ̇. S3/S4 stay zero
 * (no shear in a fluid). The acoustic Stacey injection consumes the incident
 * displacement and χ̇ (SPECFEM3D's acoustic FK coupling), not pressure.
 *
 * @param state          Acoustic state @f$(u_z, P/k)@f$ at the point.
 * @param layer          Fluid layer containing the point (supplies density).
 * @param omega          Complex angular frequency (rad/s).
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @return SeriesSpectrum with S0, S1, S2 populated.
 */
KOKKOS_INLINE_FUNCTION SeriesSpectrum
project_acoustic(const AcousticState &state,
                 const specfem::injection::fk::AcousticLayer &layer,
                 Kokkos::complex<double> omega, type_real ray_parameter) {
  using Cplx = Kokkos::complex<double>;
  const double p = static_cast<double>(ray_parameter);
  const double rho = static_cast<double>(layer.density);
  const Cplx neg_i(0.0, -1.0);
  (void)omega; // χ̇ needs no (iω) factor; the kinematic derivative is elsewhere

  SeriesSpectrum out;
  out.values[series_horizontal] =
      neg_i * Cplx(p * p / rho, 0.0) * state.pressure_over_k; // u_x
  out.values[series_vertical] = state.u_z;                    // u_z
  // Potential-rate χ̇ = i·p·(P/k), matching SPECFEM3D dχ = i·ray_p·bot_vec(2).
  out.values[series_sigma_zz] =
      Cplx(0.0, 1.0) * Cplx(p, 0.0) * state.pressure_over_k; // chi_dot
  return out;
}

} // namespace fk_impl
} // namespace injection
} // namespace specfem
