#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "impl/fk_math_impl.hpp"
#include "impl/layer_operators.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>

namespace fk_injection::impl {

/**
 * @brief Eigenmatrix column of the incident mode.
 *
 * @param type Incident wave type (P or SV).
 * @return 2 (P-up) for P incidence, 0 (SV-up) for SV incidence.
 */
inline int incident_column(fk_injection::incident_wave_type type) {
  return type == fk_injection::incident_wave_type::p ? 2 : 0;
}

/**
 * @brief Compute the incident-wave amplitude in the bottom elastic half-space.
 *
 * Faithfully mirrors fk_core.cpp lines 171–188.  For a P-type incident wave
 * the amplitude is projected onto the P-wave polarisation:
 * @f$ C_3 = A \, i \, p \, v_P @f$.
 * For an SV-type incident wave:
 * @f$ C_1 = A \, p \, v_S @f$.
 *
 * @note The vertical-slowness values @f$\eta_P, \eta_S@f$ computed in the
 *       reference at the same location are used only for the per-point phase
 *       delay (@c Tdelay) in the driver kernel and are omitted here.
 *
 * @param halfspace     The last elastic layer (half-space) of the model.
 * @param type          Incident wave type (P or SV).
 * @param amplitude     Plane-wave amplitude scaling factor.
 * @param ray_parameter Horizontal slowness @f$p = \sin\theta / V@f$ (s/m).
 * @return Amplitude of the mode at @c incident_column(type).
 */
inline std::complex<double>
incident_amplitude(const fk_injection::ElasticIsotropicLayer &halfspace,
                   fk_injection::incident_wave_type type, type_real amplitude,
                   double ray_parameter) {
  const double A = static_cast<double>(amplitude);
  const double p = ray_parameter;
  const double vp = static_cast<double>(halfspace.p_velocity);
  const double vs = static_cast<double>(halfspace.s_velocity);

  if (type == fk_injection::incident_wave_type::p) {
    return std::complex<double>(0.0, 1.0) * A * p * vp;
  }
  return std::complex<double>(A * p * vs, 0.0);
}

/**
 * @brief Solve two free-surface conditions for the reflected mode amplitudes.
 *
 * Each row holds one condition's coefficients on the four modes. The reflected
 * modes (columns 1 and 3) are the unknowns and the incident mode is the
 * right-hand side, solved by Cramer's rule.
 *
 * @throws std::runtime_error if the determinant is zero or not finite.
 *
 * @param row0      First condition.
 * @param row1      Second condition.
 * @param incident  Column of the incident mode (see @c incident_column).
 * @param amplitude Incident amplitude (from @c incident_amplitude).
 * @return Amplitudes of the modes at columns 1 and 3.
 */
inline std::array<std::complex<double>, 2>
free_surface_coefficients(const fk_injection::ComplexVector<4> &row0,
                          const fk_injection::ComplexVector<4> &row1,
                          int incident, std::complex<double> amplitude) {
  using Cplx = std::complex<double>;

  const Cplx a = row0[1];
  const Cplx b = row0[3];
  const Cplx c = row1[1];
  const Cplx d = row1[3];
  const Cplx delta = a * d - b * c;

  const double delta_magnitude = std::abs(delta);
  if (!std::isfinite(delta_magnitude) || delta_magnitude == 0.0) {
    std::ostringstream msg;
    msg << "free_surface_coefficients: singular free-surface system, |delta| = "
        << delta_magnitude;
    throw std::runtime_error(msg.str());
  }

  return { -(d * row0[incident] - b * row1[incident]) / delta * amplitude,
           -(-c * row0[incident] + a * row1[incident]) / delta * amplitude };
}

/**
 * @brief Solve for the two unknown half-space coefficients of an elastic-only
 *        model from the free-surface zero-traction condition.
 *
 * Faithfully mirrors fk_core.cpp lines 287-302. Given the propagated elastic
 * matrix @f$ N = (\text{elastic chain}) \cdot E @f$, the conditions are the
 * traction rows of @f$ N @f$ (rows 2 and 3 in 0-based indexing).
 *
 * @param n_mat     Propagated elastic matrix @f$ N @f$ (4×4 complex).
 * @param type      Incident wave type (P or SV).
 * @param amplitude Incident amplitude (from @c incident_amplitude).
 * @return Array of two complex coefficients {coeff0, coeff1}.
 */
inline std::array<std::complex<double>, 2>
halfspace_coefficients(const fk_injection::ComplexMatrix<4> &n_mat,
                       fk_injection::incident_wave_type type,
                       std::complex<double> amplitude) {
  fk_injection::ComplexVector<4> row0;
  fk_injection::ComplexVector<4> row1;
  for (int j = 0; j < 4; ++j) {
    row0[j] = n_mat(2, j);
    row1[j] = n_mat(3, j);
  }
  return free_surface_coefficients(row0, row1, incident_column(type),
                                   amplitude);
}

/**
 * @brief Solve for the two unknown half-space coefficients of a model with a
 *        fluid block from the free-surface condition.
 *
 * Faithfully mirrors fk_core.cpp lines 303-328. The conditions are zero shear
 * traction at the fluid base (row 2 of @f$ N @f$) and zero pressure at the free
 * surface, which is row 1 of the fluid propagator applied to the base state.
 *
 * @param n_mat     Propagated elastic matrix @f$ N @f$ (4×4 complex).
 * @param fluid_propagator Full acoustic propagator of the fluid block.
 * @param type      Incident wave type (P or SV).
 * @param amplitude Incident amplitude (from @c incident_amplitude).
 * @return Array of two complex coefficients {coeff0, coeff1}.
 */
inline std::array<std::complex<double>, 2>
halfspace_coefficients(const fk_injection::ComplexMatrix<4> &n_mat,
                       const fk_injection::ComplexMatrix<2> &fluid_propagator,
                       fk_injection::incident_wave_type type,
                       std::complex<double> amplitude) {
  fk_injection::ComplexVector<4> row0;
  fk_injection::ComplexVector<4> row1;
  for (int j = 0; j < 4; ++j) {
    row0[j] = n_mat(2, j);
    row1[j] = fluid_propagator(1, 0) * n_mat(1, j) -
              fluid_propagator(1, 1) * n_mat(3, j);
  }
  return free_surface_coefficients(row0, row1, incident_column(type),
                                   amplitude);
}

/**
 * @brief Assemble the 4-component bottom state vector for a given frequency.
 *
 * Faithfully mirrors fk_core.cpp lines 414–425, except for the SV placement
 * noted below.  The bottom vector encodes the incident + reflected mode
 * amplitudes at the base of the model (index layout: SV-up | SV-down | P-up |
 * P-down):
 *
 * - **P incidence**:
 *   @f$ b_1 = \text{coeff}_0,\; b_2 = C_3,\; b_3 = \text{coeff}_1,\; b_0 = 0
 * @f$
 * - **SV incidence**:
 *   @f$ b_0 = C_1,\; b_1 = \text{coeff}_0,\; b_3 = \text{coeff}_1,\; b_2 = 0
 * @f$
 *
 * @note Deviation from the reference, intentional: SPECFEM3D's
 *       couple_with_injection.f90 (bot_vec(3)=coeff(1), i.e. 0-based index 2)
 *       and the Westervelt C++ port place the reflected SV-down amplitude at
 *       column 2 (the incident P-up slot), leaving column 1 zero. Because the
 *       solve uses columns 1 and 3, that placement violates the free-surface
 *       zero-traction condition for SV. See
 *       docs/specfem3d-sv-fk-bug-report.md.
 *
 * @param type      Incident wave type (P or SV).
 * @param amplitude Incident amplitude (from @c incident_amplitude).
 * @param coeffs    Two half-space coefficients from
 *                  @c halfspace_coefficients.
 * @return 4-component complex bottom vector.
 */
inline fk_injection::ComplexVector<4>
build_bottom_vector(fk_injection::incident_wave_type type,
                    std::complex<double> amplitude,
                    const std::array<std::complex<double>, 2> &coeffs) {
  fk_injection::ComplexVector<4> bot{};
  bot[incident_column(type)] = amplitude;
  bot[1] = coeffs[0];
  bot[3] = coeffs[1];
  return bot;
}

} // namespace fk_injection::impl
