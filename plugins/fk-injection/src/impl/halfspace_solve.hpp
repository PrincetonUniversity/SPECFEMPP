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

namespace fk_injection::impl {

/**
 * @brief Incident-wave amplitudes in the bottom elastic half-space.
 *
 * Stores the two independent incoming-wave coefficients obtained from the
 * free-surface radiation condition and the wave type.  For an incident P wave
 * only @c c3 is non-zero; for an incident SV wave only @c c1 is non-zero.
 *
 * These correspond directly to the @f$ C_3 @f$ (P) and @f$ C_1 @f$ (SV)
 * amplitudes in the Westervelt / Fortran FK reference implementation.
 */
struct IncidentAmplitude {
  std::complex<double> c1 =
      std::complex<double>(0.0, 0.0); ///< Incoming SV amplitude C_1.
  std::complex<double> c3 =
      std::complex<double>(0.0, 0.0); ///< Incoming P amplitude C_3.
};

/**
 * @brief Compute the incident-wave amplitudes in the bottom elastic half-space.
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
 * @return IncidentAmplitude with @c c3 set for P, @c c1 set for SV.
 */
inline IncidentAmplitude
incident_amplitude(const fk_injection::ElasticIsotropicLayer &halfspace,
                   fk_injection::incident_wave_type type, type_real amplitude,
                   double ray_parameter) {
  const double A = static_cast<double>(amplitude);
  const double p = ray_parameter;
  const double vp = static_cast<double>(halfspace.p_velocity);
  const double vs = static_cast<double>(halfspace.s_velocity);

  IncidentAmplitude result;
  if (type == fk_injection::incident_wave_type::p) {
    result.c3 = std::complex<double>(0.0, 1.0) * A * p * vp;
    result.c1 = std::complex<double>(0.0, 0.0);
  } else {
    result.c1 = std::complex<double>(A * p * vs, 0.0);
    result.c3 = std::complex<double>(0.0, 0.0);
  }
  return result;
}

/**
 * @brief Solve for the two unknown half-space coefficients of an elastic-only
 *        model from the free-surface zero-traction condition.
 *
 * Faithfully mirrors fk_core.cpp lines 287-302. Given the propagated elastic
 * matrix @f$ N = (\text{elastic chain}) \cdot E @f$, the free-surface condition
 * reduces to a 2×2 system on the traction rows of @f$ N @f$ (rows 2 and 3 in
 * 0-based indexing) using the up-going mode columns (columns 1 and 3 for
 * P-type; columns 0 and 3 for SV-type), solved analytically.
 *
 * If the determinant @f$|\delta| \leq \epsilon@f$ (singular configuration) both
 * coefficients are set to zero.
 *
 * @param n_mat Propagated elastic matrix @f$ N @f$ (4×4 complex).
 * @param type  Incident wave type (P or SV).
 * @param c1    Incoming SV amplitude (from @c incident_amplitude).
 * @param c3    Incoming P amplitude  (from @c incident_amplitude).
 * @return Array of two complex coefficients {coeff0, coeff1}.
 */
inline std::array<std::complex<double>, 2>
halfspace_coefficients(const fk_injection::ComplexMatrix<4> &n_mat,
                       fk_injection::incident_wave_type type,
                       std::complex<double> c1, std::complex<double> c3) {
  using Cplx = std::complex<double>;
  const Cplx zero(0.0, 0.0);

  std::array<Cplx, 2> coeffs;
  coeffs[0] = zero;
  coeffs[1] = zero;

  const bool is_p = (type == fk_injection::incident_wave_type::p);

  const Cplx a = n_mat(2, 1);
  const Cplx b = n_mat(2, 3);
  const Cplx c = n_mat(3, 1);
  const Cplx d = n_mat(3, 3);
  const Cplx delta = a * d - b * c;

  if (std::abs(delta) > tinyval) {
    if (is_p) {
      coeffs[0] = -(d * n_mat(2, 2) - b * n_mat(3, 2)) / delta * c3;
      coeffs[1] = -(-c * n_mat(2, 2) + a * n_mat(3, 2)) / delta * c3;
    } else {
      coeffs[0] = -(d * n_mat(2, 0) - b * n_mat(3, 0)) / delta * c1;
      coeffs[1] = -(-c * n_mat(2, 0) + a * n_mat(3, 0)) / delta * c1;
    }
  }

  return coeffs;
}

/**
 * @brief Solve for the two unknown half-space coefficients of a model with a
 *        fluid block from the free-surface condition.
 *
 * Faithfully mirrors fk_core.cpp lines 303-328. A modified 2×2 system
 * @f$ N_1 @f$ is assembled from @f$ N @f$ and the fluid propagator rows, then
 * solved by the same Cramer formula as the elastic-only overload.
 *
 * If the determinant @f$|\delta| \leq \epsilon@f$ (singular configuration) both
 * coefficients are set to zero.
 *
 * @param n_mat     Propagated elastic matrix @f$ N @f$ (4×4 complex).
 * @param qmat_full Full acoustic propagator (leading 2×2 block used).
 * @param type      Incident wave type (P or SV).
 * @param c1        Incoming SV amplitude (from @c incident_amplitude).
 * @param c3        Incoming P amplitude  (from @c incident_amplitude).
 * @return Array of two complex coefficients {coeff0, coeff1}.
 */
inline std::array<std::complex<double>, 2>
halfspace_coefficients(const fk_injection::ComplexMatrix<4> &n_mat,
                       const fk_injection::ComplexMatrix<4> &qmat_full,
                       fk_injection::incident_wave_type type,
                       std::complex<double> c1, std::complex<double> c3) {
  using Cplx = std::complex<double>;
  const Cplx zero(0.0, 0.0);

  std::array<Cplx, 2> coeffs;
  coeffs[0] = zero;
  coeffs[1] = zero;

  const bool is_p = (type == fk_injection::incident_wave_type::p);

  const Cplx n1_00 = n_mat(2, 1);
  const Cplx n1_01 = n_mat(2, 3);
  const Cplx n1_10 =
      qmat_full(1, 0) * n_mat(1, 1) - qmat_full(1, 1) * n_mat(3, 1);
  const Cplx n1_11 =
      qmat_full(1, 0) * n_mat(1, 3) - qmat_full(1, 1) * n_mat(3, 3);

  Cplx MM;
  if (is_p) {
    MM = qmat_full(1, 0) * n_mat(1, 2) - qmat_full(1, 1) * n_mat(3, 2);
  } else {
    MM = qmat_full(1, 0) * n_mat(1, 0) - qmat_full(1, 1) * n_mat(3, 0);
  }

  const Cplx a = n1_00;
  const Cplx b = n1_01;
  const Cplx c = n1_10;
  const Cplx d = n1_11;
  const Cplx delta = a * d - b * c;

  if (std::abs(delta) > tinyval) {
    if (is_p) {
      coeffs[0] = -(d * n_mat(2, 2) - b * MM) / delta * c3;
      coeffs[1] = -(-c * n_mat(2, 2) + a * MM) / delta * c3;
    } else {
      coeffs[0] = -(d * n_mat(2, 0) - b * MM) / delta * c1;
      coeffs[1] = -(-c * n_mat(2, 0) + a * MM) / delta * c1;
    }
  }

  return coeffs;
}

/**
 * @brief Assemble the 4-component bottom state vector for a given frequency.
 *
 * Faithfully mirrors fk_core.cpp lines 414–425.  The bottom vector encodes the
 * incident + reflected mode amplitudes at the base of the model:
 *
 * - **P incidence** (index layout: SV-up | SV-down | P-up | P-down):
 *   @f$ b_1 = \text{coeff}_0,\; b_2 = C_3,\; b_3 = \text{coeff}_1,\; b_0 = 0
 * @f$
 * - **SV incidence**:
 *   @f$ b_0 = C_1,\; b_2 = \text{coeff}_0,\; b_3 = \text{coeff}_1,\; b_1 = 0
 * @f$
 *
 * @param type   Incident wave type (P or SV).
 * @param c1     Incoming SV amplitude (from @c incident_amplitude).
 * @param c3     Incoming P  amplitude (from @c incident_amplitude).
 * @param coeffs Two half-space coefficients from @c halfspace_coefficients.
 * @return 4-component complex bottom vector.
 */
inline fk_injection::ComplexVector<4>
build_bottom_vector(fk_injection::incident_wave_type type,
                    std::complex<double> c1, std::complex<double> c3,
                    const std::array<std::complex<double>, 2> &coeffs) {
  fk_injection::ComplexVector<4> bot;
  bot[0] = std::complex<double>(0.0, 0.0);
  bot[1] = std::complex<double>(0.0, 0.0);
  bot[2] = std::complex<double>(0.0, 0.0);
  bot[3] = std::complex<double>(0.0, 0.0);

  if (type == fk_injection::incident_wave_type::p) {
    // P incidence: incident P-up at column 2 (C_3); reflected SV-down at
    // column 1 and P-down at column 3.
    bot[1] = coeffs[0];
    bot[2] = c3;
    bot[3] = coeffs[1];
  } else {
    // SV incidence: incident SV-up at column 0 (C_1); reflected SV-down at
    // column 1 and P-down at column 3.
    //
    // NOTE (deviation from the reference, intentional): SPECFEM3D's
    // couple_with_injection.f90 (bot_vec(3)=coeff(1), i.e. 0-based index 2) and
    // the Westervelt C++ port place the reflected SV-down amplitude at column 2
    // (the incident P-up slot), leaving column 1 zero. Because the shared 2x2
    // solve uses columns 1 and 3, that placement violates the free-surface
    // zero-traction condition for SV. We place coeffs[0] at column 1, which is
    // physically correct (verified by the free-surface traction unit test). See
    // docs/specfem3d-sv-fk-bug-report.md.
    bot[0] = c1;
    bot[1] = coeffs[0];
    bot[3] = coeffs[1];
  }

  return bot;
}

} // namespace fk_injection::impl
