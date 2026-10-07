#pragma once

#include "specfem/injection/fk/impl/fk_math_impl.hpp"
#include "specfem/utilities/complex_matrix.hpp"

namespace specfem {
namespace injection {
namespace fk_impl {

/**
 * @brief Compute the elastic P-SV layer propagator matrix (Tong 2014 A8).
 *
 * Returns the unscaled propagator @f$ \bar{P} @f$ as a 4×4 complex matrix.
 * The driver is responsible for applying the @c elastic_gamma0 prefactor after
 * each layer product.
 *
 * The 4-component state vector is @f$(u_x, u_z, \sigma_{xz}, \sigma_{zz})@f$.
 *
 * @param layer           Elastic layer whose material properties are used.
 * @param omega           Complex angular frequency (rad/s); may include
 *                        imaginary damping @f$-i\sigma_0@f$.
 * @param thickness       Layer thickness H (m); use 0.0 for the semigroup test.
 * @param ray_parameter   Horizontal slowness p (s/m).
 * @return 4×4 unscaled propagator matrix.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexMatrix<4>
layer_propagator(const specfem::injection::fk::ElasticIsotropicLayer &layer,
                 Kokkos::complex<double> omega, type_real thickness,
                 type_real ray_parameter) {
  specfem::utilities::ComplexMatrix<4> P;

  // -------------------------------------------------------------------------
  // Tong (2014) appendix (A8) — P-SV propagator matrix
  // -------------------------------------------------------------------------
  const auto slowness = vertical_slowness(layer, ray_parameter);
  const Kokkos::complex<double> eta_alpha = slowness[0];
  const Kokkos::complex<double> eta_beta = slowness[1];

  const double g1 = elastic_gamma1(layer, ray_parameter);
  const double p = static_cast<double>(ray_parameter);
  const double H = static_cast<double>(thickness);
  const double rho = static_cast<double>(layer.density);
  const double vs = static_cast<double>(layer.s_velocity);
  const double mul = rho * vs * vs;

  // Build cosh/sinh from exp to stay device-callable.
  const Kokkos::complex<double> c1 = omega * eta_alpha * H;
  const Kokkos::complex<double> exp_p1 = Kokkos::exp(c1);
  const Kokkos::complex<double> exp_m1 = Kokkos::exp(-c1);
  const Kokkos::complex<double> ca = (exp_p1 + exp_m1) * 0.5;
  const Kokkos::complex<double> sa = (exp_p1 - exp_m1) * 0.5;
  const Kokkos::complex<double> xa = eta_alpha * sa / p;
  const Kokkos::complex<double> ya = p * sa / eta_alpha;

  const Kokkos::complex<double> c2 = omega * eta_beta * H;
  const Kokkos::complex<double> exp_p2 = Kokkos::exp(c2);
  const Kokkos::complex<double> exp_m2 = Kokkos::exp(-c2);
  const Kokkos::complex<double> cb = (exp_p2 + exp_m2) * 0.5;
  const Kokkos::complex<double> sb = (exp_p2 - exp_m2) * 0.5;
  const Kokkos::complex<double> xb = eta_beta * sb / p;
  const Kokkos::complex<double> yb = p * sb / eta_beta;

  const Kokkos::complex<double> g1c(g1, 0.0);
  const Kokkos::complex<double> g1_sq = g1c * g1c;
  const Kokkos::complex<double> two_mul(2.0 * mul, 0.0);

  P(0, 0) = ca - g1c * cb;
  P(0, 1) = xb - g1c * ya;
  P(0, 2) = (ya - xb) / two_mul;
  P(0, 3) = (cb - ca) / two_mul;

  P(1, 0) = xa - g1c * yb;
  P(1, 1) = cb - g1c * ca;
  P(1, 2) = (ca - cb) / two_mul;
  P(1, 3) = (yb - xa) / two_mul;

  P(2, 0) = two_mul * (xa - g1_sq * yb);
  P(2, 1) = two_mul * g1c * (cb - ca);
  P(2, 2) = ca - g1c * cb;
  P(2, 3) = g1c * yb - xa;

  P(3, 0) = two_mul * g1c * (ca - cb);
  P(3, 1) = two_mul * (xb - g1_sq * ya);
  P(3, 2) = g1c * ya - xb;
  P(3, 3) = cb - g1c * ca;

  return P;
}

/**
 * @brief Compute the acoustic layer propagator matrix.
 *
 * Returns the 2×2 cosh/sinh propagator for the @f$(u_z, P/k)@f$ state in
 * the leading 2×2 block of a 4×4 matrix.  The remaining entries are zero.
 *
 * @param layer           Acoustic layer whose material properties are used.
 * @param omega           Complex angular frequency (rad/s).
 * @param thickness       Layer thickness H (m).
 * @param ray_parameter   Horizontal slowness p (s/m).
 * @return 4×4 matrix with the acoustic propagator in the leading 2×2 block.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexMatrix<4>
layer_propagator(const specfem::injection::fk::AcousticLayer &layer,
                 Kokkos::complex<double> omega, type_real thickness,
                 type_real ray_parameter) {
  specfem::utilities::ComplexMatrix<4> P;

  // -------------------------------------------------------------------------
  // Acoustic (fluid) 2×2 cosh/sinh propagator.
  // State vector: (u_z, P/k) in the leading two components.
  // -------------------------------------------------------------------------
  const Kokkos::complex<double> eta_alpha =
      vertical_slowness(layer, ray_parameter);

  const double p = static_cast<double>(ray_parameter);
  const double H = static_cast<double>(thickness);
  const double rho = static_cast<double>(layer.density);

  const Kokkos::complex<double> c1 = omega * eta_alpha * H;
  const Kokkos::complex<double> exp_p1 = Kokkos::exp(c1);
  const Kokkos::complex<double> exp_m1 = Kokkos::exp(-c1);
  const Kokkos::complex<double> ca = (exp_p1 + exp_m1) * 0.5;
  const Kokkos::complex<double> sa = (exp_p1 - exp_m1) * 0.5;

  P(0, 0) = ca;
  P(1, 1) = ca;
  P(0, 1) = -sa * eta_alpha * p / rho;
  P(1, 0) = sa * rho / (p * eta_alpha);
  // Entries (0,2), (0,3), (1,2), (1,3) and the 3rd/4th rows stay zero.

  return P;
}

/**
 * @brief Compute the elastic half-space eigenmatrix E (Tong 2014 A10).
 *
 * Assembles the columns of the modal decomposition matrix for an elastic
 * half-space (or the deepest elastic layer).  The columns correspond to
 * up-going SV, down-going SV, up-going P, and down-going P wave modes.
 *
 * @param halfspace      Elastic layer record for the half-space (bottom layer).
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @return 4×4 eigenmatrix E.
 */
KOKKOS_INLINE_FUNCTION specfem::utilities::ComplexMatrix<4>
elastic_halfspace_eigenmatrix(
    const specfem::injection::fk::ElasticIsotropicLayer &halfspace,
    type_real ray_parameter) {
  const auto slowness = vertical_slowness(halfspace, ray_parameter);
  const Kokkos::complex<double> eta_alpha = slowness[0];
  const Kokkos::complex<double> eta_beta = slowness[1];

  const double g1 = elastic_gamma1(halfspace, ray_parameter);
  const double p = static_cast<double>(ray_parameter);
  const double rho = static_cast<double>(halfspace.density);
  const double vs = static_cast<double>(halfspace.s_velocity);
  const double two_mul = 2.0 * rho * vs * vs;

  const Kokkos::complex<double> two_mul_c(two_mul, 0.0);
  const Kokkos::complex<double> g1c(g1, 0.0);
  const Kokkos::complex<double> one(1.0, 0.0);

  specfem::utilities::ComplexMatrix<4> E;

  E(0, 0) = eta_beta / p;
  E(0, 1) = -eta_beta / p;
  E(0, 2) = one;
  E(0, 3) = one;

  E(1, 0) = one;
  E(1, 1) = one;
  E(1, 2) = eta_alpha / p;
  E(1, 3) = -eta_alpha / p;

  E(2, 0) = two_mul_c * g1c;
  E(2, 1) = two_mul_c * g1c;
  E(2, 2) = two_mul_c * eta_alpha / p;
  E(2, 3) = -two_mul_c * eta_alpha / p;

  E(3, 0) = two_mul_c * eta_beta / p;
  E(3, 1) = -two_mul_c * eta_beta / p;
  E(3, 2) = two_mul_c * g1c;
  E(3, 3) = two_mul_c * g1c;

  return E;
}

} // namespace fk_impl
} // namespace injection
} // namespace specfem
