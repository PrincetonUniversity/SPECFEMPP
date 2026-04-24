#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace source_time_functions {
namespace impl {

KOKKOS_INLINE_FUNCTION
type_real gaussian(const type_real t, const type_real f0) {
  // Gaussian wavelet i.e. second integral of a Ricker wavelet
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real gaussian = -1.0 * Kokkos::exp(-a * t * t) / (2.0 * a);
  return gaussian;
}

/**
 * @brief Gaussian wavelet defined using half duration hdur
 *
 * @param t time
 * @param hdur half duration
 * @return type_real
 *
 * @note The heaviside function is approximated using the error function. It is
 *       assumed that the half duration is already adjusted to mimic a
 *       triangular source time function. Meaning, hdur = hdur /
 *       SOURCE_DECAY_MIMIC_TRIANGLE
 *
 */
KOKKOS_INLINE_FUNCTION
type_real gaussian_hdur(const type_real t, const type_real hdur) {
  // Gaussian wavelet defined using half duration hdur
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;

  type_real a = 1.0 / (hdur * hdur);
  type_real gaussian = Kokkos::exp(-a * t * t) / (Kokkos::sqrt(pi) * hdur);
  return gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d1gaussian(const type_real t, const type_real f0) {
  // First derivative of Gaussian wavelet
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real d1gaussian = t * Kokkos::exp(-a * t * t);
  return d1gaussian;
}

/**
 * @brief First derivative of Gaussian wavelet defined using half duration hdur
 *
 * Likely unused, but here for completeness for use_trick_for_better_pressure
 * option
 *
 * @param t
 * @param hdur
 * @return type_real
 *
 * @note The heaviside function is approximated using the error function. It is
 *       assumed that the half duration is already adjusted to mimic a
 *       triangular source time function. Meaning, hdur = hdur /
 *       SOURCE_DECAY_MIMIC_TRIANGLE
 */
KOKKOS_INLINE_FUNCTION
type_real d1gaussian_hdur(const type_real t, const type_real hdur) {
  // First derivative of Gaussian wavelet defined using half duration hdur
  type_real a = 1.0 / (hdur * hdur);
  type_real d1gaussian_hdur = -2.0 * a * t * gaussian_hdur(t, hdur);
  return d1gaussian_hdur;
}

KOKKOS_INLINE_FUNCTION
type_real d2gaussian(const type_real t, const type_real f0) {
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real d2gaussian = (1.0 - 2.0 * a * t * t) * Kokkos::exp(-a * t * t);
  return d2gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d2gaussian_hdur(const type_real t, const type_real hdur) {
  // Second derivative of Gaussian wavelet defined using half duration hdur
  type_real a = 1.0 / (hdur * hdur);
  type_real d2gaussian_hdur =
      2.0 * a * (-1.0 + 2.0 * a * t * t) * gaussian_hdur(t, hdur);
  return d2gaussian_hdur;
}

KOKKOS_INLINE_FUNCTION
type_real d3gaussian(const type_real t, const type_real f0) {
  // Third derivative of Gaussian wavelet
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real d3gaussian =
      -2.0 * a * t * (3.0 - 2.0 * a * t * t) * Kokkos::exp(-a * t * t);
  return d3gaussian;
}

KOKKOS_INLINE_FUNCTION
type_real d4gaussian(const type_real t, const type_real f0) {
  constexpr auto pi = Kokkos::numbers::pi_v<type_real>;
  type_real a = pi * pi * f0 * f0;
  type_real d4gaussian =
      -2.0 * a * (3.0 - 12.0 * a * t * t + 4.0 * a * a * t * t * t * t) *
      Kokkos::exp(-a * t * t);
  return d4gaussian;
}

/**
 * @brief heaviside function defined using half duration hdur
 *
 * @param t time
 * @param hdur half duration
 * @return heaviside function value at time t
 *
 * @note The heaviside function is approximated using the error function. It is
 *       assumed that the half duration is already adjusted to mimic a
 *       triangular source time function. Meaning, hdur = hdur /
 *       SOURCE_DECAY_MIMIC_TRIANGLE
 *
 * @note A nonzero time to start the simulation with would lead to more
 *       high-frequency noise due to the (spatial) discretization of
 *       the point source on the mesh
 */
KOKKOS_INLINE_FUNCTION
type_real heaviside(const type_real t, const type_real hdur) {
  // Heaviside function approximated using error function

  type_real heaviside = 0.5 * (1.0 + std::erf(t / hdur));
  return heaviside;
}

} // namespace impl
} // namespace source_time_functions
} // namespace specfem
