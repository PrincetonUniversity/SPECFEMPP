#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace specfem {
namespace utilities {
namespace cubic_bspline {

// Device-callable (raw-pointer, KOKKOS_INLINE_FUNCTION) cubic B-spline
// prefilter and evaluation, used per point inside Kokkos kernels. Host-only
// alternatives such as Boost.Math splines cannot run in device code, and Boost
// is an optional dependency.

/**
 * @brief Evaluate the piecewise cubic B-spline basis function @f$ B_3(x) @f$.
 *
 * The support is @f$ |x| < 2 @f$:
 * @f[
 *   B_3(x) = \begin{cases}
 *     \tfrac{2}{3} - |x|^2 + \tfrac{|x|^3}{2} & |x| < 1 \\
 *     \tfrac{(2-|x|)^3}{6}                      & 1 \le |x| < 2 \\
 *     0                                          & |x| \ge 2
 *   \end{cases}
 * @f]
 *
 * @param x Argument (continuous sample-unit distance from the knot).
 * @return Basis value @f$ B_3(x) @f$.
 */
KOKKOS_INLINE_FUNCTION double cubic_bspline_basis(double x) {
  const double ax = Kokkos::fabs(x);
  if (ax >= 2.0) {
    return 0.0;
  }
  if (ax < 1.0) {
    return (2.0 / 3.0) - ax * ax + 0.5 * ax * ax * ax;
  }
  // 1 <= ax < 2
  const double t = 2.0 - ax;
  return (t * t * t) / 6.0;
}

/**
 * @brief Compute cubic B-spline coefficients from a uniformly sampled signal.
 *
 * Implements the causal/anti-causal IIR filter from the Fortran reference
 * @c compute_spline_coef_to_store using the first-order recursive pole
 * @f$ z_1 = \sqrt{3} - 2 @f$.
 *
 * Both @p signal and @p coefficients must point to arrays of length @p n.
 * The input @p signal is not modified; the output is written to @p
 * coefficients.
 *
 * @param signal       Input signal samples (length @p n).
 * @param n            Number of samples.
 * @param coefficients Output B-spline coefficients (length @p n).
 */
KOKKOS_INLINE_FUNCTION void compute_coefficients(const double *signal, int n,
                                                 double *coefficients) {
  constexpr double error = 1.0e-24;
  const double z1 = Kokkos::sqrt(3.0) - 2.0; // ≈ -0.2679

  // Pre-scale: each sample multiplied by (1 - z1) * (1 - 1/z1).
  const double scale = (1.0 - z1) * (1.0 - 1.0 / z1);
  for (int i = 0; i < n; ++i) {
    coefficients[i] = signal[i] * scale;
  }

  // Initial condition for the causal pass.
  // n_init = ceil( log(error) / log(|z1|) ), clamped to n.
  int n_init = static_cast<int>(
      Kokkos::ceil(Kokkos::log(error) / Kokkos::log(Kokkos::fabs(z1))));
  if (n_init > n)
    n_init = n;

  // Faithful port of the reference: sumc starts at coefficients[0], and the
  // loop accumulates zn * coefficients[i-1] with zn starting at z1 over
  // i = 1..n_init (so the i=1 term re-adds z1 * coefficients[0]).
  double sumc = coefficients[0];
  double zn = z1;
  for (int i = 1; i <= n_init; ++i) {
    sumc += zn * coefficients[i - 1];
    zn *= z1;
  }
  coefficients[0] = sumc;

  // Causal pass (forward): coefficients[i] += z1 * coefficients[i-1]
  // Reference: for i=2..n: coeff[i-1] += z1*coeff[i-2]  (1-based, i starts at
  // 2)
  for (int i = 1; i < n; ++i) {
    coefficients[i] += z1 * coefficients[i - 1];
  }

  // Anti-causal boundary condition: coeff[n-1] *= z1 / (z1 - 1)
  coefficients[n - 1] = (z1 / (z1 - 1.0)) * coefficients[n - 1];

  // Anti-causal pass (backward): coefficients[i] = z1*(coefficients[i+1] -
  // coefficients[i]) Reference: for i=n-1 downto 1: coeff[i-1] =
  // z1*(coeff[i]-coeff[i-1])
  //   (1-based: i=n-1..1 → 0-based: i=(n-2)..0)
  for (int i = n - 2; i >= 0; --i) {
    coefficients[i] = z1 * (coefficients[i + 1] - coefficients[i]);
  }
}

/**
 * @brief Reconstruct a signal value at a continuous abscissa from B-spline
 * coefficients.
 *
 * Evaluates the cubic B-spline interpolant at position @p t (in sample units):
 * @f[
 *   s(t) = \sum_{j} c_j \, B_3(t - j)
 * @f]
 * where the sum runs over the (at most 4) neighbouring integer knots and
 * indices are clamped to @f$ [0, n-1] @f$.
 *
 * @param coefficients B-spline coefficients (length @p n), produced by
 *                     @c compute_coefficients.
 * @param n            Number of coefficients / samples.
 * @param t            Continuous evaluation abscissa in sample units.
 * @return Interpolated value.
 */
KOKKOS_INLINE_FUNCTION double evaluate(const double *coefficients, int n,
                                       double t) {
  // Nearest integer knot below t.
  const int j0 = static_cast<int>(Kokkos::floor(t));

  double result = 0.0;
  // B_3 has support (-2, 2), so knots j0-1 .. j0+2 contribute.
  for (int dj = -1; dj <= 2; ++dj) {
    const int j = j0 + dj;
    // Clamp index to valid range.
    const int jc = (j < 0) ? 0 : ((j >= n) ? n - 1 : j);
    result +=
        coefficients[jc] * cubic_bspline_basis(t - static_cast<double>(j));
  }
  return result;
}

} // namespace cubic_bspline
} // namespace utilities
} // namespace specfem
