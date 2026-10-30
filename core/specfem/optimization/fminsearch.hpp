#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>

namespace specfem {
namespace optimization {

/**
 * @brief Tag for the SPECFEM3D_GLOBE port of MATLAB's fminsearch.
 *
 * Unlike NelderMeadSimplex, this algorithm reproduces the Fortran routine
 * (src/meshfem3D/model_attenuation.f90) step for step: double precision,
 * globe's trial-point expressions, its heap sort, and its iteration and
 * function-evaluation budgets. It is used by the 3D attenuation Q fit.
 *
 * @see optimize(Fminsearch, Func &&objective, FminsearchOptions<N> options)
 */
struct Fminsearch {};

/**
 * @brief Configuration for Fminsearch.
 *
 * Defaults are globe's: both budgets are \f$200N\f$ and both tolerances are
 * the single-precision literal 1.0e-4 promoted to double, as in the Fortran.
 *
 * @tparam N Number of optimization variables
 */
template <int N> struct FminsearchOptions {
  /// Initial guess
  Kokkos::View<double[N], Kokkos::LayoutRight, Kokkos::HostSpace> x0;
  int max_iterations = -1;           ///< Iteration budget (-1 = 200*N)
  int max_function_evaluations = -1; ///< Evaluation budget (-1 = 200*N)
  double tol_f = 1.0e-4f;            ///< Tolerance on function value spread
  double tol_x = 1.0e-4f;            ///< Tolerance on simplex size
};

/**
 * @brief Ascending heap sort returning the permutation, as globe's
 *        heap_sort_local (src/shared/heap_sort.f90).
 *
 * Uses Fortran's 1-based indexing so tie handling matches exactly.
 *
 * @tparam M Number of values
 * @param x Values, sorted in place (entries 1..M are used)
 * @param y Output permutation (entries 1..M are used)
 */
template <int M>
void fminsearch_heap_sort(std::array<double, M + 1> &x,
                          std::array<int, M + 1> &y) {
  for (int i = 1; i <= M; ++i)
    y[i] = i;
  if (M < 2)
    return;
  const auto siftdown = [&](const int start, const int bottom) {
    int i = start;
    const double xtmp = x[i];
    const int ytmp = y[i];
    int j = 2 * i;
    while (j <= bottom) {
      if (j < bottom && x[j] <= x[j + 1])
        ++j;
      if (x[j] < xtmp)
        break;
      x[i] = x[j];
      y[i] = y[j];
      i = j;
      j = 2 * i;
    }
    x[i] = xtmp;
    y[i] = ytmp;
  };
  for (int i = M / 2; i >= 1; --i)
    siftdown(i, M);
  for (int i = M; i >= 2; --i) {
    std::swap(x[1], x[i]);
    std::swap(y[1], y[i]);
    siftdown(1, i - 1);
  }
}

/**
 * @brief Minimize with SPECFEM3D_GLOBE's fminsearch.
 *
 * Builds the initial simplex by scaling each nonzero coordinate by
 * \f$1+0.05\f$ (zero coordinates become 0.00025), then applies reflection,
 * expansion, outside/inside contraction and shrink with globe's
 * coefficients \f$\rho=1, \chi=2, \psi=1/2, \sigma=1/2\f$ and its exact
 * trial-point expressions, e.g. \f$x_r=(1+\rho)\bar{x}-\rho v_{N+1}\f$.
 * Convergence is checked at the top of each iteration and requires both the
 * simplex size and the function-value spread to be at most the tolerances.
 * The initial simplex counts as iteration one and costs \f$N+1\f$
 * evaluations; a started iteration always completes.
 *
 * @tparam N Problem dimension
 * @tparam Func Callable with signature double(View<double[N]>)
 * @param tag Algorithm selector (Fminsearch{})
 * @param objective Function to minimize
 * @param options Initial guess, budgets and tolerances
 * @return OptimizationResult with the best vertex; converged is false when a
 * budget ran out first
 */
template <int N, typename Func>
OptimizationResult<N, double> optimize(Fminsearch tag, Func &&objective,
                                       FminsearchOptions<N> options) {
  (void)tag;
  using Point = std::array<double, N>;
  using PointView =
      Kokkos::View<double[N], Kokkos::LayoutRight, Kokkos::HostSpace>;

  const int max_iterations =
      options.max_iterations > 0 ? options.max_iterations : 200 * N;
  const int max_evaluations = options.max_function_evaluations > 0
                                  ? options.max_function_evaluations
                                  : 200 * N;
  constexpr double rho = 1.0, chi = 2.0, psi = 0.5, sigma = 0.5;
  // Globe assigns default-real literals to these double variables.
  constexpr double usual_delta = 0.05f;
  constexpr double zero_term_delta = 0.00025f;

  PointView point("fminsearch_point");
  int func_evals = 0;
  const auto evaluate = [&](const Point &x) {
    for (int i = 0; i < N; ++i)
      point(i) = x[i];
    ++func_evals;
    return static_cast<double>(objective(point));
  };

  // Vertices v[1..N+1] and values fv[1..N+1], as in the Fortran.
  std::array<Point, N + 2> v{};
  std::array<double, N + 2> fv{};
  std::array<int, N + 2> place{};
  const auto sort_simplex = [&]() {
    fminsearch_heap_sort<N + 1>(fv, place);
    std::array<Point, N + 2> sorted{};
    for (int i = 1; i <= N + 1; ++i)
      sorted[i] = v[place[i]];
    v = sorted;
  };

  Point xin;
  for (int i = 0; i < N; ++i)
    xin[i] = options.x0(i);
  v[1] = xin;
  fv[1] = evaluate(xin);
  for (int j = 0; j < N; ++j) {
    Point y = xin;
    y[j] = y[j] != 0.0 ? (1.0 + usual_delta) * y[j] : zero_term_delta;
    v[j + 2] = y;
    fv[j + 2] = evaluate(y);
  }
  sort_simplex();

  int itercount = 1;
  bool converged = false;
  while (func_evals < max_evaluations && itercount < max_iterations) {
    double size = 0.0, spread = 0.0;
    for (int i = 0; i < N; ++i)
      for (int j = 2; j <= N + 1; ++j)
        size = std::max(size, std::abs(v[j][i] - v[1][i]));
    for (int i = 2; i <= N + 1; ++i)
      spread = std::max(spread, std::abs(fv[1] - fv[i]));
    if (size <= options.tol_x && spread <= options.tol_f) {
      converged = true;
      break;
    }

    Point xbar{};
    for (int i = 0; i < N; ++i) {
      for (int j = 1; j <= N; ++j)
        xbar[i] += v[j][i];
      xbar[i] /= N * 1.0;
    }
    const auto combine = [&](const double a, const double b) {
      Point x;
      for (int i = 0; i < N; ++i)
        x[i] = a * xbar[i] + b * v[N + 1][i];
      return x;
    };

    const Point xr = combine(1 + rho, -rho);
    const double fxr = evaluate(xr);
    bool shrink = false;
    if (fxr < fv[1]) {
      const Point xe = combine(1 + rho * chi, -rho * chi);
      const double fxe = evaluate(xe);
      if (fxe < fxr) {
        v[N + 1] = xe;
        fv[N + 1] = fxe;
      } else {
        v[N + 1] = xr;
        fv[N + 1] = fxr;
      }
    } else if (fxr < fv[N]) {
      v[N + 1] = xr;
      fv[N + 1] = fxr;
    } else if (fxr < fv[N + 1]) {
      const Point xc = combine(1 + psi * rho, -psi * rho);
      const double fxc = evaluate(xc);
      if (fxc <= fxr) {
        v[N + 1] = xc;
        fv[N + 1] = fxc;
      } else {
        shrink = true;
      }
    } else {
      const Point xcc = combine(1 - psi, psi);
      const double fxcc = evaluate(xcc);
      if (fxcc < fv[N + 1]) {
        v[N + 1] = xcc;
        fv[N + 1] = fxcc;
      } else {
        shrink = true;
      }
    }
    if (shrink) {
      for (int j = 2; j <= N + 1; ++j) {
        for (int i = 0; i < N; ++i)
          v[j][i] = v[1][i] + sigma * (v[j][i] - v[1][i]);
        fv[j] = evaluate(v[j]);
      }
    }
    sort_simplex();
    ++itercount;
  }

  OptimizationResult<N, double> result;
  result.x = PointView("result_x");
  for (int i = 0; i < N; ++i)
    result.x(i) = v[1][i];
  result.min_value = fv[1];
  result.iterations = itercount;
  result.function_evaluations = func_evals;
  result.converged = converged;
  return result;
}

} // namespace optimization
} // namespace specfem
