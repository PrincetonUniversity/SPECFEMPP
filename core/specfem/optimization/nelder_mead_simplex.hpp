#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace specfem {
namespace optimization {

// ============================================================================
// Nelder-Mead Tag
// ============================================================================

/**
 * @brief Tag for Nelder-Mead simplex algorithm
 *
 * Derivative-free method suitable for non-smooth objective functions.
 * Converges slowly but robustly for problems with few variables (N < 10).
 *
 * @see optimize(NelderMeadSimplex, Func &&objective, NelderMeadOptions<N,
 * ValueType> options)
 */
struct NelderMeadSimplex {};

// ============================================================================
// Nelder-Mead Options
// ============================================================================

/**
 * @brief Configuration for Nelder-Mead simplex algorithm
 *
 * Defaults follow fminsearch (Lagarias et al. 1998), the variant used by the
 * SPECFEM Fortran codes for the attenuation fit.
 *
 * @tparam N Number of optimization variables
 * @tparam ValueType Scalar type (defaults to type_real)
 */
template <int N, typename ValueType = type_real> struct NelderMeadOptions {
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace>
      x0;                  ///< Initial guess
  int max_iterations = -1; ///< Iteration budget (nonpositive = 200*N)
  /// Evaluation budget (nonpositive = 200*N). A started iteration always
  /// completes and may exceed this budget.
  int max_function_evaluations = -1;
  ValueType tol_f = 1.0e-4; ///< Tolerance on function value spread
  ValueType tol_x = 1.0e-4; ///< Tolerance on simplex diameter

  // Simplex transformation coefficients (standard values recommended)
  ValueType reflection = 1.0;  ///< Reflection coefficient
  ValueType expansion = 2.0;   ///< Expansion coefficient
  ValueType contraction = 0.5; ///< Contraction coefficient
  ValueType shrink = 0.5;      ///< Shrink coefficient

  // Initial simplex
  ValueType initial_step = 0.05; ///< Relative step for nonzero coordinates
  ValueType zero_step = 0.00025; ///< Absolute step for zero coordinates
};

} // namespace optimization

namespace optimization_impl {

/**
 * @brief Ascending heap sort returning the permutation.
 *
 * Gives the simplex a deterministic order for tied function values, and the
 * same order as heap_sort_local (src/shared/heap_sort.f90) of the SPECFEM
 * Fortran codes, so fits reproduce theirs. Uses 1-based indexing to match
 * that tie handling exactly.
 *
 * @tparam M Number of values
 * @param x Values, sorted in place (entries 1..M are used)
 * @param y Output permutation (entries 1..M are used)
 */
template <int M>
void heap_sort(std::array<double, M + 1> &x, std::array<int, M + 1> &y) {
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

} // namespace optimization_impl

namespace optimization {

// ============================================================================
// Nelder-Mead optimize specialization
// ============================================================================

/**
 * @brief Nelder-Mead downhill simplex minimization
 *
 * Derivative-free optimization using @f$N+1@f$ simplex vertices in
 * @f$N@f$-dimensional space. Iteratively applies reflection, expansion,
 * contraction, and shrink operations to locate a local minimum. Follows
 * fminsearch: the initial simplex counts as iteration one, trial points are
 * @f$(1+t)\bar{x}-t\,x_{N}@f$, and the run stops once
 * @f$\max_i|f_0-f_i|\le@f$ tol_f and @f$\max_{i,j}|x_{ij}-x_{0j}|\le@f$ tol_x,
 * or either budget is spent.
 *
 * @tparam N Problem dimension
 * @tparam ValueType Scalar type
 * @tparam Func Callable with signature: ValueType(View<ValueType[N]>)
 *
 * @param tag Algorithm selector (NelderMeadSimplex{})
 * @param objective Function to minimize
 * @param options Configuration with initial guess and tolerances
 * @return OptimizationResult with solution, value, iterations, evaluations and
 * convergence flag
 *
 * @code
 * // Minimize (x-1)^2 + (y-2)^2
 * auto f = [](auto x) { return (x(0)-1)*(x(0)-1) + (x(1)-2)*(x(1)-2); };
 * Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
 * x0(0) = 0.0; x0(1) = 0.0;
 * NelderMeadOptions<2> opts{x0};
 * auto result = optimize(NelderMeadSimplex{}, f, opts);
 * // result.x(0) ≈ 1.0, result.x(1) ≈ 2.0
 * @endcode
 */
template <int N, typename ValueType, typename Func>
OptimizationResult<N, ValueType>
optimize(NelderMeadSimplex, Func &&objective,
         NelderMeadOptions<N, ValueType> options) {
  using point_type =
      Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace>;

  int function_evaluations = 0;
  const auto evaluate = [&](const point_type &point) {
    ++function_evaluations;
    return static_cast<ValueType>(objective(point));
  };
  const int max_evaluations = options.max_function_evaluations > 0
                                  ? options.max_function_evaluations
                                  : 200 * N;
  const int max_iterations =
      options.max_iterations > 0 ? options.max_iterations : 200 * N;
  const ValueType rho = options.reflection;
  const ValueType chi = options.expansion;
  const ValueType psi = options.contraction;
  const ValueType sigma = options.shrink;

  // Simplex has N+1 vertices, each is an N-dimensional point
  Kokkos::View<ValueType[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      simplex("simplex");
  Kokkos::View<ValueType[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace> fvals(
      "fvals");
  point_type point("point");

  // Vertex 0 is x0; vertex i perturbs coordinate i-1.
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j < N; ++j) {
      simplex(i, j) = options.x0(j);
    }
    if (i > 0) {
      const ValueType x = options.x0(i - 1);
      simplex(i, i - 1) =
          x != 0 ? (1 + options.initial_step) * x : options.zero_step;
    }
    for (int j = 0; j < N; ++j) {
      point(j) = simplex(i, j);
    }
    fvals(i) = evaluate(point);
  }

  // Sort vertices by ascending function value.
  Kokkos::View<ValueType[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      sorted_simplex("sorted_simplex");
  Kokkos::View<ValueType[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace>
      sorted_fvals("sorted_fvals");
  const auto sort_simplex = [&]() {
    std::array<double, N + 2> values{};
    std::array<int, N + 2> places{};
    for (int i = 0; i <= N; ++i)
      values[i + 1] = fvals(i);
    specfem::optimization_impl::heap_sort<N + 1>(values, places);
    for (int i = 0; i <= N; ++i) {
      const int from = places[i + 1] - 1;
      sorted_fvals(i) = fvals(from);
      for (int j = 0; j < N; ++j)
        sorted_simplex(i, j) = simplex(from, j);
    }
    for (int i = 0; i <= N; ++i) {
      fvals(i) = sorted_fvals(i);
      for (int j = 0; j < N; ++j)
        simplex(i, j) = sorted_simplex(i, j);
    }
  };
  sort_simplex();

  // Trial point (1 + t) * centroid - t * worst.
  point_type centroid("centroid");
  const auto trial = [&](const ValueType t, point_type &x) {
    for (int j = 0; j < N; ++j)
      x(j) = (1 + t) * centroid(j) - t * simplex(N, j);
  };
  const auto replace_worst = [&](const point_type &x, const ValueType f) {
    for (int j = 0; j < N; ++j)
      simplex(N, j) = x(j);
    fvals(N) = f;
  };
  point_type xr("xr"); // reflection
  point_type xe("xe"); // expansion
  point_type xc("xc"); // contraction

  int iter = 1; // the initial simplex
  bool converged = false;

  while (iter < max_iterations && function_evaluations < max_evaluations) {
    // Function value spread and simplex diameter about the best vertex.
    ValueType fval_spread = 0;
    ValueType max_dist = 0;
    for (int i = 1; i <= N; ++i) {
      fval_spread = std::max(fval_spread, std::abs(fvals(0) - fvals(i)));
      for (int j = 0; j < N; ++j)
        max_dist = std::max(max_dist, std::abs(simplex(i, j) - simplex(0, j)));
    }
    if (fval_spread <= options.tol_f && max_dist <= options.tol_x) {
      converged = true;
      break;
    }

    // Centroid of all vertices except the worst
    for (int j = 0; j < N; ++j) {
      centroid(j) = 0;
      for (int i = 0; i < N; ++i)
        centroid(j) += simplex(i, j);
      centroid(j) /= static_cast<ValueType>(N);
    }

    trial(rho, xr);
    const ValueType fr = evaluate(xr);
    bool do_shrink = false;

    if (fr < fvals(0)) {
      trial(rho * chi, xe);
      const ValueType fe = evaluate(xe);
      if (fe < fr)
        replace_worst(xe, fe);
      else
        replace_worst(xr, fr);
    } else if (fr < fvals(N - 1)) {
      replace_worst(xr, fr);
    } else if (fr < fvals(N)) {
      // Outside contraction
      trial(psi * rho, xc);
      const ValueType fc = evaluate(xc);
      if (fc <= fr)
        replace_worst(xc, fc);
      else
        do_shrink = true;
    } else {
      // Inside contraction
      trial(-psi, xc);
      const ValueType fc = evaluate(xc);
      if (fc < fvals(N))
        replace_worst(xc, fc);
      else
        do_shrink = true;
    }

    if (do_shrink) {
      // Move all vertices toward the best
      for (int i = 1; i <= N; ++i) {
        for (int j = 0; j < N; ++j) {
          simplex(i, j) =
              simplex(0, j) + sigma * (simplex(i, j) - simplex(0, j));
          point(j) = simplex(i, j);
        }
        fvals(i) = evaluate(point);
      }
    }

    sort_simplex();
    ++iter;
  }

  OptimizationResult<N, ValueType> result;
  result.x = point_type("result_x");
  for (int j = 0; j < N; ++j) {
    result.x(j) = simplex(0, j);
  }
  result.min_value = fvals(0);
  result.iterations = iter;
  result.converged = converged;
  result.function_evaluations = function_evaluations;

  return result;
}

} // namespace optimization
} // namespace specfem
