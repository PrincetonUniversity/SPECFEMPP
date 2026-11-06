#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

namespace specfem {
namespace optimization {

// ============================================================================
// Nelder-Mead Tag
// ============================================================================

/**
 * @brief Tag for Nelder-Mead simplex algorithm
 *
 *
 * Derivative-free method suitable for non-smooth objective functions.
 * Converges slowly but robustly for problems with few variables (N < 10).
 *
 * @see optimize(NelderMeadSimplex, Func &&objective, NelderMeadOptions<N>
 * options)
 */
struct NelderMeadSimplex {};

// ============================================================================
// Nelder-Mead Options
// ============================================================================

/**
 * @brief Configuration for Nelder-Mead simplex algorithm
 *
 * Controls convergence criteria and simplex transformation coefficients.
 * Default values work well for most problems.
 *
 * With Globe=true, use double precision, Fortran-rounded defaults, exact-zero
 * initialization, heap-sort tie ordering and GLOBE trial expressions.
 * Convergence includes equality; the initial simplex counts as iteration one.
 *
 * @tparam ValueType Scalar type (defaults to type_real)
 * @tparam Globe Enable SPECFEM3D_GLOBE compatibility
 * @tparam N Number of optimization variables
 */
template <int N, typename ValueType = type_real, bool Globe = false>
struct NelderMeadOptions {
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace>
      x0;                  ///< Initial guess
  int max_iterations = -1; ///< Maximum iterations (-1 = 200*N)
  ValueType tol_f =
      Globe ? 1.0e-4f : 1.0e-4; ///< Tolerance on function value spread
  ValueType tol_x = Globe ? 1.0e-4f : 1.0e-4; ///< Tolerance on simplex diameter

  // Simplex transformation coefficients (standard values recommended)
  ValueType reflection = 1.0;  ///< Reflection coefficient
  ValueType expansion = 2.0;   ///< Expansion coefficient
  ValueType contraction = 0.5; ///< Contraction coefficient
  ValueType shrink = 0.5;      ///< Shrink coefficient
  /// Evaluation budget (nonpositive = 200*N for Globe, unlimited otherwise).
  /// A started iteration always completes and may exceed this budget.
  int max_function_evaluations = -1;
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
void globe_heap_sort(std::array<double, M + 1> &x, std::array<int, M + 1> &y) {
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

// ============================================================================
// Nelder-Mead optimize specialization
// ============================================================================

/**
 * @brief Nelder-Mead downhill simplex minimization
 *
 * Derivative-free optimization using @f$N+1@f$ simplex vertices in
 * @f$N@f$-dimensional space. Iteratively applies reflection, expansion,
 * contraction, and shrink operations to locate a local minimum.
 *
 * @tparam N Problem dimension
 * @tparam Func Callable with signature: ValueType(View<ValueType[N]>)
 *
 * @param tag Algorithm selector (NelderMeadSimplex{})
 * @param objective Function to minimize
 * @param options Configuration with initial guess and tolerances
 * @return OptimizationResult with solution, value, iterations, and convergence
 * flag
 *
 * Convergence occurs when both:
 * - Function value spread < tol_f
 * - Simplex diameter < tol_x
 *
 * With Globe=true, both comparisons include equality and the initial simplex
 * counts as iteration one. The default evaluation budget is 200*N.
 * @tparam ValueType Scalar type
 * @tparam Globe Enable GLOBE initialization, arithmetic and stopping rules
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
template <int N, typename ValueType, bool Globe, typename Func>
OptimizationResult<N, ValueType>
optimize(NelderMeadSimplex, Func &&objective,
         NelderMeadOptions<N, ValueType, Globe> options) {
  static_assert(!Globe || std::is_same_v<ValueType, double>,
                "GLOBE compatibility requires double precision");
  int function_evaluations = 0;
  const auto evaluate = [&](const auto &point) {
    ++function_evaluations;
    return static_cast<ValueType>(objective(point));
  };
  const int max_evaluations = options.max_function_evaluations > 0
                                  ? options.max_function_evaluations
                                  : (Globe ? 200 * N : 0);
  // Extract options
  auto x0 = options.x0;
  int max_iterations = options.max_iterations;
  const ValueType tol_f = options.tol_f;
  const ValueType tol_x = options.tol_x;
  const ValueType rho = options.reflection;
  const ValueType chi = options.expansion;
  const ValueType psi = options.contraction;
  const ValueType sigma = options.shrink;

  // Initial simplex construction parameters
  constexpr ValueType usual_delta = Globe ? 0.05f : 0.05;
  constexpr ValueType zero_term_delta = Globe ? 0.00025f : 0.00025;

  // Default max iterations
  if (max_iterations < 0 || (Globe && max_iterations == 0)) {
    max_iterations = 200 * N;
  }

  // Simplex has N+1 vertices, each is an N-dimensional point
  Kokkos::View<ValueType[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      simplex("simplex");
  Kokkos::View<ValueType[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace> fvals(
      "fvals");

  // Initialize first vertex with x0
  for (int j = 0; j < N; ++j) {
    simplex(0, j) = x0(j);
  }

  // Build initial simplex by perturbing each coordinate
  for (int i = 1; i <= N; ++i) {
    for (int j = 0; j < N; ++j) {
      simplex(i, j) = x0(j);
    }
    // Perturb the (i-1)-th coordinate
    if (Globe ? x0(i - 1) != 0.0 : std::abs(x0(i - 1)) > 1.0e-10) {
      simplex(i, i - 1) = x0(i - 1) * (1.0 + usual_delta);
    } else {
      simplex(i, i - 1) = zero_term_delta;
    }
  }

  // Evaluate objective at all simplex vertices
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace> point(
      "point");
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j < N; ++j) {
      point(j) = simplex(i, j);
    }
    fvals(i) = evaluate(point);
  }

  // Sort indices by function value (ascending)
  int indices[N + 1];
  for (int i = 0; i <= N; ++i) {
    indices[i] = i;
  }
  const auto sort_indices = [&]() {
    if constexpr (Globe) {
      std::array<double, N + 2> values{};
      std::array<int, N + 2> places{};
      for (int i = 0; i <= N; ++i)
        values[i + 1] = fvals(i);
      globe_heap_sort<N + 1>(values, places);
      for (int i = 0; i <= N; ++i)
        indices[i] = places[i + 1] - 1;
    } else {
      std::sort(indices, indices + N + 1,
                [&fvals](int a, int b) { return fvals(a) < fvals(b); });
    }
  };
  sort_indices();

  // Reorder simplex and fvals according to sorted order
  Kokkos::View<ValueType[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      temp_simplex("temp_simplex");
  Kokkos::View<ValueType[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace>
      temp_fvals("temp_fvals");

  for (int i = 0; i <= N; ++i) {
    temp_fvals(i) = fvals(indices[i]);
    for (int j = 0; j < N; ++j) {
      temp_simplex(i, j) = simplex(indices[i], j);
    }
  }
  for (int i = 0; i <= N; ++i) {
    fvals(i) = temp_fvals(i);
    for (int j = 0; j < N; ++j) {
      simplex(i, j) = temp_simplex(i, j);
    }
  }

  // Working arrays
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace> centroid(
      "centroid");
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace> xr(
      "xr"); // reflection
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace> xe(
      "xe"); // expansion
  Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace> xc(
      "xc"); // contraction

  int iter = Globe ? 1 : 0;
  bool converged = false;

  while (iter < max_iterations &&
         (max_evaluations == 0 || function_evaluations < max_evaluations)) {
    if constexpr (!Globe)
      ++iter;

    // Check convergence: function value spread
    ValueType fval_spread = fvals(N) - fvals(0);
    if constexpr (Globe) {
      fval_spread = 0;
      for (int i = 1; i <= N; ++i)
        fval_spread = std::max(fval_spread, std::abs(fvals(0) - fvals(i)));
    }

    // Check convergence: simplex diameter (max coordinate-wise abs difference
    // from best point, matching Fortran SPECFEM3D max_size_simplex)
    ValueType max_dist = 0.0;
    for (int i = 1; i <= N; ++i) {
      for (int j = 0; j < N; ++j) {
        ValueType diff = std::abs(simplex(i, j) - simplex(0, j));
        if (diff > max_dist) {
          max_dist = diff;
        }
      }
    }

    if (Globe ? (fval_spread <= tol_f && max_dist <= tol_x)
              : (fval_spread < tol_f && max_dist < tol_x)) {
      converged = true;
      break;
    }

    // Compute centroid of all points except the worst (last)
    for (int j = 0; j < N; ++j) {
      centroid(j) = 0.0;
      for (int i = 0; i < N; ++i) { // exclude worst point (index N)
        centroid(j) += simplex(i, j);
      }
      centroid(j) /= static_cast<ValueType>(N);
    }

    // Reflection: xr = centroid + rho * (centroid - worst)
    for (int j = 0; j < N; ++j) {
      if constexpr (Globe)
        xr(j) = (1 + rho) * centroid(j) + (-rho) * simplex(N, j);
      else
        xr(j) = centroid(j) + rho * (centroid(j) - simplex(N, j));
    }
    ValueType fr = evaluate(xr);

    if (fr < fvals(0)) {
      // Reflection is best so far, try expansion
      for (int j = 0; j < N; ++j) {
        if constexpr (Globe)
          xe(j) = (1 + rho * chi) * centroid(j) + (-rho * chi) * simplex(N, j);
        else
          xe(j) = centroid(j) + chi * (xr(j) - centroid(j));
      }
      ValueType fe = evaluate(xe);

      if (fe < fr) {
        // Accept expansion
        for (int j = 0; j < N; ++j) {
          simplex(N, j) = xe(j);
        }
        fvals(N) = fe;
      } else {
        // Accept reflection
        for (int j = 0; j < N; ++j) {
          simplex(N, j) = xr(j);
        }
        fvals(N) = fr;
      }
    } else if (fr < fvals(N - 1)) {
      // Reflection is better than second worst, accept it
      for (int j = 0; j < N; ++j) {
        simplex(N, j) = xr(j);
      }
      fvals(N) = fr;
    } else {
      // Contraction
      bool do_shrink = false;

      if (fr < fvals(N)) {
        // Outside contraction: xc = centroid + psi * (xr - centroid)
        for (int j = 0; j < N; ++j) {
          if constexpr (Globe)
            xc(j) =
                (1 + psi * rho) * centroid(j) + (-psi * rho) * simplex(N, j);
          else
            xc(j) = centroid(j) + psi * (xr(j) - centroid(j));
        }
        ValueType fc = evaluate(xc);

        if (fc <= fr) {
          for (int j = 0; j < N; ++j) {
            simplex(N, j) = xc(j);
          }
          fvals(N) = fc;
        } else {
          do_shrink = true;
        }
      } else {
        // Inside contraction: xc = centroid - psi * (centroid - worst)
        for (int j = 0; j < N; ++j) {
          if constexpr (Globe)
            xc(j) = (1 - psi) * centroid(j) + psi * simplex(N, j);
          else
            xc(j) = centroid(j) - psi * (centroid(j) - simplex(N, j));
        }
        ValueType fc = evaluate(xc);

        if (fc < fvals(N)) {
          for (int j = 0; j < N; ++j) {
            simplex(N, j) = xc(j);
          }
          fvals(N) = fc;
        } else {
          do_shrink = true;
        }
      }

      if (do_shrink) {
        // Shrink: move all points toward the best
        for (int i = 1; i <= N; ++i) {
          for (int j = 0; j < N; ++j) {
            simplex(i, j) =
                simplex(0, j) + sigma * (simplex(i, j) - simplex(0, j));
            point(j) = simplex(i, j);
          }
          fvals(i) = evaluate(point);
        }
      }
    }

    // Re-sort simplex by function value
    for (int i = 0; i <= N; ++i) {
      indices[i] = i;
    }
    sort_indices();

    for (int i = 0; i <= N; ++i) {
      temp_fvals(i) = fvals(indices[i]);
      for (int j = 0; j < N; ++j) {
        temp_simplex(i, j) = simplex(indices[i], j);
      }
    }
    for (int i = 0; i <= N; ++i) {
      fvals(i) = temp_fvals(i);
      for (int j = 0; j < N; ++j) {
        simplex(i, j) = temp_simplex(i, j);
      }
    }
    if constexpr (Globe)
      ++iter;
  }

  // Prepare result
  OptimizationResult<N, ValueType> result;
  result.x = Kokkos::View<ValueType[N], Kokkos::LayoutRight, Kokkos::HostSpace>(
      "result_x");
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
