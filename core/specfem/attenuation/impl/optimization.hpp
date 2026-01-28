#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

namespace specfem {
namespace attenuation {
namespace impl {

/**
 * @brief Result from Nelder-Mead optimization
 *
 * @tparam N Number of optimization variables \f$N\f$
 */
template <int N>
struct OptimizationResult {
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> x;
  type_real min_value;
  int iterations;
  bool converged;
};

/**
 * @brief Nelder-Mead simplex minimization algorithm
 *
 * Implements the Nelder-Mead (downhill simplex) method for derivative-free
 * optimization. This algorithm iteratively updates a simplex of \f$N+1\f$ points
 * in \f$N\f$-dimensional space to find a local minimum.
 *
 * The algorithm uses the following operations:
 * - Reflection: reflect the worst point through the centroid
 * - Expansion: if reflection is good, try expanding further
 * - Contraction: if reflection is bad, try contracting
 * - Shrink: if all else fails, shrink the simplex toward the best point
 *
 * Matching SPECFEM3D Fortran implementation constants:
 * - \f$\rho = 1.0\f$ (reflection coefficient)
 * - \f$\chi = 2.0\f$ (expansion coefficient)
 * - \f$\psi = 0.5\f$ (contraction coefficient)
 * - \f$\sigma = 0.5\f$ (shrink coefficient)
 *
 * @tparam N Number of optimization variables \f$N\f$
 * @tparam Func Callable type with signature type_real(View<type_real[N]>)
 *
 * @param objective The objective function \f$f(\mathbf{x})\f$ to minimize
 * @param x0 Initial guess \f$\mathbf{x}_0\f$ (\f$N\f$-dimensional point)
 * @param max_iterations Maximum iterations (\f$-1\f$ for default \f$200N\f$)
 * @param tol_f Function value tolerance for convergence
 * @param tol_x Variable tolerance for convergence
 *
 * @return OptimizationResult containing the minimum point, value, iteration
 * count, and convergence status
 *
 * @code
 * // Example: minimize (x-1)^2 + (y-2)^2
 * auto objective = [](auto x) { return (x(0)-1)*(x(0)-1) + (x(1)-2)*(x(1)-2);
 * }; Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace>
 * x0("x0"); x0(0) = 0.0; x0(1) = 0.0; auto result = fmin_search<2>(objective,
 * x0);
 * // result.x(0) ≈ 1.0, result.x(1) ≈ 2.0
 * @endcode
 */
template <int N, typename Func>
OptimizationResult<N>
fmin_search(Func &&objective,
            Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
                x0,
            int max_iterations = -1, type_real tol_f = 1.0e-4,
            type_real tol_x = 1.0e-4) {
  // Nelder-Mead parameters (matching Fortran SPECFEM3D)
  constexpr type_real rho = 1.0;   // reflection coefficient
  constexpr type_real chi = 2.0;   // expansion coefficient
  constexpr type_real psi = 0.5;   // contraction coefficient
  constexpr type_real sigma = 0.5; // shrink coefficient

  // Initial simplex construction parameters
  constexpr type_real usual_delta = 0.05;
  constexpr type_real zero_term_delta = 0.00025;

  // Default max iterations
  if (max_iterations < 0) {
    max_iterations = 200 * N;
  }

  // Simplex has N+1 vertices, each is an N-dimensional point
  Kokkos::View<type_real[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      simplex("simplex");
  Kokkos::View<type_real[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace> fvals(
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
    if (std::abs(x0(i - 1)) > 1.0e-10) {
      simplex(i, i - 1) = x0(i - 1) * (1.0 + usual_delta);
    } else {
      simplex(i, i - 1) = zero_term_delta;
    }
  }

  // Evaluate objective at all simplex vertices
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> point(
      "point");
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j < N; ++j) {
      point(j) = simplex(i, j);
    }
    fvals(i) = objective(point);
  }

  // Sort indices by function value (ascending)
  int indices[N + 1];
  for (int i = 0; i <= N; ++i) {
    indices[i] = i;
  }
  std::sort(indices, indices + N + 1,
            [&fvals](int a, int b) { return fvals(a) < fvals(b); });

  // Reorder simplex and fvals according to sorted order
  Kokkos::View<type_real[N + 1][N], Kokkos::LayoutRight, Kokkos::HostSpace>
      temp_simplex("temp_simplex");
  Kokkos::View<type_real[N + 1], Kokkos::LayoutRight, Kokkos::HostSpace>
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
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> centroid(
      "centroid");
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> xr(
      "xr"); // reflection
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> xe(
      "xe"); // expansion
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> xc(
      "xc"); // contraction

  int iter = 0;
  bool converged = false;

  while (iter < max_iterations) {
    ++iter;

    // Check convergence: function value spread
    type_real fval_spread = fvals(N) - fvals(0);

    // Check convergence: simplex diameter (max distance from best point)
    type_real max_dist = 0.0;
    for (int i = 1; i <= N; ++i) {
      type_real dist = 0.0;
      for (int j = 0; j < N; ++j) {
        type_real diff = simplex(i, j) - simplex(0, j);
        dist += diff * diff;
      }
      dist = std::sqrt(dist);
      if (dist > max_dist) {
        max_dist = dist;
      }
    }

    if (fval_spread < tol_f && max_dist < tol_x) {
      converged = true;
      break;
    }

    // Compute centroid of all points except the worst (last)
    for (int j = 0; j < N; ++j) {
      centroid(j) = 0.0;
      for (int i = 0; i < N; ++i) { // exclude worst point (index N)
        centroid(j) += simplex(i, j);
      }
      centroid(j) /= static_cast<type_real>(N);
    }

    // Reflection: xr = centroid + rho * (centroid - worst)
    for (int j = 0; j < N; ++j) {
      xr(j) = centroid(j) + rho * (centroid(j) - simplex(N, j));
    }
    type_real fr = objective(xr);

    if (fr < fvals(0)) {
      // Reflection is best so far, try expansion
      for (int j = 0; j < N; ++j) {
        xe(j) = centroid(j) + chi * (xr(j) - centroid(j));
      }
      type_real fe = objective(xe);

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
          xc(j) = centroid(j) + psi * (xr(j) - centroid(j));
        }
        type_real fc = objective(xc);

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
          xc(j) = centroid(j) - psi * (centroid(j) - simplex(N, j));
        }
        type_real fc = objective(xc);

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
            simplex(i, j) = simplex(0, j) + sigma * (simplex(i, j) - simplex(0, j));
            point(j) = simplex(i, j);
          }
          fvals(i) = objective(point);
        }
      }
    }

    // Re-sort simplex by function value
    for (int i = 0; i <= N; ++i) {
      indices[i] = i;
    }
    std::sort(indices, indices + N + 1,
              [&fvals](int a, int b) { return fvals(a) < fvals(b); });

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
  }

  // Prepare result
  OptimizationResult<N> result;
  result.x = Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>(
      "result_x");
  for (int j = 0; j < N; ++j) {
    result.x(j) = simplex(0, j);
  }
  result.min_value = fvals(0);
  result.iterations = iter;
  result.converged = converged;

  return result;
}

} // namespace impl
} // namespace attenuation
} // namespace specfem
