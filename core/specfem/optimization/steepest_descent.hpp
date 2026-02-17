#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>

namespace specfem {
namespace optimization {

// ============================================================================
// Steepest Descent Tag
// ============================================================================

/**
 * @brief Tag for steepest descent (gradient descent) algorithm
 *
 * First-order gradient-based method using backtracking line search.
 * Computes gradient numerically via central finite differences.
 * Faster than Nelder-Mead for smooth functions but requires differentiability.
 *
 * @note This is mainly intended for educational purposes, just to show how to
 * implement interfaces with and without gradients.
 *
 * @see optimize(SteepestDescent, Func &&objective, SteepestDescentOptions<N>
 * options)
 * @see optimize(SteepestDescent, Func &&objective, GradFunc &&gradient,
 * SteepestDescentOptions<N> options)
 */
struct SteepestDescent {};

// ============================================================================
// Steepest Descent Options
// ============================================================================

/**
 * @brief Configuration for steepest descent algorithm
 *
 * Controls convergence criteria and step size parameters.
 * Uses backtracking line search with Armijo condition.
 *
 *
 * @tparam N Number of optimization variables
 */
template <int N> struct SteepestDescentOptions {
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
      x0;                      ///< Initial guess
  int max_iterations = -1;     ///< Maximum iterations (-1 = 1000*N)
  type_real tol_grad = 1.0e-6; ///< Tolerance on gradient norm
  type_real tol_f = 1.0e-8;    ///< Tolerance on function value change
  type_real tol_x = 1.0e-8;    ///< Tolerance on step size

  // Line search parameters (Armijo backtracking)
  type_real initial_step = 1.0;  ///< Initial step size for line search
  type_real armijo_c = 1.0e-4;   ///< Armijo condition constant (0 < c < 1)
  type_real backtrack_rho = 0.5; ///< Backtracking factor (0 < rho < 1)
  int max_line_search = 50;      ///< Maximum line search iterations

  // Numerical gradient parameters (only used when gradient not provided)
  // Use sqrt(machine epsilon) for finite difference step size
  // For float: ~1e-4, for double: ~1e-8
  type_real grad_epsilon = std::sqrt(
      std::numeric_limits<type_real>::epsilon()); ///< Finite difference step
                                                  ///< size
};

// ============================================================================
// Steepest Descent with user-provided gradient
// ============================================================================

/**
 * @brief Steepest descent minimization with user-provided gradient
 *
 * First-order optimization method that iteratively moves in the direction
 * of the negative gradient. Uses backtracking line search with Armijo
 * condition to determine step size.
 *
 * @tparam N Problem dimension
 * @tparam Func Callable with signature: type_real(View<type_real[N]>)
 * @tparam GradFunc Callable with signature: void(View<type_real[N]> x,
 *                                                View<type_real[N]> grad_out)
 *
 * @param tag Algorithm selector (SteepestDescent{})
 * @param objective Function to minimize
 * @param gradient Function computing gradient: gradient(x, grad_out)
 * @param options Configuration with initial guess and tolerances
 * @return OptimizationResult with solution, value, iterations, and convergence
 * flag
 *
 * Convergence occurs when any of:
 * - Gradient norm < tol_grad
 * - Function value change < tol_f
 * - Step size < tol_x
 *
 * @code
 * // Minimize (x-1)^2 + (y-2)^2 with analytical gradient
 * auto f = [](auto x) { return (x(0)-1)*(x(0)-1) + (x(1)-2)*(x(1)-2); };
 * auto grad_f = [](auto x, auto g) { g(0) = 2*(x(0)-1); g(1) = 2*(x(1)-2); };
 * Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
 * x0(0) = 0.0; x0(1) = 0.0;
 * SteepestDescentOptions<2> opts{x0};
 * auto result = optimize(SteepestDescent{}, f, grad_f, opts);
 * // result.x(0) ≈ 1.0, result.x(1) ≈ 2.0
 * @endcode
 */
template <int N, typename Func, typename GradFunc>
OptimizationResult<N> optimize(SteepestDescent, Func &&objective,
                               GradFunc &&gradient,
                               SteepestDescentOptions<N> options) {
  // Extract options
  auto x0 = options.x0;
  int max_iterations = options.max_iterations;
  const type_real tol_grad = options.tol_grad;
  const type_real tol_f = options.tol_f;
  const type_real tol_x = options.tol_x;
  const type_real initial_step = options.initial_step;
  const type_real armijo_c = options.armijo_c;
  const type_real backtrack_rho = options.backtrack_rho;
  const int max_line_search = options.max_line_search;

  // Default max iterations
  if (max_iterations < 0) {
    max_iterations = 1000 * N;
  }

  // Current point
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> x("x");
  for (int j = 0; j < N; ++j) {
    x(j) = x0(j);
  }

  // Working arrays
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> grad(
      "grad");
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> x_new(
      "x_new");

  // Helper lambda for gradient norm
  auto gradient_norm = [](auto &g) {
    type_real norm = 0.0;
    for (int j = 0; j < N; ++j) {
      norm += g(j) * g(j);
    }
    return std::sqrt(norm);
  };

  type_real f_val = objective(x);
  int iter = 0;
  bool converged = false;

  while (iter < max_iterations) {
    ++iter;

    // Compute gradient at current point using user-provided function
    gradient(x, grad);

    // Check gradient norm convergence
    type_real grad_norm = gradient_norm(grad);
    if (grad_norm < tol_grad) {
      converged = true;
      break;
    }

    // Backtracking line search with Armijo condition
    // Find step size alpha such that:
    // f(x - alpha * grad) <= f(x) - c * alpha * ||grad||^2
    type_real alpha = initial_step;
    type_real expected_decrease = armijo_c * grad_norm * grad_norm;

    int line_search_iter = 0;
    bool found_step = false;

    while (line_search_iter < max_line_search) {
      // Compute trial point: x_new = x - alpha * grad
      for (int j = 0; j < N; ++j) {
        x_new(j) = x(j) - alpha * grad(j);
      }

      type_real f_new = objective(x_new);

      // Armijo condition
      if (f_new <= f_val - alpha * expected_decrease) {
        found_step = true;
        break;
      }

      // Backtrack
      alpha *= backtrack_rho;
      ++line_search_iter;
    }

    // If line search failed, try a very small step
    if (!found_step) {
      alpha = 1.0e-10;
      for (int j = 0; j < N; ++j) {
        x_new(j) = x(j) - alpha * grad(j);
      }
    }

    // Compute actual step size
    type_real step_size = 0.0;
    for (int j = 0; j < N; ++j) {
      type_real diff = x_new(j) - x(j);
      step_size += diff * diff;
    }
    step_size = std::sqrt(step_size);

    // Check step size convergence
    if (step_size < tol_x) {
      converged = true;
      // Update x to x_new before breaking
      for (int j = 0; j < N; ++j) {
        x(j) = x_new(j);
      }
      f_val = objective(x);
      break;
    }

    // Compute new function value
    type_real f_new = objective(x_new);

    // Check function value convergence
    type_real f_change = std::abs(f_val - f_new);
    if (f_change < tol_f) {
      converged = true;
      // Update x to x_new before breaking
      for (int j = 0; j < N; ++j) {
        x(j) = x_new(j);
      }
      f_val = f_new;
      break;
    }

    // Update for next iteration
    for (int j = 0; j < N; ++j) {
      x(j) = x_new(j);
    }
    f_val = f_new;
  }

  // Prepare result
  OptimizationResult<N> result;
  result.x = Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>(
      "result_x");
  for (int j = 0; j < N; ++j) {
    result.x(j) = x(j);
  }
  result.min_value = f_val;
  result.iterations = iter;
  result.converged = converged;

  return result;
}

// ============================================================================
// Steepest Descent with numerical gradient (convenience overload)
// ============================================================================

/**
 * @brief Steepest descent minimization with numerical gradient
 *
 * Convenience overload that computes the gradient numerically using
 * central finite differences. For better performance and accuracy,
 * prefer the overload that accepts an analytical gradient function.
 *
 * @tparam N Problem dimension
 * @tparam Func Callable with signature: type_real(View<type_real[N]>)
 *
 * @param tag Algorithm selector (SteepestDescent{})
 * @param objective Function to minimize
 * @param options Configuration with initial guess and tolerances
 * @return OptimizationResult with solution, value, iterations, and convergence
 * flag
 *
 * @code
 * // Minimize (x-1)^2 + (y-2)^2 with numerical gradient
 * auto f = [](auto x) { return (x(0)-1)*(x(0)-1) + (x(1)-2)*(x(1)-2); };
 * Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
 * x0(0) = 0.0; x0(1) = 0.0;
 * SteepestDescentOptions<2> opts{x0};
 * auto result = optimize(SteepestDescent{}, f, opts);
 * // result.x(0) ≈ 1.0, result.x(1) ≈ 2.0
 * @endcode
 */
template <int N, typename Func>
OptimizationResult<N> optimize(SteepestDescent, Func &&objective,
                               SteepestDescentOptions<N> options) {
  const type_real grad_epsilon = options.grad_epsilon;

  // Working array for finite difference computation - use unique label
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace> x_temp(
      "x_temp_for_numerical_gradient");

  // Create numerical gradient functor - capture x_temp by value
  auto numerical_gradient = [objective, x_temp, grad_epsilon](
                                const auto &point, auto &grad_out) mutable {
    const int dim = point.extent(0); // Get dimension from the View itself

    for (int j = 0; j < dim; ++j) {
      // Copy point to temp
      for (int k = 0; k < dim; ++k) {
        x_temp(k) = point(k);
      }

      // Forward perturbation
      x_temp(j) = point(j) + grad_epsilon;
      type_real f_plus = objective(x_temp);

      // Backward perturbation
      x_temp(j) = point(j) - grad_epsilon;
      type_real f_minus = objective(x_temp);

      // Central difference
      grad_out(j) = (f_plus - f_minus) / (2.0 * grad_epsilon);
    }
  };

  // Don't forward objective since we captured it by value in the lambda
  return optimize(SteepestDescent{}, objective, numerical_gradient, options);
}

} // namespace optimization
} // namespace specfem
