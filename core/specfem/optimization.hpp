#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

/**
 * @brief This namespace contains optimization algorithms and related types
 *
 * The main interface is the optimize() function, which is overloaded for
 * different algorithms and supports both numerical and user-provided gradients.
 *
 * For implementing a new algorithm simply define a new tag struct (e.g.,
 * MyAlgorithm) and provide specializations of optimize() for that tag. This
 * design allows for a clean separation of algorithms and a consistent user
 * interface.
 */
namespace optimization {

// ============================================================================
// Result
// ============================================================================

/**
 * @brief Result structure returned by optimization algorithms
 *
 * @tparam N Number of optimization variables
 */
template <int N> struct OptimizationResult {
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
      x;               ///< Optimal point found
  type_real min_value; ///< Function value at x
  int iterations;      ///< Number of iterations performed
  bool converged;      ///< True if tolerances were met

  /// @brief Accessor returning optimal point
  Kokkos::View<type_real[N], Kokkos::LayoutRight, Kokkos::HostSpace>
  operator()() const {
    return x;
  }
};

// ============================================================================
// Base optimize template (specializations in algorithm headers)
// ============================================================================

/**
 * @brief Generic optimization interface for derivative-free optimization
 * algorithms
 *
 * Provides optimization methods for unconstrained minimization problems. All
 * algorithms operate on Kokkos::View arrays for device compatibility.
 *
 * @par Usage pattern:
 * 1. Select an algorithm tag (e.g., NelderMeadSimplex)
 * 2. Configure options (initial point, tolerances)
 * 3. Call optimize() with objective function
 *
 * Example:
 * @code
 * auto objective = [](auto x) { return x(0)*x(0) + x(1)*x(1); };
 * NelderMeadOptions<2> opts; opts.x0 = initial_guess; auto result =
 * optimize(NelderMeadSimplex{}, objective, opts);
 * @endcode
 *
 * @tparam AlgorithmTag Algorithm selector (e.g., NelderMeadSimplex)
 * @tparam N Dimension of optimization problem
 * @tparam Func Objective function type: type_real(View<type_real[N]>)
 * @tparam Options Algorithm-specific options structure
 *
 * @param tag Algorithm tag selecting the optimization method
 * @param objective Function to minimize
 * @param options Algorithm configuration and initial point
 * @return OptimizationResult containing solution and diagnostics
 *
 * @note Specializations for each algorithm tag are in separate headers.
 */
template <typename AlgorithmTag, int N, typename Func, typename Options>
OptimizationResult<N> optimize(AlgorithmTag tag, Func &&objective,
                               Options options);

/**
 * @brief Generic optimization interface with user-provided gradient
 *
 * Provides optimization methods for unconstrained minimization problems. All
 * algorithms operate on Kokkos::View arrays for device compatibility.
 *
 * @par Usage pattern:
 * 1. Select an algorithm tag (e.g., SteepestDescent)
 * 2. Configure options (initial point, tolerances)
 * 3. Call optimize() with objective function and gradient function
 *
 * Example:
 * @code
 * auto objective = [](auto x) { return x(0)*x(0) + x(1)*x(1); };
 * SteepestDescentOptions<2> opts; opts.x0 = initial_guess; auto result =
 * optimize(SteepestDescent{}, objective, gradient, opts);
 * @endcode
 *
 *
 * @tparam AlgorithmTag Algorithm selector (e.g., SteepestDescent)
 * @tparam N Dimension of optimization problem
 * @tparam Func Objective function type: type_real(View<type_real[N]>)
 * @tparam GradFunc Gradient function type: void(View<type_real[N]> x,
 *                                               View<type_real[N]> grad_out)
 * @tparam Options Algorithm-specific options structure
 *
 * @param tag Algorithm tag selecting the optimization method
 * @param objective Function to minimize
 * @param gradient Function computing gradient at a point
 * @param options Algorithm configuration and initial point
 * @return OptimizationResult containing solution and diagnostics
 *
 * @note Only supported by gradient-based algorithms (e.g., SteepestDescent).
 */
template <typename AlgorithmTag, int N, typename Func, typename GradFunc,
          typename Options>
OptimizationResult<N> optimize(AlgorithmTag tag, Func &&objective,
                               GradFunc &&gradient, Options options);

} // namespace optimization
} // namespace specfem

// Include algorithm implementations
#include "optimization/nelder_mead_simplex.hpp"
#include "optimization/steepest_descent.hpp"
