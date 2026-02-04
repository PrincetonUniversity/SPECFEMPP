#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @namespace specfem::optimization
 * @brief Derivative-free optimization algorithms
 *
 * Provides optimization methods for unconstrained minimization problems.
 * All algorithms operate on Kokkos::View arrays for device compatibility.
 *
 * @par Usage pattern:
 * 1. Select an algorithm tag (e.g., NelderMeadSimplex)
 * 2. Configure options (initial point, tolerances)
 * 3. Call optimize() with objective function
 *
 * Example:
 * @code
 * auto objective = [](auto x) { return x(0)*x(0) + x(1)*x(1); };
 * NelderMeadOptions<2> opts;
 * opts.x0 = initial_guess;
 * auto result = optimize(NelderMeadSimplex{}, objective, opts);
 * @endcode
 */
namespace specfem {
namespace optimization {

// ============================================================================
// Algorithm Tags
// ============================================================================

/**
 * @brief Tag for Nelder-Mead simplex algorithm
 *
 * Derivative-free method suitable for non-smooth objective functions.
 * Converges slowly but robustly for problems with few variables (N < 10).
 */
struct NelderMeadSimplex {};
// struct BFGS {};              // Future: gradient-based
// struct ConjugateGradient {}; // Future: gradient-based

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
 * @brief Generic optimization interface
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

} // namespace optimization
} // namespace specfem

// Include algorithm implementations
#include "optimization/neldermeadsimplex.hpp"
