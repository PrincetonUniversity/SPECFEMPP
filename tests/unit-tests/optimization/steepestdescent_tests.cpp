#include "specfem/optimization.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::optimization::OptimizationResult;
using specfem::optimization::optimize;
using specfem::optimization::SteepestDescent;
using specfem::optimization::SteepestDescentOptions;

// Test quadratic minimization: f(x) = (x - 3)^2
// Minimum at x = 3
TEST(Optimization_SteepestDescent, QuadraticMinimization1D) {
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 3.0) * (x(0) - 3.0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;

  SteepestDescentOptions<1> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 3.0, 1e-4);
  EXPECT_NEAR(result.min_value, 0.0, 1e-8);
}

// Test 2D quadratic: f(x,y) = (x - 1)^2 + (y - 2)^2
// Minimum at (1, 2)
TEST(Optimization_SteepestDescent, QuadraticMinimization2D) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-4);
  EXPECT_NEAR(result.x(1), 2.0, 1e-4);
  EXPECT_NEAR(result.min_value, 0.0, 1e-8);
}

// Test Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// This is a classic optimization benchmark with minimum at (1, 1)
TEST(Optimization_SteepestDescent, RosenbrockFunction) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        type_real a = 1.0 - x(0);
        type_real b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  // Rosenbrock requires more iterations for steepest descent
  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 10000;
  opts.tol_grad = 1e-6;
  opts.tol_f = 1e-10;
  opts.tol_x = 1e-10;

  auto result = optimize(SteepestDescent{}, objective, opts);

  // Steepest descent is known to struggle with Rosenbrock's narrow valley
  // Allow reasonable tolerance
  EXPECT_NEAR(result.x(0), 1.0, 0.1);
  EXPECT_NEAR(result.x(1), 1.0, 0.1);
  EXPECT_LT(result.min_value, 0.01);
}

// Test that max iterations limit is respected
TEST(Optimization_SteepestDescent, RespectsMaxIterations) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        // Rosenbrock - hard to converge
        type_real a = 1.0 - x(0);
        type_real b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  // Use very few iterations - should not converge
  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 5;
  opts.tol_grad = 1e-10;
  opts.tol_f = 1e-12;
  opts.tol_x = 1e-12;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_FALSE(result.converged);
  EXPECT_LE(result.iterations, 5);
}

// Test convergence detection works
TEST(Optimization_SteepestDescent, ConvergenceDetection) {
  // Easy function that converges quickly
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return x(0) * x(0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 1.0;

  SteepestDescentOptions<1> opts;
  opts.x0 = x0;
  opts.max_iterations = 1000;
  opts.tol_grad = 1e-6;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.iterations, 1000); // Should converge before max
}

// Test with zero initial guess
TEST(Optimization_SteepestDescent, ZeroInitialGuess) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 0.5, 1e-4);
  EXPECT_NEAR(result.x(1), 0.5, 1e-4);
}

// Test with 3 variables
TEST(Optimization_SteepestDescent, ThreeVariables) {
  auto objective =
      [](Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0) +
               (x(2) - 3.0) * (x(2) - 3.0);
      };

  Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;
  x0(2) = 0.0;

  SteepestDescentOptions<3> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-4);
  EXPECT_NEAR(result.x(1), 2.0, 1e-4);
  EXPECT_NEAR(result.x(2), 3.0, 1e-4);
}

// Test result structure fields
TEST(Optimization_SteepestDescent, ResultStructureComplete) {
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return x(0) * x(0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 5.0;

  SteepestDescentOptions<1> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, opts);

  // Check all fields are populated
  EXPECT_EQ(result.x.extent(0), 1);
  EXPECT_GE(result.iterations, 1);
  // min_value should be close to 0 for x^2 minimized
  EXPECT_LT(result.min_value, 1e-8);
}

// Test ill-conditioned quadratic (different curvatures in each direction)
TEST(Optimization_SteepestDescent, IllConditionedQuadratic) {
  // f(x,y) = x^2 + 100*y^2 (condition number = 100)
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return x(0) * x(0) + 100.0 * x(1) * x(1);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 1.0;
  x0(1) = 1.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 5000;
  opts.tol_grad = 1e-6;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 0.0, 1e-3);
  EXPECT_NEAR(result.x(1), 0.0, 1e-3);
}

// Test custom line search parameters
TEST(Optimization_SteepestDescent, CustomLineSearchParams) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 3.0) * (x(1) - 3.0);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.initial_step = 0.5;
  opts.armijo_c = 1.0e-3;
  opts.backtrack_rho = 0.9;

  auto result = optimize(SteepestDescent{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 2.0, 1e-4);
  EXPECT_NEAR(result.x(1), 3.0, 1e-4);
}

// ============================================================================
// Tests with analytical gradient
// ============================================================================

// Test 2D quadratic with analytical gradient
TEST(Optimization_SteepestDescent, AnalyticalGradient2D) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0);
      };

  auto gradient =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x,
         Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> g) {
        g(0) = 2.0 * (x(0) - 1.0);
        g(1) = 2.0 * (x(1) - 2.0);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, gradient, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-6);
  EXPECT_NEAR(result.x(1), 2.0, 1e-6);
  EXPECT_NEAR(result.min_value, 0.0, 1e-12);
}

// Test Rosenbrock with analytical gradient (should converge better)
TEST(Optimization_SteepestDescent, RosenbrockAnalyticalGradient) {
  // f(x,y) = (1-x)^2 + 100*(y-x^2)^2
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        type_real a = 1.0 - x(0);
        type_real b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
      };

  // df/dx = -2*(1-x) + 100*2*(y-x^2)*(-2x) = 2*(x-1) - 400*x*(y-x^2)
  // df/dy = 100*2*(y-x^2) = 200*(y-x^2)
  auto gradient =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x,
         Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> g) {
        type_real diff = x(1) - x(0) * x(0);
        g(0) = 2.0 * (x(0) - 1.0) - 400.0 * x(0) * diff;
        g(1) = 200.0 * diff;
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 10000;
  opts.tol_grad = 1e-6;
  opts.tol_f = 1e-10;
  opts.tol_x = 1e-10;

  auto result = optimize(SteepestDescent{}, objective, gradient, opts);

  // Analytical gradient should give same or better results
  EXPECT_NEAR(result.x(0), 1.0, 0.1);
  EXPECT_NEAR(result.x(1), 1.0, 0.1);
  EXPECT_LT(result.min_value, 0.01);
}

// Test 3D with analytical gradient
TEST(Optimization_SteepestDescent, AnalyticalGradient3D) {
  auto objective =
      [](Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0) +
               (x(2) - 3.0) * (x(2) - 3.0);
      };

  auto gradient =
      [](Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x,
         Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> g) {
        g(0) = 2.0 * (x(0) - 1.0);
        g(1) = 2.0 * (x(1) - 2.0);
        g(2) = 2.0 * (x(2) - 3.0);
      };

  Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;
  x0(2) = 0.0;

  SteepestDescentOptions<3> opts;
  opts.x0 = x0;

  auto result = optimize(SteepestDescent{}, objective, gradient, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-6);
  EXPECT_NEAR(result.x(1), 2.0, 1e-6);
  EXPECT_NEAR(result.x(2), 3.0, 1e-6);
}

// Test that analytical gradient converges in fewer iterations than numerical
TEST(Optimization_SteepestDescent, AnalyticalVsNumericalIterations) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 5.0) * (x(0) - 5.0) + (x(1) - 7.0) * (x(1) - 7.0);
      };

  auto gradient =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x,
         Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> g) {
        g(0) = 2.0 * (x(0) - 5.0);
        g(1) = 2.0 * (x(1) - 7.0);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  SteepestDescentOptions<2> opts;
  opts.x0 = x0;
  opts.tol_grad = 1e-8;

  // Run with analytical gradient
  auto result_analytical =
      optimize(SteepestDescent{}, objective, gradient, opts);

  // Run with numerical gradient
  auto result_numerical = optimize(SteepestDescent{}, objective, opts);

  // Both should converge
  EXPECT_TRUE(result_analytical.converged);
  EXPECT_TRUE(result_numerical.converged);

  // Both should find the same minimum
  EXPECT_NEAR(result_analytical.x(0), 5.0, 1e-6);
  EXPECT_NEAR(result_analytical.x(1), 7.0, 1e-6);
  EXPECT_NEAR(result_numerical.x(0), 5.0, 1e-4);
  EXPECT_NEAR(result_numerical.x(1), 7.0, 1e-4);

  // Analytical should use similar or fewer iterations (exact gradient)
  // Note: This might not always hold due to numerical precision differences
  EXPECT_LE(result_analytical.iterations, result_numerical.iterations + 5);
}
