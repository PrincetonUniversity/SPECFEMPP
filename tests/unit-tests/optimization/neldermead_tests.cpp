#include "specfem/optimization.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::optimization::NelderMeadOptions;
using specfem::optimization::NelderMeadSimplex;
using specfem::optimization::OptimizationResult;
using specfem::optimization::optimize;

// Test quadratic minimization: f(x) = (x - 3)^2
// Minimum at x = 3
TEST(Optimization_NelderMead, QuadraticMinimization1D) {
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 3.0) * (x(0) - 3.0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;

  NelderMeadOptions<1> opts;
  opts.x0 = x0;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 3.0, 1e-3);
  EXPECT_NEAR(result.min_value, 0.0, 1e-6);
}

// Test 2D quadratic: f(x,y) = (x - 1)^2 + (y - 2)^2
// Minimum at (1, 2)
TEST(Optimization_NelderMead, QuadraticMinimization2D) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  NelderMeadOptions<2> opts;
  opts.x0 = x0;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-3);
  EXPECT_NEAR(result.x(1), 2.0, 1e-3);
  EXPECT_NEAR(result.min_value, 0.0, 1e-6);
}

// Test Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// This is a classic optimization benchmark with minimum at (1, 1)
TEST(Optimization_NelderMead, RosenbrockFunction) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        type_real a = 1.0 - x(0);
        type_real b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  // Rosenbrock requires more iterations
  NelderMeadOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 2000;
  opts.tol_f = 1e-6;
  opts.tol_x = 1e-6;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  // Rosenbrock is notoriously difficult, allow some tolerance
  EXPECT_NEAR(result.x(0), 1.0, 0.05);
  EXPECT_NEAR(result.x(1), 1.0, 0.05);
  EXPECT_LT(result.min_value, 0.01);
}

// Test that max iterations limit is respected
TEST(Optimization_NelderMead, RespectsMaxIterations) {
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
  NelderMeadOptions<2> opts;
  opts.x0 = x0;
  opts.max_iterations = 5;
  opts.tol_f = 1e-8;
  opts.tol_x = 1e-8;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_FALSE(result.converged);
  EXPECT_LE(result.iterations, 5);
}

// Test convergence detection works
TEST(Optimization_NelderMead, ConvergenceDetection) {
  // Easy function that converges quickly
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return x(0) * x(0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 1.0;

  NelderMeadOptions<1> opts;
  opts.x0 = x0;
  opts.max_iterations = 1000;
  opts.tol_f = 1e-4;
  opts.tol_x = 1e-4;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.iterations, 1000); // Should converge before max
}

// Test with zero initial guess (exercises zero_term_delta)
TEST(Optimization_NelderMead, ZeroInitialGuess) {
  auto objective =
      [](Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
      };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  NelderMeadOptions<2> opts;
  opts.x0 = x0;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 0.5, 1e-3);
  EXPECT_NEAR(result.x(1), 0.5, 1e-3);
}

// Test with 3 variables
TEST(Optimization_NelderMead, ThreeVariables) {
  auto objective =
      [](Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0) +
               (x(2) - 3.0) * (x(2) - 3.0);
      };

  Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;
  x0(2) = 0.0;

  NelderMeadOptions<3> opts;
  opts.x0 = x0;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-3);
  EXPECT_NEAR(result.x(1), 2.0, 1e-3);
  EXPECT_NEAR(result.x(2), 3.0, 1e-3);
}

// Test result structure fields
TEST(Optimization_NelderMead, ResultStructureComplete) {
  auto objective =
      [](Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x) {
        return x(0) * x(0);
      };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 5.0;

  NelderMeadOptions<1> opts;
  opts.x0 = x0;

  auto result = optimize(NelderMeadSimplex{}, objective, opts);

  // Check all fields are populated
  EXPECT_EQ(result.x.extent(0), 1);
  EXPECT_GE(result.iterations, 1);
  // min_value should be close to 0 for x^2 minimized
  EXPECT_LT(result.min_value, 1e-6);
}
