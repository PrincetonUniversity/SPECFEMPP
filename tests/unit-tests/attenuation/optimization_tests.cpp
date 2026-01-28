#include "specfem/attenuation/impl/optimization.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::attenuation::impl::fmin_search;
using specfem::attenuation::impl::OptimizationResult;

// Test quadratic minimization: f(x) = (x - 3)^2
// Minimum at x = 3
TEST(Optimization_FminSearch, QuadraticMinimization1D) {
  auto objective = [](Kokkos::View<type_real[1], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    return (x(0) - 3.0) * (x(0) - 3.0);
  };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;

  auto result = fmin_search<1>(objective, x0);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 3.0, 1e-3);
  EXPECT_NEAR(result.min_value, 0.0, 1e-6);
}

// Test 2D quadratic: f(x,y) = (x - 1)^2 + (y - 2)^2
// Minimum at (1, 2)
TEST(Optimization_FminSearch, QuadraticMinimization2D) {
  auto objective = [](Kokkos::View<type_real[2], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0);
  };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  auto result = fmin_search<2>(objective, x0);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-3);
  EXPECT_NEAR(result.x(1), 2.0, 1e-3);
  EXPECT_NEAR(result.min_value, 0.0, 1e-6);
}

// Test Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// This is a classic optimization benchmark with minimum at (1, 1)
TEST(Optimization_FminSearch, RosenbrockFunction) {
  auto objective = [](Kokkos::View<type_real[2], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    type_real a = 1.0 - x(0);
    type_real b = x(1) - x(0) * x(0);
    return a * a + 100.0 * b * b;
  };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  // Rosenbrock requires more iterations
  auto result = fmin_search<2>(objective, x0, 2000, 1e-6, 1e-6);

  // Rosenbrock is notoriously difficult, allow some tolerance
  EXPECT_NEAR(result.x(0), 1.0, 0.05);
  EXPECT_NEAR(result.x(1), 1.0, 0.05);
  EXPECT_LT(result.min_value, 0.01);
}

// Test that max iterations limit is respected
TEST(Optimization_FminSearch, RespectsMaxIterations) {
  auto objective = [](Kokkos::View<type_real[2], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    // Rosenbrock - hard to converge
    type_real a = 1.0 - x(0);
    type_real b = x(1) - x(0) * x(0);
    return a * a + 100.0 * b * b;
  };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = -1.0;
  x0(1) = 1.0;

  // Use very few iterations - should not converge
  auto result = fmin_search<2>(objective, x0, 5, 1e-8, 1e-8);

  EXPECT_FALSE(result.converged);
  EXPECT_LE(result.iterations, 5);
}

// Test convergence detection works
TEST(Optimization_FminSearch, ConvergenceDetection) {
  // Easy function that converges quickly
  auto objective = [](Kokkos::View<type_real[1], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) { return x(0) * x(0); };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 1.0;

  auto result = fmin_search<1>(objective, x0, 1000, 1e-4, 1e-4);

  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.iterations, 1000); // Should converge before max
}

// Test with zero initial guess (exercises zero_term_delta)
TEST(Optimization_FminSearch, ZeroInitialGuess) {
  auto objective = [](Kokkos::View<type_real[2], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    return (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
  };

  Kokkos::View<type_real[2], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;

  auto result = fmin_search<2>(objective, x0);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 0.5, 1e-3);
  EXPECT_NEAR(result.x(1), 0.5, 1e-3);
}

// Test with 3 variables
TEST(Optimization_FminSearch, ThreeVariables) {
  auto objective = [](Kokkos::View<type_real[3], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) {
    return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) - 2.0) * (x(1) - 2.0) +
           (x(2) - 3.0) * (x(2) - 3.0);
  };

  Kokkos::View<type_real[3], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 0.0;
  x0(1) = 0.0;
  x0(2) = 0.0;

  auto result = fmin_search<3>(objective, x0);

  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1e-3);
  EXPECT_NEAR(result.x(1), 2.0, 1e-3);
  EXPECT_NEAR(result.x(2), 3.0, 1e-3);
}

// Test result structure fields
TEST(Optimization_FminSearch, ResultStructureComplete) {
  auto objective = [](Kokkos::View<type_real[1], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> x) { return x(0) * x(0); };

  Kokkos::View<type_real[1], Kokkos::LayoutRight, Kokkos::HostSpace> x0("x0");
  x0(0) = 5.0;

  auto result = fmin_search<1>(objective, x0);

  // Check all fields are populated
  EXPECT_EQ(result.x.extent(0), 1);
  EXPECT_GE(result.iterations, 1);
  // min_value should be close to 0 for x^2 minimized
  EXPECT_LT(result.min_value, 1e-6);
}
