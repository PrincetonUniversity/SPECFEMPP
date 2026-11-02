#include "specfem/optimization.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <gtest/gtest.h>

using specfem::optimization::NelderMeadOptions;
using specfem::optimization::NelderMeadSimplex;
using specfem::optimization::optimize;

TEST(Optimization_NelderMeadGlobe, MinimizesQuadratic) {
  using View = Kokkos::View<double[2], Kokkos::LayoutRight, Kokkos::HostSpace>;
  NelderMeadOptions<2, double, true> opts;
  opts.x0 = View("x0");
  opts.x0(0) = 0.5;
  opts.x0(1) = 0.5;
  const auto result = optimize(
      NelderMeadSimplex{},
      [](const View &x) {
        return (x(0) - 1) * (x(0) - 1) + (x(1) - 2) * (x(1) - 2);
      },
      opts);
  EXPECT_TRUE(result.converged);
  EXPECT_NEAR(result.x(0), 1.0, 1.e-2);
  EXPECT_NEAR(result.x(1), 2.0, 1.e-2);
  EXPECT_LE(result.function_evaluations, 400);
}

TEST(Optimization_NelderMeadGlobe,
     CountsInitialAndShrinkEvaluationsAtBudgetBoundary) {
  using View = Kokkos::View<double[2], Kokkos::LayoutRight, Kokkos::HostSpace>;
  NelderMeadOptions<2, double, true> opts;
  opts.x0 = View("x0");
  Kokkos::deep_copy(opts.x0, 1.0);
  opts.max_function_evaluations = 4;
  opts.tol_x = 0;
  opts.tol_f = 0;
  int calls = 0;
  const auto result = optimize(
      NelderMeadSimplex{},
      [&](const View &) {
        ++calls;
        return 1.0;
      },
      opts);
  // Initial simplex (3), reflection (1), inside contraction (1), shrink (2).
  // As in globe, a started iteration completes even when it crosses the budget.
  EXPECT_EQ(calls, 7);
  EXPECT_EQ(result.function_evaluations, calls);
  EXPECT_EQ(result.iterations, 2);
  EXPECT_FALSE(result.converged);
}

TEST(Optimization_NelderMeadGlobe, ConvergenceIncludesToleranceBoundary) {
  using View = Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace>;
  NelderMeadOptions<1, double, true> opts;
  opts.x0 = View("x0");
  opts.x0(0) = 1;
  // Initial simplex size, using globe's promoted 0.05 literal.
  opts.tol_x = (1.0 + static_cast<double>(0.05f)) * 1.0 - 1.0;
  opts.tol_f = 0;
  const auto result =
      optimize(NelderMeadSimplex{}, [](const View &) { return 1.0; }, opts);
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(result.function_evaluations, 2);
}

TEST(Optimization_NelderMeadGlobe, IterationBudgetIncludesInitialSimplex) {
  using View = Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace>;
  NelderMeadOptions<1, double, true> opts;
  opts.x0 = View("x0");
  opts.x0(0) = 1;
  opts.max_iterations = 1;
  const auto result = optimize(
      NelderMeadSimplex{}, [](const View &x) { return x(0) * x(0); }, opts);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_EQ(result.function_evaluations, 2);
}

TEST(Optimization_NelderMeadGlobe, ZeroCoordinateUsesZeroTermDelta) {
  using View = Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace>;
  NelderMeadOptions<1, double, true> opts;
  opts.x0 = View("x0");
  opts.x0(0) = 0;
  opts.max_iterations = 1;
  std::array<double, 2> evaluated{};
  int calls = 0;
  optimize(
      NelderMeadSimplex{},
      [&](const View &x) {
        evaluated[calls++] = x(0);
        return x(0);
      },
      opts);
  ASSERT_EQ(calls, 2);
  EXPECT_NEAR(evaluated[1], static_cast<double>(0.00025f), 0.0);
}

TEST(Optimization_NelderMeadGlobe, HeapSortReturnsAscendingPermutation) {
  std::array<double, 5> x = { 0, 3, 1, 4, 2 }; // 1-based: x[1..4]
  std::array<int, 5> y{};
  specfem::optimization::globe_heap_sort<4>(x, y);
  for (int i = 1; i <= 4; ++i)
    EXPECT_NEAR(x[i], static_cast<double>(i), 0.0);
  EXPECT_EQ(y[1], 2);
  EXPECT_EQ(y[2], 4);
  EXPECT_EQ(y[3], 1);
  EXPECT_EQ(y[4], 3);
}

TEST(Optimization_NelderMeadGlobe, SharedSolverPreservesTinyNonzeroCoordinate) {
  using view_type =
      Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace>;
  specfem::optimization::NelderMeadOptions<1, double, true> opts;
  opts.x0 = view_type("x0");
  opts.x0(0) = 1.e-12;
  opts.max_iterations = 1;
  int calls = 0;
  double perturbed = 0;
  const auto result = optimize(
      specfem::optimization::NelderMeadSimplex{},
      [&](const view_type &x) {
        if (++calls == 2)
          perturbed = x(0);
        return x(0);
      },
      opts);
  EXPECT_NEAR(perturbed, (1.0 + static_cast<double>(0.05f)) * 1.e-12, 0.0);
  EXPECT_EQ(result.function_evaluations, 2);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_FALSE(result.converged);
}

TEST(Optimization_NelderMeadGlobe,
     StandardSolverCountsCompletedShrinkAtBudget) {
  using view_type =
      Kokkos::View<double[1], Kokkos::LayoutRight, Kokkos::HostSpace>;
  specfem::optimization::NelderMeadOptions<1, double> opts;
  opts.x0 = view_type("x0");
  opts.x0(0) = 1;
  opts.tol_f = 0;
  opts.tol_x = 0;
  opts.max_function_evaluations = 3;
  int calls = 0;
  const auto result = optimize(
      specfem::optimization::NelderMeadSimplex{},
      [&](const view_type &) {
        ++calls;
        return 1.0;
      },
      opts);
  EXPECT_EQ(calls, 5);
  EXPECT_EQ(result.function_evaluations, calls);
  EXPECT_EQ(result.iterations, 1);
  EXPECT_FALSE(result.converged);
}
