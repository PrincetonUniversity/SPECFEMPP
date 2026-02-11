#include "../SPECFEM_Environment.hpp"
#include "specfem/utilities.hpp"
#include "test_macros.hpp"
#include <cmath>
#include <gtest/gtest.h>

using specfem::utilities::is_close;
using specfem::utilities::logspace;

TEST(UtilitiesLogspace, BasicRange) {
  constexpr int N = 5;
  auto result = logspace<N>(1.0, 10000.0);

  EXPECT_TRUE(is_close(result(0), type_real(1.0)))
      << expected_got(type_real(1.0), result(0));
  EXPECT_TRUE(is_close(result(N - 1), type_real(10000.0)))
      << expected_got(type_real(10000.0), result(N - 1));
}

TEST(UtilitiesLogspace, TwoPoints) {
  auto result = logspace<2>(10.0, 1000.0);

  EXPECT_TRUE(is_close(result(0), type_real(10.0)))
      << expected_got(type_real(10.0), result(0));
  EXPECT_TRUE(is_close(result(1), type_real(1000.0)))
      << expected_got(type_real(1000.0), result(1));
}

TEST(UtilitiesLogspace, LogarithmicSpacing) {
  constexpr int N = 5;
  // Expected: 1, 10, 100, 1000, 10000
  auto result = logspace<N>(1.0, 10000.0);

  const type_real expected_values[N] = { 1.0, 10.0, 100.0, 1000.0, 10000.0 };
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(is_close(result(i), expected_values[i]))
        << expected_got(expected_values[i], result(i));
  }
}

TEST(UtilitiesLogspace, EqualMinMax) {
  constexpr int N = 4;
  auto result = logspace<N>(100.0, 100.0);

  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(is_close(result(i), type_real(100.0)))
        << expected_got(type_real(100.0), result(i));
  }
}

TEST(UtilitiesLogspace, FractionalValues) {
  constexpr int N = 3;
  // Expected: 0.01, 0.1, 1.0
  auto result = logspace<N>(0.01, 1.0);

  const type_real expected_values[N] = { 0.01, 0.1, 1.0 };
  for (int i = 0; i < N; ++i) {
    EXPECT_TRUE(is_close(result(i), expected_values[i]))
        << expected_got(expected_values[i], result(i));
  }
}

TEST(UtilitiesLogspace, InvalidInputZeroMin) {
  LOCAL_EXPECT_THROW(logspace<3>(0.0, 10.0), std::invalid_argument);
}

TEST(UtilitiesLogspace, InvalidInputNegativeMin) {
  LOCAL_EXPECT_THROW(logspace<3>(-1.0, 10.0), std::invalid_argument);
}

TEST(UtilitiesLogspace, InvalidInputNegativeMax) {
  LOCAL_EXPECT_THROW(logspace<3>(1.0, -10.0), std::invalid_argument);
}

TEST(UtilitiesLogspace, InvalidInputZeroMax) {
  LOCAL_EXPECT_THROW(logspace<3>(1.0, 0.0), std::invalid_argument);
}

TEST(UtilitiesLogspace, MonotonicallyIncreasing) {
  constexpr int N = 10;
  auto result = logspace<N>(0.001, 1000.0);

  for (int i = 1; i < N; ++i) {
    EXPECT_GT(result(i), result(i - 1));
  }
}

TEST(UtilitiesLogspace, LogSpacingUniform) {
  // In log10 space, the points should be uniformly spaced
  constexpr int N = 6;
  auto result = logspace<N>(1.0, 100000.0);

  const type_real expected_log_step =
      (std::log10(type_real(100000.0)) - std::log10(type_real(1.0))) / (N - 1);

  for (int i = 1; i < N; ++i) {
    type_real log_step = std::log10(result(i)) - std::log10(result(i - 1));
    EXPECT_TRUE(is_close(log_step, expected_log_step))
        << expected_got(expected_log_step, log_step);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
