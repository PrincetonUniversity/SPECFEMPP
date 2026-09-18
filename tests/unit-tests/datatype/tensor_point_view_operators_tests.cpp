#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {

using T = double;

/**
 * @brief Test matrix multiplication operator for TensorPointViewType
 *
 * Tests the (M×N) * (N×K) → M×K matrix multiplication
 */
TEST(TensorPointViewTypeOperators, MatrixMultiplication2x3_3x2) {
  // Test (2×3) * (3×2) → (2×2) matrix product
  // A = [[1, 2, 3],     B = [[1, 2],
  //      [4, 5, 6]]          [3, 4],
  //                           [5, 6]]
  // Expected C = [[22, 28],  (1*1 + 2*3 + 3*5 = 22, 1*2 + 2*4 + 3*6 = 28)
  //               [49, 64]]  (4*1 + 5*3 + 6*5 = 49, 4*2 + 5*4 + 6*6 = 64)

  auto A = specfem::datatype::TensorPointViewType<T, 2, 3, false>();
  auto B = specfem::datatype::TensorPointViewType<T, 3, 2, false>();

  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(0, 2) = 3.0;
  A(1, 0) = 4.0;
  A(1, 1) = 5.0;
  A(1, 2) = 6.0;

  B(0, 0) = 1.0;
  B(0, 1) = 2.0;
  B(1, 0) = 3.0;
  B(1, 1) = 4.0;
  B(2, 0) = 5.0;
  B(2, 1) = 6.0;

  auto C = A * B;

  EXPECT_DOUBLE_EQ(C(0, 0), 22.0);
  EXPECT_DOUBLE_EQ(C(0, 1), 28.0);
  EXPECT_DOUBLE_EQ(C(1, 0), 49.0);
  EXPECT_DOUBLE_EQ(C(1, 1), 64.0);
}

/**
 * @brief Test matrix multiplication for 3D case
 *
 * Tests (3×3) * (3×3) → (3×3) matrix product
 */
TEST(TensorPointViewTypeOperators, MatrixMultiplication3x3_3x3) {
  // Identity matrix multiplication: I * I = I
  auto I = specfem::datatype::TensorPointViewType<T, 3, 3, false>();
  I(0, 0) = 1.0;
  I(0, 1) = 0.0;
  I(0, 2) = 0.0;
  I(1, 0) = 0.0;
  I(1, 1) = 1.0;
  I(1, 2) = 0.0;
  I(2, 0) = 0.0;
  I(2, 1) = 0.0;
  I(2, 2) = 1.0;

  auto result = I * I;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double expected = (i == j) ? 1.0 : 0.0;
      EXPECT_DOUBLE_EQ(result(i, j), expected);
    }
  }
}

/**
 * @brief Test scalar multiplication operator
 *
 * Tests operator*= for element-wise scalar multiplication
 */
TEST(TensorPointViewTypeOperators, ScalarMultiplication) {
  auto A = specfem::datatype::TensorPointViewType<T, 2, 2, false>();
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(1, 0) = 3.0;
  A(1, 1) = 4.0;

  A *= 2.0;

  EXPECT_DOUBLE_EQ(A(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(A(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(A(1, 0), 6.0);
  EXPECT_DOUBLE_EQ(A(1, 1), 8.0);
}

/**
 * @brief Test matrix multiplication with scaling
 *
 * Verifies that matrix product followed by scalar multiplication
 * produces correct results
 */
TEST(TensorPointViewTypeOperators, MatrixMultiplicationWithScaling) {
  // Simple case: (2×2) * (2×2) * scalar
  auto A = specfem::datatype::TensorPointViewType<T, 2, 2, false>();
  A(0, 0) = 1.0;
  A(0, 1) = 2.0;
  A(1, 0) = 3.0;
  A(1, 1) = 4.0;

  auto B = specfem::datatype::TensorPointViewType<T, 2, 2, false>();
  B(0, 0) = 2.0;
  B(0, 1) = 0.0;
  B(1, 0) = 0.0;
  B(1, 1) = 2.0;

  auto C = A * B; // Multiply by 2x2 diagonal matrix
  C *= 0.5;       // Then scale by 0.5

  // A * B should give [[2, 4], [6, 8]]
  // Then C *= 0.5 should give [[1, 2], [3, 4]]
  EXPECT_DOUBLE_EQ(C(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(C(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(C(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(C(1, 1), 4.0);
}

} // namespace

// Note: Additional integration tests should verify mathematical equivalence
// with the original explicit stress transformation implementations
