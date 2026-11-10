#include "fk_injection/utilities/complex_matrix.hpp"
#include <gtest/gtest.h>

#include <complex>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Cx = std::complex<double>;
using Mat2 = fk_injection::ComplexMatrix<2>;
using Mat4 = fk_injection::ComplexMatrix<4>;
using Vec2 = fk_injection::ComplexVector<2>;

// ---------------------------------------------------------------------------
// Default construction is the zero matrix; the generator fills (i, j) in
// row-major order, the same element to_mdspan() addresses
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, DefaultIsZero) {
  const Mat4 zero;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(zero(i, j), Cx(0.0, 0.0));
    }
  }
}

TEST(ComplexMatrix, GeneratorFillsRowMajor) {
  const Mat2 matrix(
      [](int i, int j) { return Cx(static_cast<double>(10 * i + j), 1.0); });
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(matrix(i, j), Cx(static_cast<double>(10 * i + j), 1.0));
      EXPECT_EQ(matrix.values[i * 2 + j], matrix(i, j));
      EXPECT_EQ(matrix.to_mdspan()(i, j), matrix(i, j));
    }
  }
}

// ---------------------------------------------------------------------------
// A * identity == A  and  identity * A == A  (4x4)
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, MultiplyByIdentity4x4) {
  constexpr double kTol = 1.0e-10;

  const Mat4 A([](int i, int j) {
    return Cx(static_cast<double>(i * 4 + j + 1), static_cast<double>(i - j));
  });
  const Mat4 I([](int i, int j) { return i == j ? 1.0 : 0.0; });

  const Mat4 AI = A * I;
  const Mat4 IA = I * A;

  double max_err = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double e1 = std::abs(AI(i, j) - A(i, j));
      double e2 = std::abs(IA(i, j) - A(i, j));
      if (e1 > max_err)
        max_err = e1;
      if (e2 > max_err)
        max_err = e2;
    }
  }
  EXPECT_LT(max_err, kTol);
}

// ---------------------------------------------------------------------------
// Mat-vec: known 2x2 times known vector
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, MatVec2x2) {
  // A = [[1+0i, 0+1i], [2+0i, 0-1i]]
  // v = [1+0i, 0+1i]
  // A*v = [(1)(1)+(0+1i)(0+1i), (2)(1)+(0-1i)(0+1i)]
  //     = [1 + i^2, 2 - i^2]
  //     = [1 - 1, 2 + 1]
  //     = [0, 3]
  constexpr double kTol = 1.0e-10;

  Mat2 A;
  A(0, 0) = Cx(1.0, 0.0);
  A(0, 1) = Cx(0.0, 1.0);
  A(1, 0) = Cx(2.0, 0.0);
  A(1, 1) = Cx(0.0, -1.0);

  Vec2 v;
  v[0] = Cx(1.0, 0.0);
  v[1] = Cx(0.0, 1.0);

  Vec2 r = A * v;

  EXPECT_NEAR(r[0].real(), 0.0, kTol);
  EXPECT_NEAR(r[0].imag(), 0.0, kTol);
  EXPECT_NEAR(r[1].real(), 3.0, kTol);
  EXPECT_NEAR(r[1].imag(), 0.0, kTol);
}

// ---------------------------------------------------------------------------
// In-place scaling multiplies every element
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, ScaleInPlace) {
  Mat2 matrix([](int i, int j) { return Cx(1.0 + i, j); });
  matrix *= Cx(0.0, 2.0);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(matrix(i, j), Cx(1.0 + i, j) * Cx(0.0, 2.0));
    }
  }
}
