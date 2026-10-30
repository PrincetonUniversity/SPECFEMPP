#include "specfem/fk_plugin/utilities/complex_matrix.hpp"
#include <gtest/gtest.h>

#include <complex>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Cx = std::complex<double>;
using Mat2 = specfem::fk_plugin::ComplexMatrix<2>;
using Mat4 = specfem::fk_plugin::ComplexMatrix<4>;
using Vec2 = specfem::fk_plugin::ComplexVector<2>;

// ---------------------------------------------------------------------------
// A * identity == A  and  identity * A == A  (4x4)
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, MultiplyByIdentity4x4) {
  constexpr double kTol = 1.0e-10;

  Mat4 A;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      A(i, j) =
          Cx(static_cast<double>(i * 4 + j + 1), static_cast<double>(i - j));
    }
  }

  Mat4 I = Mat4::identity();
  Mat4 AI = A * I;
  Mat4 IA = I * A;

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
