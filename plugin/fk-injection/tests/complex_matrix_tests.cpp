#include "specfem/fk_plugin/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Cx = Kokkos::complex<double>;
using Mat2 = specfem::fk_plugin::ComplexMatrix<2>;
using Mat4 = specfem::fk_plugin::ComplexMatrix<4>;
using Vec2 = specfem::fk_plugin::ComplexVector<2>;

// Device kernels are expressed as named functors rather than extended
// __device__ lambdas: nvcc forbids an extended lambda in gtest's private
// TestBody(), so the lambdas must live in an ordinary type.  File-unique
// namespace (not anonymous) keeps the unity build ODR-safe.
namespace complex_matrix_tests_impl {

struct MultiplyByIdentity4x4 {
  Kokkos::View<double *> d_err;
  KOKKOS_FUNCTION void operator()(const int) const {
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
        double e1 = Kokkos::abs(AI(i, j) - A(i, j));
        double e2 = Kokkos::abs(IA(i, j) - A(i, j));
        if (e1 > max_err)
          max_err = e1;
        if (e2 > max_err)
          max_err = e2;
      }
    }
    d_err(0) = max_err;
  }
};

struct MatVec2x2 {
  Kokkos::View<double *> d_re;
  Kokkos::View<double *> d_im;
  KOKKOS_FUNCTION void operator()(const int) const {
    Mat2 A;
    A(0, 0) = Cx(1.0, 0.0);
    A(0, 1) = Cx(0.0, 1.0);
    A(1, 0) = Cx(2.0, 0.0);
    A(1, 1) = Cx(0.0, -1.0);

    Vec2 v;
    v[0] = Cx(1.0, 0.0);
    v[1] = Cx(0.0, 1.0);

    Vec2 r = A * v;
    d_re(0) = r[0].real();
    d_im(0) = r[0].imag();
    d_re(1) = r[1].real();
    d_im(1) = r[1].imag();
  }
};

} // namespace complex_matrix_tests_impl

// ---------------------------------------------------------------------------
// A * identity == A  and  identity * A == A  (4x4)
// ---------------------------------------------------------------------------

TEST(ComplexMatrix, MultiplyByIdentity4x4) {
  constexpr double kTol = 1.0e-10;
  // Allocate device storage for 16 entries (flattened results).
  Kokkos::View<double *> d_err("err", 1);
  Kokkos::deep_copy(d_err, 0.0);

  Kokkos::parallel_for(
      "cm_identity_4x4", Kokkos::RangePolicy<>(0, 1),
      complex_matrix_tests_impl::MultiplyByIdentity4x4{ d_err });
  Kokkos::fence();

  auto h_err = Kokkos::create_mirror_view(d_err);
  Kokkos::deep_copy(h_err, d_err);
  EXPECT_LT(h_err(0), kTol);
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
  Kokkos::View<double *> d_re("re", 2);
  Kokkos::View<double *> d_im("im", 2);

  Kokkos::parallel_for("cm_matvec", Kokkos::RangePolicy<>(0, 1),
                       complex_matrix_tests_impl::MatVec2x2{ d_re, d_im });
  Kokkos::fence();

  auto h_re = Kokkos::create_mirror_view(d_re);
  auto h_im = Kokkos::create_mirror_view(d_im);
  Kokkos::deep_copy(h_re, d_re);
  Kokkos::deep_copy(h_im, d_im);

  EXPECT_NEAR(h_re(0), 0.0, kTol);
  EXPECT_NEAR(h_im(0), 0.0, kTol);
  EXPECT_NEAR(h_re(1), 3.0, kTol);
  EXPECT_NEAR(h_im(1), 0.0, kTol);
}
