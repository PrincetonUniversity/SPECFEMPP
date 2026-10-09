#pragma once

#include <Kokkos_Array.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

namespace specfem {
namespace utilities {

/**
 * @brief Dense N×N matrix with complex scalar entries stored in row-major
 * order.
 *
 * Uses a flat @c Kokkos::Array so the object lives in registers on the device.
 * All member functions are @c KOKKOS_INLINE_FUNCTION and therefore callable
 * from inside a Kokkos kernel.
 *
 * Intended for small fixed-size matrices created per thread inside a kernel
 * (e.g. the 4×4 FK layer propagators). A @c Kokkos::View is heap-allocated
 * and reference-counted, so it cannot be allocated per thread in a kernel.
 *
 * @tparam N Matrix dimension (rows == columns == N).
 * @tparam Scalar Element type; defaults to @c Kokkos::complex<double>.
 */
template <int N, typename Scalar = Kokkos::complex<double>>
struct ComplexMatrix {
  Kokkos::Array<Scalar, N * N> data; ///< Row-major flat storage.

  /** @brief Zero-initialise every element. */
  KOKKOS_INLINE_FUNCTION ComplexMatrix() {
    for (int k = 0; k < N * N; ++k) {
      data[k] = Scalar(0.0, 0.0);
    }
  }

  /**
   * @brief Element access (mutable).
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Reference to @c data[i*N + j].
   */
  KOKKOS_INLINE_FUNCTION Scalar &operator()(int i, int j) {
    return data[i * N + j];
  }

  /**
   * @brief Element access (read-only).
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Const reference to @c data[i*N + j].
   */
  KOKKOS_INLINE_FUNCTION const Scalar &operator()(int i, int j) const {
    return data[i * N + j];
  }

  /**
   * @brief Construct the N×N identity matrix.
   * @return Identity matrix with diagonal entries @c Scalar(1,0).
   */
  KOKKOS_INLINE_FUNCTION static ComplexMatrix identity() {
    ComplexMatrix result; // zero-initialised by default ctor
    for (int i = 0; i < N; ++i) {
      result(i, i) = Scalar(1.0, 0.0);
    }
    return result;
  }

  /**
   * @brief Matrix–matrix product.
   * @param rhs Right-hand-side matrix.
   * @return A fresh matrix equal to @c (*this) * rhs.
   */
  KOKKOS_INLINE_FUNCTION ComplexMatrix
  operator*(const ComplexMatrix &rhs) const {
    ComplexMatrix result;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        Scalar sum(0.0, 0.0);
        for (int k = 0; k < N; ++k) {
          sum = sum + (*this)(i, k) * rhs(k, j);
        }
        result(i, j) = sum;
      }
    }
    return result;
  }

  /**
   * @brief Matrix–vector product.
   * @param v Column vector of length N.
   * @return Result vector @c (*this) * v.
   */
  KOKKOS_INLINE_FUNCTION Kokkos::Array<Scalar, N>
  operator*(const Kokkos::Array<Scalar, N> &v) const {
    Kokkos::Array<Scalar, N> result;
    for (int i = 0; i < N; ++i) {
      Scalar sum(0.0, 0.0);
      for (int j = 0; j < N; ++j) {
        sum = sum + (*this)(i, j) * v[j];
      }
      result[i] = sum;
    }
    return result;
  }

  /**
   * @brief In-place scalar scaling.
   * @param s Scalar multiplier.
   * @return Reference to @c *this after scaling.
   */
  KOKKOS_INLINE_FUNCTION ComplexMatrix &operator*=(Scalar s) {
    for (int k = 0; k < N * N; ++k) {
      data[k] = data[k] * s;
    }
    return *this;
  }

  /**
   * @brief Compute the matrix inverse via Gauss–Jordan elimination with
   * partial (column) pivoting.
   *
   * Works for any N; pivot selection uses the largest @c Kokkos::abs among
   * remaining rows.  Returns a zero matrix when the matrix is (near-)singular.
   *
   * @param ok Set to @c true on success, @c false when the leading pivot
   *           magnitude is ≤ @p tolerance (singular matrix).
   * @param tolerance Pivot-magnitude threshold below which the matrix is
   *                  considered singular (default 1e-12).
   * @return Inverse matrix, or the zero matrix if @p ok is @c false.
   */
  KOKKOS_INLINE_FUNCTION ComplexMatrix
  inverse(bool &ok, double tolerance = 1.0e-12) const {
    // Augmented matrix [A | I] stored as two NxN blocks.
    ComplexMatrix aug(*this);       // left block: copy of A
    ComplexMatrix inv = identity(); // right block: starts as I

    for (int col = 0; col < N; ++col) {
      // Find pivot row with largest absolute value in this column.
      int pivot_row = col;
      double max_abs = Kokkos::abs(aug(col, col));
      for (int row = col + 1; row < N; ++row) {
        double candidate = Kokkos::abs(aug(row, col));
        if (candidate > max_abs) {
          max_abs = candidate;
          pivot_row = row;
        }
      }

      if (max_abs <= tolerance) {
        ok = false;
        return ComplexMatrix(); // zero matrix
      }

      // Swap rows if necessary.
      if (pivot_row != col) {
        for (int j = 0; j < N; ++j) {
          Scalar tmp_a = aug(col, j);
          aug(col, j) = aug(pivot_row, j);
          aug(pivot_row, j) = tmp_a;

          Scalar tmp_i = inv(col, j);
          inv(col, j) = inv(pivot_row, j);
          inv(pivot_row, j) = tmp_i;
        }
      }

      // Scale pivot row so the diagonal becomes 1.
      Scalar diag_inv = Scalar(1.0, 0.0) / aug(col, col);
      for (int j = 0; j < N; ++j) {
        aug(col, j) = aug(col, j) * diag_inv;
        inv(col, j) = inv(col, j) * diag_inv;
      }

      // Eliminate all other rows.
      for (int row = 0; row < N; ++row) {
        if (row == col)
          continue;
        Scalar factor = aug(row, col);
        for (int j = 0; j < N; ++j) {
          aug(row, j) = aug(row, j) - factor * aug(col, j);
          inv(row, j) = inv(row, j) - factor * inv(col, j);
        }
      }
    }

    ok = true;
    return inv;
  }
};

/**
 * @brief Alias for a length-N complex vector as a @c Kokkos::Array.
 *
 * @tparam N Vector length.
 * @tparam Scalar Element type; defaults to @c Kokkos::complex<double>.
 */
template <int N, typename Scalar = Kokkos::complex<double>>
using ComplexVector = Kokkos::Array<Scalar, N>;

} // namespace utilities
} // namespace specfem
