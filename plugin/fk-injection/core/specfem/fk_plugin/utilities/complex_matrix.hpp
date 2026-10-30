#pragma once

#include <array>
#include <complex>

namespace specfem {
namespace fk_plugin {

/**
 * @brief Dense N×N matrix with complex scalar entries stored in row-major
 * order.
 *
 * Uses a flat @c std::array, so the object is a stack local with no heap
 * allocation.
 *
 * Intended for small fixed-size matrices created many times per frequency
 * (e.g. the 4×4 FK layer propagators).
 *
 * @tparam N Matrix dimension (rows == columns == N).
 * @tparam Scalar Element type; defaults to @c std::complex<double>.
 */
template <int N, typename Scalar = std::complex<double>> struct ComplexMatrix {
  std::array<Scalar, N * N> data; ///< Row-major flat storage.

  /** @brief Zero-initialise every element. */
  ComplexMatrix() {
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
  Scalar &operator()(int i, int j) { return data[i * N + j]; }

  /**
   * @brief Element access (read-only).
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Const reference to @c data[i*N + j].
   */
  const Scalar &operator()(int i, int j) const { return data[i * N + j]; }

  /**
   * @brief Construct the N×N identity matrix.
   * @return Identity matrix with diagonal entries @c Scalar(1,0).
   */
  static ComplexMatrix identity() {
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
  ComplexMatrix operator*(const ComplexMatrix &rhs) const {
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
  std::array<Scalar, N> operator*(const std::array<Scalar, N> &v) const {
    std::array<Scalar, N> result;
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
  ComplexMatrix &operator*=(Scalar s) {
    for (int k = 0; k < N * N; ++k) {
      data[k] = data[k] * s;
    }
    return *this;
  }
};

/**
 * @brief Alias for a length-N complex vector as a @c std::array.
 *
 * @tparam N Vector length.
 * @tparam Scalar Element type; defaults to @c std::complex<double>.
 */
template <int N, typename Scalar = std::complex<double>>
using ComplexVector = std::array<Scalar, N>;

} // namespace fk_plugin
} // namespace specfem
