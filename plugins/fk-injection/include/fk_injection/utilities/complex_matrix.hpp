#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <complex>
#include <concepts>
#include <type_traits>

namespace fk_injection {
namespace impl {

// Provides mdspan, extents and layouts; std once C++23 is required.
namespace mdspan_namespace = Kokkos;

} // namespace impl

/**
 * @brief Fixed-size N×N complex matrix.
 *
 * A flat stack array indexed through an mdspan row-major layout mapping, so
 * the object is a register/stack local with no heap allocation. Element access
 * is @c m(i, j); @c to_mdspan returns the matching mdspan. The arithmetic is
 * free functions. Intended for small matrices created many times per
 * frequency (e.g. the 4×4 FK layer propagators).
 *
 * @tparam N Matrix dimension (rows == columns == N).
 * @tparam Scalar Element type; defaults to @c std::complex<double>.
 */
template <int N, typename Scalar = std::complex<double>> struct ComplexMatrix {
  using extents_type =
      fk_injection::impl::mdspan_namespace::extents<int, N,
                                                    N>; ///< Static N×N
                                                        ///< extents.
  using layout_type =
      fk_injection::impl::mdspan_namespace::layout_right; ///< Row-major.
  using mapping_type =
      typename layout_type::template mapping<extents_type>; ///< Maps (i, j) to
                                                            ///< the storage
                                                            ///< offset.
  using mdspan_type = fk_injection::impl::mdspan_namespace::mdspan<
      Scalar, extents_type, layout_type>; ///< Mutable mdspan over the storage.
  using const_mdspan_type =
      fk_injection::impl::mdspan_namespace::mdspan<const Scalar, extents_type,
                                                   layout_type>; ///< Read-only
                                                                 ///< mdspan
                                                                 ///< over the
                                                                 ///< storage.

  std::array<Scalar, N * N> values{}; ///< Storage, zero-initialised.

  /** @brief Zero matrix. */
  ComplexMatrix() = default;

  /**
   * @brief Fill element (i, j) with @c generator(i, j).
   *
   * @code
   * const ComplexMatrix<2> identity(
   *     [](int i, int j) { return i == j ? 1.0 : 0.0; });
   * @endcode
   *
   * @tparam Generator Callable as @c generator(int, int), returning a value
   *                   convertible to @c Scalar.
   * @param generator Element generator.
   */
  template <typename Generator>
    requires std::invocable<const Generator &, int, int>
  explicit ComplexMatrix(const Generator &generator) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        (*this)(i, j) = Scalar(generator(i, j));
      }
    }
  }

  /**
   * @brief Element access (mutable).
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Reference to element (i, j).
   */
  Scalar &operator()(const int i, const int j) {
    return values[mapping_type{}(i, j)];
  }

  /**
   * @brief Element access (read-only).
   * @param i Row index (0-based).
   * @param j Column index (0-based).
   * @return Const reference to element (i, j).
   */
  const Scalar &operator()(const int i, const int j) const {
    return values[mapping_type{}(i, j)];
  }

  /**
   * @brief Mutable mdspan over the storage.
   * @return mdspan with this matrix's extents and layout.
   */
  mdspan_type to_mdspan() { return mdspan_type(values.data()); }

  /**
   * @brief Read-only mdspan over the storage.
   * @return mdspan with this matrix's extents and layout.
   */
  const_mdspan_type to_mdspan() const {
    return const_mdspan_type(values.data());
  }
};

/**
 * @brief Length-N complex vector.
 *
 * @tparam N Vector length.
 * @tparam Scalar Element type; defaults to @c std::complex<double>.
 */
template <int N, typename Scalar = std::complex<double>>
using ComplexVector = std::array<Scalar, N>;

/**
 * @brief Matrix–matrix product.
 * @param left Left factor.
 * @param right Right factor.
 * @return @c left * right.
 */
template <int N, typename Scalar>
ComplexMatrix<N, Scalar> operator*(const ComplexMatrix<N, Scalar> &left,
                                   const ComplexMatrix<N, Scalar> &right) {
  ComplexMatrix<N, Scalar> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Scalar sum(0.0, 0.0);
      for (int k = 0; k < N; ++k) {
        sum += left(i, k) * right(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

/**
 * @brief Matrix–vector product.
 * @param matrix Matrix.
 * @param vector Column vector of length N (N is deduced from @p matrix only).
 * @return @c matrix * vector.
 */
template <int N, typename Scalar>
ComplexVector<N, Scalar>
operator*(const ComplexMatrix<N, Scalar> &matrix,
          const std::type_identity_t<ComplexVector<N, Scalar>> &vector) {
  ComplexVector<N, Scalar> result;
  for (int i = 0; i < N; ++i) {
    Scalar sum(0.0, 0.0);
    for (int j = 0; j < N; ++j) {
      sum += matrix(i, j) * vector[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * @brief In-place scalar scaling.
 * @param matrix Matrix to scale.
 * @param factor Scalar multiplier.
 * @return @p matrix after scaling.
 */
template <int N, typename Scalar>
ComplexMatrix<N, Scalar> &operator*=(ComplexMatrix<N, Scalar> &matrix,
                                     const Scalar &factor) {
  for (auto &value : matrix.values) {
    value *= factor;
  }
  return matrix;
}

} // namespace fk_injection
