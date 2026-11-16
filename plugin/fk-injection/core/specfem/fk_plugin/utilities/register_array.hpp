#pragma once

#include <Kokkos_Core.hpp>

#include <complex>
#include <concepts>
#include <cstddef>

namespace specfem {
namespace fk_plugin_impl {

// Provides extents and layouts; std once C++23 is required.
namespace mdspan_namespace = Kokkos;

} // namespace fk_plugin_impl

namespace fk_plugin {

/**
 * @brief Stack-allocated multi-dimensional array with a compile-time layout.
 *
 * Elements are addressed as @c a(i, ...) through the mdspan layout mapping.
 *
 * @tparam T Element type.
 * @tparam Extents Static array dimensions.
 * @tparam Layout Memory layout; row-major by default.
 */
template <typename T, typename Extents,
          typename Layout =
              specfem::fk_plugin_impl::mdspan_namespace::layout_right>
class RegisterArray {
private:
  using mapping = typename Layout::template mapping<Extents>;
  constexpr static std::size_t size = mapping().required_span_size();

public:
  using value_type = T; ///< Element type.

  /** @brief Zero-initialised array. */
  RegisterArray() = default;

  /**
   * @brief Construct from every element, in storage order.
   * @tparam Args Element types, convertible to @c T.
   * @param args One value per element.
   */
  template <typename... Args>
    requires(sizeof...(Args) == size && size > 1 &&
             (std::convertible_to<Args, T> && ...))
  RegisterArray(const Args &...args) : values_{ static_cast<T>(args)... } {}

  /**
   * @brief Fill element (i, j) of a rank-2 array with @c generator(i, j).
   *
   * @code
   * const ComplexMatrix<2> identity(
   *     [](int i, int j) { return i == j ? 1.0 : 0.0; });
   * @endcode
   *
   * @tparam Generator Callable as @c generator(int, int), returning a value
   *                   convertible to @c T.
   * @param generator Element generator.
   */
  template <typename Generator>
    requires(Extents::rank() == 2 &&
             std::invocable<const Generator &, int, int>)
  explicit RegisterArray(const Generator &generator) {
    const int rows = static_cast<int>(Extents::static_extent(0));
    const int columns = static_cast<int>(Extents::static_extent(1));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        (*this)(i, j) = T(generator(i, j));
      }
    }
  }

  /**
   * @brief Element access (mutable).
   * @tparam Index Index types, one per rank.
   * @param index Multi-dimensional index.
   * @return Reference to the element.
   */
  template <typename... Index>
    requires(sizeof...(Index) == Extents::rank())
  T &operator()(const Index &...index) {
    return values_[mapping()(index...)];
  }

  /**
   * @brief Element access (read-only).
   * @tparam Index Index types, one per rank.
   * @param index Multi-dimensional index.
   * @return Const reference to the element.
   */
  template <typename... Index>
    requires(sizeof...(Index) == Extents::rank())
  const T &operator()(const Index &...index) const {
    return values_[mapping()(index...)];
  }

private:
  T values_[size]{}; ///< Element storage.
};

/**
 * @brief N×N complex matrix, created many times per frequency (e.g. the 4×4
 * FK layer propagators).
 *
 * @tparam N Matrix dimension.
 */
template <std::size_t N>
using ComplexMatrix = RegisterArray<
    std::complex<double>,
    specfem::fk_plugin_impl::mdspan_namespace::extents<std::size_t, N, N>>;

/**
 * @brief Length-N complex vector.
 *
 * @tparam N Vector length.
 */
template <std::size_t N>
using ComplexVector = RegisterArray<
    std::complex<double>,
    specfem::fk_plugin_impl::mdspan_namespace::extents<std::size_t, N>>;

/**
 * @brief Matrix–matrix product.
 * @param left Left factor.
 * @param right Right factor.
 * @return @c left * right.
 */
template <std::size_t N>
ComplexMatrix<N> operator*(const ComplexMatrix<N> &left,
                           const ComplexMatrix<N> &right) {
  ComplexMatrix<N> result;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < N; ++k) {
        result(i, j) += left(i, k) * right(k, j);
      }
    }
  }
  return result;
}

/**
 * @brief Matrix–vector product.
 * @param matrix Matrix.
 * @param vector Column vector.
 * @return @c matrix * vector.
 */
template <std::size_t N>
ComplexVector<N> operator*(const ComplexMatrix<N> &matrix,
                           const ComplexVector<N> &vector) {
  ComplexVector<N> result;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      result(i) += matrix(i, j) * vector(j);
    }
  }
  return result;
}

/**
 * @brief In-place scalar scaling.
 * @param matrix Matrix to scale.
 * @param factor Scalar multiplier.
 * @return @p matrix after scaling.
 */
template <std::size_t N>
ComplexMatrix<N> &operator*=(ComplexMatrix<N> &matrix,
                             const std::complex<double> &factor) {
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      matrix(i, j) *= factor;
    }
  }
  return matrix;
}

} // namespace fk_plugin
} // namespace specfem
