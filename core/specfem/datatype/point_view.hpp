/**
 * @file point_view.hpp
 * @brief Point-based data storage types for GLL quadrature points
 */
#pragma once

#include "register_array.hpp"
#include "simd.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Stack-allocated vector storage for GLL quadrature points
 *
 * Stores scalar components at a single GLL point using register-based arrays
 * for optimal performance. Supports both scalar and SIMD operations.
 *
 * @tparam T Scalar data type (float, double)
 * @tparam Components Number of scalar components at point
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <typename T, std::size_t Components, bool UseSIMD>
struct VectorPointViewType
    : public RegisterArray<
          typename specfem::datatype::simd<T, UseSIMD>::datatype,
          Kokkos::extents<std::size_t, Components>, Kokkos::layout_left> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type =
      RegisterArray<typename specfem::datatype::simd<T, UseSIMD>::datatype,
                    Kokkos::extents<std::size_t, Components>,
                    Kokkos::layout_left>; ///< Underlying data type used to
                                          ///< store values
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using value_type =
      typename base_type::value_type; ///< Value type used to store
                                      ///< the elements of the array
  constexpr static bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int components = Components; ///< Number of scalar values at
                                                ///< the GLL point
  ///@}

  /**
   * @name Constructors and assignment operators
   */
  ///@{
  /// Inherit all base class constructors
  using base_type::base_type;
  ///@}

  /**
   * @brief Compute dot product with another vector
   * @param other Vector to compute dot product with
   * @return Dot product result
   */
  KOKKOS_INLINE_FUNCTION value_type
  operator*(const VectorPointViewType &other) const {
    constexpr int N = VectorPointViewType::components;
    value_type result{ 0.0 };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      result += (*this)(i)*other(i);
    }
    return result;
  }

  /**
   * @brief Multiply all components by scalar value
   * @param other Scalar value to multiply by
   * @return Reference to this object
   */
  KOKKOS_FORCEINLINE_FUNCTION constexpr auto &
  operator*=(const value_type &other) {
    constexpr int N = VectorPointViewType::components;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      (*this)(i) *= other;
    }
    return *this;
  }
};

/**
 * @brief Stack-allocated tensor storage for GLL quadrature points
 *
 * Stores multi-dimensional tensor components at a single GLL point using
 * register-based arrays. Optimized for small tensor sizes with SIMD support.
 *
 * @tparam T Scalar data type (float, double)
 * @tparam Components Number of tensor components
 * @tparam Dimensions Spatial dimensions of tensor
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <typename T, int Components, int Dimensions, bool UseSIMD>
struct TensorPointViewType
    : public RegisterArray<
          typename specfem::datatype::simd<T, UseSIMD>::datatype,
          Kokkos::extents<std::size_t, Components, Dimensions>,
          Kokkos::layout_left> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type =
      RegisterArray<typename specfem::datatype::simd<T, UseSIMD>::datatype,
                    Kokkos::extents<std::size_t, Components, Dimensions>,
                    Kokkos::layout_left>; ///< Underlying data type used to
                                          ///< store values
  using simd = specfem::datatype::simd<T, UseSIMD>; ///< SIMD data type
  using value_type =
      typename base_type::value_type; ///< Value type used to store
                                      ///< the elements of the array
  constexpr static bool using_simd =
      UseSIMD; ///< Use SIMD datatypes for the array. If false,
               ///< std::is_same<value_type, base_type>::value is true
  ///@}

  /**
   * @name Compile time constants
   *
   */
  ///@{
  constexpr static int components = Components; ///< Number of components of the
                                                ///< vector
  constexpr static int dimensions = Dimensions; ///< Number of dimensions
                                                ///< of the vector
  ///@}

  /// Inherit all base class constructors
  using base_type::base_type;

  /**
   * @brief Matrix multiplication: (M×N) * (N×K) → M×K
   *
   * Computes result(i, k) = Σ_j (*this)(i, j) * other(j, k),
   * where *this is M×N (Components×Dimensions) and other is N×K.
   *
   * @tparam K Number of columns in the right-hand matrix
   * @param other Right-hand side N×K tensor
   * @return M×K matrix product
   */
  template <int K>
  KOKKOS_INLINE_FUNCTION TensorPointViewType<T, Components, K, UseSIMD>
  operator*(const TensorPointViewType<T, Dimensions, K, UseSIMD> &other) const {
    TensorPointViewType<T, Components, K, UseSIMD> result;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int i = 0; i < Components; ++i) {
      for (int k = 0; k < K; ++k) {
        value_type sum{ 0.0 };
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
        for (int j = 0; j < Dimensions; ++j) {
          sum += (*this)(i, j) * other(j, k);
        }
        result(i, k) = sum;
      }
    }
    return result;
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr auto &
  operator*=(const value_type &scalar) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int i = 0; i < Components; ++i)
      for (int j = 0; j < Dimensions; ++j)
        (*this)(i, j) *= scalar;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION TensorPointViewType
  operator*(const value_type &scalar) const {
    TensorPointViewType result(*this);
    result *= scalar;
    return result;
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr auto &
  operator/=(const value_type &scalar) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
    for (int i = 0; i < Components; ++i)
      for (int j = 0; j < Dimensions; ++j)
        (*this)(i, j) /= scalar;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION TensorPointViewType
  operator/(const value_type &scalar) const {
    TensorPointViewType result(*this);
    result /= scalar;
    return result;
  }
};

} // namespace datatype
} // namespace specfem
