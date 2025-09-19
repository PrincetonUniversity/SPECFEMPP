#pragma once

#include "impl/register_array.hpp"
#include "simd.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace datatype {

/**
 * @brief Datatype used to scalar values at a GLL point. If N is small,
 * generates a datatype within a register.
 *
 * @tparam T Data type of the scalar values
 * @tparam Components Number of scalar values (components) at the GLL point
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, std::size_t Components, bool UseSIMD>
struct VectorPointViewType
    : public impl::RegisterArray<
          typename specfem::datatype::simd<T, UseSIMD>::datatype,
          Kokkos::extents<std::size_t, Components>, Kokkos::layout_left> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::RegisterArray<
      typename specfem::datatype::simd<T, UseSIMD>::datatype,
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
   *
   */
  ///@{
  using base_type::base_type;
  ///@}

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
 * @brief Datatype used to store vector values at a GLL point. If
 * NumberOfDimensions && Components is small, generates a datatype within a
 * register.
 *
 * @tparam T Data type of the vector values
 * @tparam Components Number of components of the vector
 * @tparam Dimensions Number of dimensions of the vector
 * @tparam UseSIMD Use SIMD datatypes for the array. If true, value_type is a
 * SIMD type
 */
template <typename T, int Components, int Dimensions, bool UseSIMD>
struct TensorPointViewType
    : public impl::RegisterArray<
          typename specfem::datatype::simd<T, UseSIMD>::datatype,
          Kokkos::extents<std::size_t, Components, Dimensions>,
          Kokkos::layout_left> {

  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = impl::RegisterArray<
      typename specfem::datatype::simd<T, UseSIMD>::datatype,
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

  using base_type::base_type; ///< Inherit constructors from base class
};

} // namespace datatype
} // namespace specfem
