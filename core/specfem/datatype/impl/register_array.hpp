/**
 * @file register_array.hpp
 * @brief Stack-allocated arrays with specified layouts for high-performance
 * computing
 */
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <mdspan/mdspan.hpp>
#include <sstream>
#include <type_traits>

namespace specfem {
namespace datatype {
namespace impl {

/**
 * @brief Compute total size from static extents
 * @tparam Extents Extent specification type
 * @return Total number of elements
 */
template <typename Extents> constexpr size_t compute_size() {

  size_t size = 1;
  for (size_t i = 0; i < Extents::rank(); ++i) {
    size *= Extents::static_extent(i);
  }
  return size;
}

/**
 * @brief Check if indices are within bounds at compile time
 * @tparam Extents Extent specification type
 * @tparam IndexType Variadic index types
 * @param i Index values to check
 * @return True if all indices are within bounds
 */
template <typename Extents, typename... IndexType>
KOKKOS_INLINE_FUNCTION constexpr bool check_bounds(const IndexType &...i) {
  std::size_t index = 0;
  return ((i >= 0 && i < Extents::static_extent(index++)) && ...);
}

/**
 * @brief Stack-allocated multi-dimensional array with compile-time layout
 *
 * High-performance register array for small, fixed-size data with
 * specified memory layout.
 *
 * @tparam T Base data type of the array elements
 * @tparam Extents Array dimensions and sizes
 * @tparam Layout Memory layout specification
 */
template <typename T, typename Extents, typename Layout> class RegisterArray {

private:
  /// Array rank (number of dimensions)
  constexpr static std::size_t rank = Extents::rank();
  /// Total number of elements
  constexpr static std::size_t size = impl::compute_size<Extents>();
  /// Memory layout mapping type
  using mapping = typename Layout::template mapping<Extents>;

public:
  /// Element value type
  using value_type = T;

  /**
   * @brief Construct array filled with single value
   * @param value Fill value for all elements
   */
  KOKKOS_INLINE_FUNCTION
  RegisterArray(const value_type value) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = value;
    }
  }

  /**
   * @brief Construct from individual element values
   * @tparam Args Argument types (must match array size)
   * @param args Element values in linear order
   */
  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == size, bool>::type = true>
  KOKKOS_INLINE_FUNCTION RegisterArray(const Args &...args)
      : m_value{ static_cast<value_type>(args)... } {}

  /**
   * @brief Copy constructor
   * @param other Array to copy from
   */
  KOKKOS_INLINE_FUNCTION
  RegisterArray(const RegisterArray &other) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = other.m_value[i];
    }
  }

  /**
   * @brief Default constructor (zero-initialized)
   */
  KOKKOS_INLINE_FUNCTION
  RegisterArray() {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = 0.0;
    }
  }

  /**
   * @brief Construct from C-style array
   * @param values Pointer to array data (must have at least 'size' elements)
   */
  KOKKOS_INLINE_FUNCTION
  RegisterArray(const T *values) {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = values[i];
    }
  }

  /**
   * @brief Multi-dimensional element access (non-const)
   * @tparam IndexType Variadic index types
   * @param i Multi-dimensional indices
   * @return Reference to element
   */
  template <typename... IndexType>
  KOKKOS_INLINE_FUNCTION constexpr value_type &
  operator()(const IndexType &...i) {
#ifndef NDEBUG
    // check if the indices are within bounds
    if (!check_bounds<Extents>(i...)) {
      // Abort the program with an error message
      Kokkos::abort("Index out of bounds");
    }
#endif
    return m_value[mapping()(i...)];
  }

  /**
   * @brief Multi-dimensional element access (const)
   * @tparam IndexType Variadic index types
   * @param i Multi-dimensional indices
   * @return Const reference to element
   */
  template <typename... IndexType>
  KOKKOS_INLINE_FUNCTION constexpr const value_type &
  operator()(const IndexType &...i) const {
#ifndef NDEBUG
    // check if the indices are within bounds
    if (!check_bounds<Extents>(i...)) {
      // Abort the program with an error message
      Kokkos::abort("Index out of bounds");
    }
#endif
    return m_value[mapping()(i...)];
  }

  /**
   * @brief Compute L2 norm (for 1D arrays)
   * @return L2 norm value
   */
  KOKKOS_INLINE_FUNCTION
  T l2_norm() const {
    return l2_norm(std::integral_constant<bool, rank == 1>());
  }

  /**
   * @brief Equality comparison for integral types
   * @param other Array to compare with
   * @return True if arrays are equal
   */
  template <typename U = T,
            std::enable_if_t<std::is_integral<U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (m_value[i] != other.m_value[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::abs(m_value[i] - other.m_value[i]) >
          1e-6 * Kokkos::abs(m_value[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<std::is_integral<typename U::value_type>::value,
                             int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::Experimental::any_of(m_value[i] != other.m_value[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<
                std::is_floating_point<typename U::value_type>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::Experimental::any_of(
              Kokkos::abs(m_value[i] - other.m_value[i]) >
              1e-6 * Kokkos::abs(m_value[i]))) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Inequality comparison
   * @param other Array to compare with
   * @return True if arrays are not equal
   */
  KOKKOS_INLINE_FUNCTION bool operator!=(const RegisterArray &other) const {
    return !(*this == other);
  }

private:
  /// Stack-allocated data storage
  T m_value[size];

  KOKKOS_INLINE_FUNCTION
  T l2_norm(const std::true_type &) const {
    T norm = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
      norm += m_value[i] * m_value[i];
    }
    return Kokkos::sqrt(norm);
  }

  KOKKOS_INLINE_FUNCTION
  T l2_norm(const std::false_type &) const {
    static_assert(rank == 1, "l2_norm is only implemented for 1-D arrays");
    return 0.0;
  }
};

} // namespace impl
} // namespace datatype
} // namespace specfem
