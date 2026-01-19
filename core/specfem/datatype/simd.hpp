/**
 * @file simd.hpp
 * @brief SIMD wrapper types for vectorized operations in spectral element
 * computations
 */
#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace datatype {
/**
 * @brief Wrapper around Kokkos::Experimental::native_simd to provide a
 * consistent interface for SIMD operations.
 *
 * @tparam T The base type of the SIMD vector.
 * @tparam UseSIMD Whether to use SIMD operations or not. If false, base type is
 * used directly.
 */
template <typename T, bool UseSIMD> struct simd;

/**
 * @brief Specialization of simd for when UseSIMD is false.
 *
 * @tparam T The base type of the SIMD vector.
 */
template <typename T> struct simd<T, false> {
  using base_type = T; ///< The base type of the SIMD vector.
  using datatype = T;  ///< The type of the SIMD vector. In this case, it is the
                       ///< same as the base type.
  constexpr static bool using_simd =
      false;              ///< Whether SIMD operations are used or not.
  using mask_type = bool; ///< The type of the mask used for SIMD operations.
  /**
   * @brief Returns the size of the SIMD vector.
   *
   * @return constexpr static int The size of the SIMD vector.
   */
  KOKKOS_FUNCTION constexpr static int size() { return 1; }
};

/**
 * @brief Specialization enabling SIMD operations
 *
 * Uses Kokkos SIMD types when available, falls back to scalar operations.
 *
 * @tparam T Base scalar type for SIMD vector
 */
template <typename T> struct simd<T, true> {
  /// Base scalar type
  using base_type = T;
#ifdef SPECFEM_ENABLE_SIMD
  /// SIMD vector type (native when enabled)
  using datatype = Kokkos::Experimental::simd<T>;
#else
  /// SIMD vector type (scalar fallback)
  using datatype =
      Kokkos::Experimental::basic_simd<T,
                                       Kokkos::Experimental::simd_abi::scalar>;
#endif
  /// SIMD operations enabled
  constexpr static bool using_simd = true;
  /**
   * @brief Returns the size of the SIMD vector.
   *
   * @return constexpr static int The size of the SIMD vector.
   */
  KOKKOS_FUNCTION constexpr static int size() {
#ifdef SPECFEM_ENABLE_SIMD
    return Kokkos::Experimental::simd<T>::size();
#else
    return Kokkos::Experimental::basic_simd<
        T, Kokkos::Experimental::simd_abi::scalar>::size();
#endif
  }
  /// SIMD mask type for conditional operations
  using mask_type = typename datatype::mask_type;
  /// Memory alignment tag for SIMD operations
  using tag_type = Kokkos::Experimental::element_aligned_tag;
};

/**
 * @namespace specfem::datatype::impl
 * @brief Implementation details for SIMD-like data types
 */
namespace impl {
/**
 * @brief Template for SIMD-like value storage
 *
 * Provides consistent interface for both SIMD and scalar storage modes.
 *
 * @tparam T Base data type
 * @tparam simd_type SIMD reference type
 * @tparam UseSIMD Enable SIMD operations
 */
template <typename T, typename simd_type, bool UseSIMD>
struct simd_like_value_type;

/**
 * @brief Scalar specialization (no SIMD)
 *
 * Stores single value with scalar operations.
 */
template <typename T, typename simd_type>
struct simd_like_value_type<T, simd_type, false> {
private:
  /// Single scalar value
  T m_value;
  using value_type = simd_like_value_type<T, simd_type, false>;

public:
  /**
   * @brief Addition assignment operator
   * @tparam U Compatible type for addition
   * @param other Value to add
   * @return Reference to this object
   */
  template <typename U> KOKKOS_FUNCTION value_type &operator+=(const U &other) {
    this->m_value += other;
    return *this;
  }

  /**
   * @brief Equality comparison
   * @tparam U Compatible type for comparison
   * @param other Value to compare with
   * @return True if equal
   */
  template <typename U> KOKKOS_FUNCTION bool operator==(const U &other) const {
    return this->m_value == other;
  }

  /**
   * @brief Inequality comparison
   * @tparam U Compatible type for comparison
   * @param other Value to compare with
   * @return True if not equal
   */
  template <typename U> KOKKOS_FUNCTION bool operator!=(const U &other) const {
    return this->m_value != other;
  }

  /**
   * @brief Get vector size (always 1 for scalar)
   * @return Size of vector (1)
   */
  KOKKOS_FUNCTION
  constexpr static int size() { return 1; }

  /// Scalar mask type
  using mask_type = bool;
};

/**
 * @brief SIMD specialization with vector storage
 *
 * Stores array of values for SIMD lane operations.
 */
template <typename T, typename simd_type>
struct simd_like_value_type<T, simd_type, true> {
private:
  /// SIMD vector size
  constexpr static int simd_size =
      specfem::datatype::simd<simd_type, true>::size();

public:
  /// Array storage for SIMD lanes
  T m_value[simd_size];

  /**
   * @brief Access SIMD lane value (const)
   * @param lane Lane index
   * @return Value at lane
   */
  KOKKOS_FUNCTION
  T operator[](const std::size_t lane) const { return this->m_value[lane]; }

  /**
   * @brief Access SIMD lane value (non-const)
   * @param lane Lane index
   * @return Reference to value at lane
   */
  KOKKOS_FUNCTION
  T &operator[](const std::size_t lane) { return this->m_value[lane]; }

  /// SIMD mask type for conditional operations
  using mask_type =
      typename specfem::datatype::simd<simd_type, true>::mask_type;
};

} // namespace impl

/**
 * @brief SIMD like data type that can be used when SIMD operations are not
 * defined for type T.
 *
 * @tparam T Type of the data.
 * @tparam simd_type Type of mask used for SIMD operations.
 * @tparam UseSIMD Whether to use SIMD operations or not.
 */
template <typename T, typename simd_type, bool UseSIMD> struct simd_like {
  using datatype =
      impl::simd_like_value_type<T, simd_type, UseSIMD>; ///< The data type.
  using base_type = T; ///< The base type of the data type.
  constexpr static bool using_simd =
      UseSIMD; ///< Whether SIMD operations are used or not.
  using mask_type = typename datatype::mask_type; ///< The type of the mask used
                                                  ///< for SIMD operations.
  /**
   * @brief Returns the size of the SIMD vector.
   *
   * @return constexpr static int The size of the SIMD vector.
   */
  constexpr static int size() { return datatype::size(); }
};

/**
 * @brief Type trait to detect SIMD mask types
 * @tparam T Type to check
 */
template <typename T> struct is_simd_mask : std::false_type {};

/**
 * @brief Specialization for Kokkos SIMD mask types
 * @tparam T Base type of SIMD mask
 * @tparam ABI SIMD ABI specification
 */
template <typename T, typename ABI>
struct is_simd_mask<Kokkos::Experimental::basic_simd_mask<T, ABI> >
    : std::true_type {};

/**
 * @brief Check if all mask elements are true
 *
 * Efficiently evaluates mask conditions using SIMD operations when available.
 * Falls back to scalar boolean check for non-SIMD masks.
 *
 * @tparam mask_type SIMD mask or boolean type
 * @param mask Mask to evaluate
 * @return True if all elements are true
 */
template <typename mask_type>
KOKKOS_INLINE_FUNCTION bool all_of(const mask_type &mask) {
  if constexpr (is_simd_mask<mask_type>::value) {
    return Kokkos::Experimental::all_of(mask);
  } else {
    return mask;
  }
};

} // namespace datatype
} // namespace specfem

/**
 * @brief Stream output operator for Kokkos SIMD types
 * @tparam T Base type of SIMD vector
 * @tparam Abi SIMD ABI specification
 * @param os Output stream
 * @param value SIMD value to print
 * @return Output stream reference
 */
template <typename T, typename Abi>
std::ostream &
operator<<(std::ostream &os,
           const Kokkos::Experimental::basic_simd<T, Abi> &value) {
  os << "[";
  for (int i = 0; i < value.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << value[i];
  }
  os << "]";
  return os;
}
