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

template <typename T> struct simd<T, true> {
  using base_type = T; ///< The base type of the SIMD vector.
#ifdef ENABLE_SIMD
  using datatype = Kokkos::Experimental::simd<T>; ///< The type of the
                                                  ///< SIMD vector.
#else
  using datatype = Kokkos::Experimental::basic_simd<
      T, Kokkos::Experimental::simd_abi::scalar>; ///< The type of the SIMD
                                                  ///< vector.
#endif
  constexpr static bool using_simd =
      true; ///< Whether SIMD operations are used or not.
  /**
   * @brief Returns the size of the SIMD vector.
   *
   * @return constexpr static int The size of the SIMD vector.
   */
  KOKKOS_FUNCTION constexpr static int size() {
#ifdef ENABLE_SIMD
    return Kokkos::Experimental::simd<T>::size();
#else
    return Kokkos::Experimental::basic_simd<
        T, Kokkos::Experimental::simd_abi::scalar>::size();
#endif
  }
  using mask_type = typename datatype::mask_type; ///< The type of the mask used
                                                  ///< for SIMD operations.
  using tag_type =
      Kokkos::Experimental::element_aligned_tag; ///< The tag type used for SIMD
                                                 ///< operations.
};

namespace impl {
template <typename T, typename simd_type, bool UseSIMD>
struct simd_like_value_type;

template <typename T, typename simd_type>
struct simd_like_value_type<T, simd_type, false> {
private:
  T m_value;
  using value_type = simd_like_value_type<T, simd_type, false>;

public:
  template <typename U> KOKKOS_FUNCTION value_type &operator+=(const U &other) {
    this->m_value += other;
    return *this;
  }

  template <typename U> KOKKOS_FUNCTION bool operator==(const U &other) const {
    return this->m_value == other;
  }

  template <typename U> KOKKOS_FUNCTION bool operator!=(const U &other) const {
    return this->m_value != other;
  }

  KOKKOS_FUNCTION
  constexpr static int size() { return 1; }

  using mask_type = bool;
};

template <typename T, typename simd_type>
struct simd_like_value_type<T, simd_type, true> {
private:
  constexpr static int simd_size =
      specfem::datatype::simd<simd_type, true>::size();

public:
  T m_value[simd_size];

  KOKKOS_FUNCTION
  T operator[](const std::size_t lane) const { return this->m_value[lane]; }

  KOKKOS_FUNCTION
  T &operator[](const std::size_t lane) { return this->m_value[lane]; }

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

template <typename T> struct is_simd_mask : std::false_type {};

template <typename T, typename ABI>
struct is_simd_mask<Kokkos::Experimental::basic_simd_mask<T, ABI> >
    : std::true_type {};

/**
 * @brief Checks if all elements in the mask are true.
 *
 * This function is specialized for SIMD masks and uses
 * Kokkos::Experimental::all_of for efficient evaluation. For non-SIMD masks, it
 * simply checks the boolean value.
 *
 * @tparam mask_type The type of the mask.
 * @param mask The mask to check.
 * @return true if all elements in the mask are true, false otherwise.
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
 * @brief Overloaded operator<< for printing simd values.
 * @param os The output stream.
 * @param value The simd value to print.
 * @return The output stream after printing the simd value.
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
