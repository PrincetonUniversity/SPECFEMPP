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
  using datatype = Kokkos::Experimental::native_simd<T>; ///< The type of the
                                                         ///< SIMD vector.
#else
  using datatype = Kokkos::Experimental::simd<
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
    return Kokkos::Experimental::native_simd<T>::size();
#else
    return Kokkos::Experimental::simd<
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
} // namespace datatype
} // namespace specfem
