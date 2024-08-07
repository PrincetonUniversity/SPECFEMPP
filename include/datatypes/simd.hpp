#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace datatype {

template <typename T, bool UseSIMD> struct simd;

template <typename T> struct simd<T, false> {
  using base_type = T;
  using datatype = T;
  constexpr static bool using_simd = false;
  using mask_type = bool;
  KOKKOS_FUNCTION constexpr static int size() { return 1; }
};

#ifdef ENABLE_SIMD
template <typename T> struct simd<T, true> {
  using base_type = T;
  using datatype = Kokkos::Experimental::native_simd<T>;
  constexpr static bool using_simd = true;
  KOKKOS_FUNCTION constexpr static int size() {
    return Kokkos::Experimental::native_simd<T>::size();
  }
  using mask_type = typename datatype::mask_type;
  using tag_type = Kokkos::Experimental::element_aligned_tag;
};
#else  // ENABLE_SIMD
template <typename T> struct simd<T, true> {
  using base_type = T;
  using datatype =
      Kokkos::Experimental::simd<T, Kokkos::Experimental::simd_abi::scalar>;
  constexpr static bool using_simd = true;
  KOKKOS_FUNCTION constexpr static int size() {
    return Kokkos::Experimental::simd<
        T, Kokkos::Experimental::simd_abi::scalar>::size();
  }
  using mask_type = typename datatype::mask_type;
  using tag_type = Kokkos::Experimental::element_aligned_tag;
};
#endif // ENABLE_SIMD

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

template <typename T, typename simd_type, bool UseSIMD> struct simd_like {
  using datatype = impl::simd_like_value_type<T, simd_type, UseSIMD>;
  constexpr static bool using_simd = UseSIMD;
  using mask_type = typename datatype::mask_type;
  constexpr static int size() { return datatype::size(); }
};
} // namespace datatype
} // namespace specfem
