#pragma once

#include "specfem_setup.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace datatype {

template <typename T, bool UseSIMD> struct simd;

template <typename T> struct simd<T, false> {
  using datatype = T;
  constexpr static bool using_simd = false;
  constexpr static int size() { return 1; }
};

#ifdef ENABLE_SIMD
template <typename T> struct simd<T, true> {
  using datatype = Kokkos::Experimental::native_simd<T>;
  constexpr static bool using_simd = true;
  constexpr static int size() {
    return Kokkos::Experimental::native_simd<T>::size();
  }
  using mask_type = typename datatype::mask_type;
  using tag_type = Kokkos::Experimental::element_aligned_tag;
};
#else  // ENABLE_SIMD
template <typename T> struct simd<T, true> {
  using datatype =
      Kokkos::Experimental::simd<T, Kokkos::Experimental::simd_abi::scalar>;
  constexpr static bool using_simd = true;
  constexpr static int size() {
    return Kokkos::Experimental::simd<
        T, Kokkos::Experimental::simd_abi::scalar>::size();
  }
  using mask_type = typename datatype::mask_type;
  using tag_type = Kokkos::Experimental::element_aligned_tag;
};
#endif // ENABLE_SIMD

} // namespace datatype
} // namespace specfem
