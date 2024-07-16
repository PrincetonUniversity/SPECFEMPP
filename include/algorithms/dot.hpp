#ifndef _ALGORITHMS_DOT_HPP
#define _ALGORITHMS_DOT_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
template <int N, bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    typename specfem::datatype::simd<type_real, UseSIMD>::datatype
    dot(const specfem::datatype::ScalarPointViewType<type_real, N, UseSIMD> &a,
        const specfem::datatype::ScalarPointViewType<type_real, N, UseSIMD>
            &b) {
  typename specfem::datatype::simd<type_real, UseSIMD>::datatype result{ 0.0 };
#ifdef KOKKOS_ENABLE_CUDA
#pragma unroll
#endif
  for (int i = 0; i < N; ++i) {
    result += a(i) * b(i);
  }
  return result;
}

} // namespace algorithms
} // namespace specfem

#endif /* _ALGORITHMS_DOT_HPP */
