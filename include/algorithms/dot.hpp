#ifndef _ALGORITHMS_DOT_HPP
#define _ALGORITHMS_DOT_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
template <int N>
KOKKOS_INLINE_FUNCTION type_real
dot(const specfem::datatype::ScalarPointViewType<type_real, N> &a,
    const specfem::datatype::ScalarPointViewType<type_real, N> &b) {
  type_real result = 0.0;
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
