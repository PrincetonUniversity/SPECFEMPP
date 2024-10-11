#ifndef _ALGORITHMS_DOT_HPP
#define _ALGORITHMS_DOT_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
template <typename ScalarPointViewType>
KOKKOS_INLINE_FUNCTION typename ScalarPointViewType::value_type
dot(const ScalarPointViewType &a, const ScalarPointViewType &b) {

  static_assert(ScalarPointViewType::isPointViewType,
                "Invalid ViewType: not a PointViewType");

  static_assert(ScalarPointViewType::isScalarViewType,
                "Invalid ViewType: not a ScalarViewType");

  using value_type = typename ScalarPointViewType::value_type;
  constexpr int N = ScalarPointViewType::components;
  value_type result{ 0.0 };

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
