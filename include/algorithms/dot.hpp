#pragma once

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {
template <typename VectorPointViewType>
KOKKOS_INLINE_FUNCTION typename VectorPointViewType::value_type
dot(const VectorPointViewType &a, const VectorPointViewType &b) {

  static_assert(VectorPointViewType::isPointViewType,
                "Invalid ViewType: not a PointViewType");

  static_assert(VectorPointViewType::isScalarViewType,
                "Invalid ViewType: not a ScalarViewType");

  using value_type = typename VectorPointViewType::value_type;
  constexpr int N = VectorPointViewType::components;
  value_type result{ 0.0 };

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#pragma unroll
#endif
  for (int i = 0; i < N; ++i) {
    result += a(i) * b(i);
  }
  return result;
}

} // namespace algorithms
} // namespace specfem
