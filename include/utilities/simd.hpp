#pragma once

#include "datatypes/simd.hpp"

namespace specfem {
namespace utilities {

template <typename T>
KOKKOS_INLINE_FUNCTION bool is_close(const T &a, const T &b,
                                     const T &tol = static_cast<T>(1e-6)) {
  return specfem::datatype::all_of(
      Kokkos::abs(a - b) <= tol * Kokkos::max(Kokkos::abs(a), Kokkos::abs(b)));
}

} // namespace utilities
} // namespace specfem
