#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "specfem_setup.hpp"
#include <string>

namespace specfem {
namespace utilities {

/**
 * @brief Check if two values are close within a tolerance.
 *
 * This function checks whether two values are considered "close" to each other
 * within specified tolerances. The comparison uses both relative and absolute
 * tolerances to handle both large and small numbers appropriately.
 *
 * @tparam T Type of values to compare. Must support arithmetic operations.
 * @param a First value to compare.
 * @param b Second value to compare.
 * @param rel_tol Relative tolerance for comparison (default: 1e-6).
 * @param abs_tol Absolute tolerance for comparison (default: 1e-7).
 * @return bool True if values are considered close according to the formula:
 *              |a - b| <= abs_tol + rel_tol * max(|a|, |b|)
 *              For simd types, returns true only if all lanes meet the
 * condition.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION bool is_close(const T &a, const T &b,
                                     const T &rel_tol = static_cast<T>(1e-6),
                                     const T &abs_tol = static_cast<T>(1e-7)) {
  // Follow NumPy's symmetric approach: |a - b| <= abs_tol + rel_tol * max(|a|,
  // |b|)
  return specfem::datatype::all_of(
      Kokkos::abs(a - b) <=
      (abs_tol + rel_tol * Kokkos::max(Kokkos::abs(a), Kokkos::abs(b))));
}

} // namespace utilities
} // namespace specfem
