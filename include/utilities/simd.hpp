#pragma once

#include "datatypes/simd.hpp"

namespace specfem {
namespace utilities {

/**
 * @brief Check if two values are close within a tolerance.
 *
 * This function checks whether two values are considered "close" to each other
 * within a specified tolerance. The comparison uses a relative error
 * calculation that accounts for the magnitude of the values.
 *
 * @tparam T Type of values to compare. Must support arithmetic operations.
 * @param a First value to compare.
 * @param b Second value to compare.
 * @param tol Relative tolerance for comparison (default: 1e-6).
 * @return bool True if values are considered close according to the formula:
 *              |a - b| <= tol * max(|a|, |b|)
 *              For simd types, returns true only if all lanes meet the
 * condition.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION bool is_close(const T &a, const T &b,
                                     const T &tol = static_cast<T>(1e-6)) {
  return specfem::datatype::all_of(
      Kokkos::abs(a - b) <= tol * Kokkos::max(Kokkos::abs(a), Kokkos::abs(b)));
}

} // namespace utilities
} // namespace specfem
