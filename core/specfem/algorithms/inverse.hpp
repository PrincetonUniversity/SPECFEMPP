#pragma once

#include "specfem/datatype.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace algorithms {

/**
 * @brief Compute the inverse of a 2x2 tensor
 *
 * @tparam T Scalar data type
 * @tparam UseSIMD Enable SIMD vectorization
 * @param M Input 2x2 tensor (row-major: M(row, col))
 * @return Inverse of M
 */
template <typename T, bool UseSIMD>
KOKKOS_FUNCTION specfem::datatype::TensorPointViewType<T, 2, 2, UseSIMD>
inverse(const specfem::datatype::TensorPointViewType<T, 2, 2, UseSIMD> &M) {
  using value_type =
      typename specfem::datatype::TensorPointViewType<T, 2, 2,
                                                      UseSIMD>::value_type;
  const auto det = M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1);
  const auto inv_det = static_cast<value_type>(1.0) / det;
  specfem::datatype::TensorPointViewType<T, 2, 2, UseSIMD> result;
  result(0, 0) = M(1, 1) * inv_det;
  result(0, 1) = -M(0, 1) * inv_det;
  result(1, 0) = -M(1, 0) * inv_det;
  result(1, 1) = M(0, 0) * inv_det;
  return result;
}

/**
 * @brief Compute the inverse of a 3x3 tensor
 *
 * @tparam T Scalar data type
 * @tparam UseSIMD Enable SIMD vectorization
 * @param M Input 3x3 tensor (row-major: M(row, col))
 * @return Inverse of M
 */
template <typename T, bool UseSIMD>
KOKKOS_FUNCTION specfem::datatype::TensorPointViewType<T, 3, 3, UseSIMD>
inverse(const specfem::datatype::TensorPointViewType<T, 3, 3, UseSIMD> &M) {
  using value_type =
      typename specfem::datatype::TensorPointViewType<T, 3, 3,
                                                      UseSIMD>::value_type;
  const auto det = M(0, 0) * (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)) -
                   M(1, 0) * (M(0, 1) * M(2, 2) - M(2, 1) * M(0, 2)) +
                   M(2, 0) * (M(0, 1) * M(1, 2) - M(1, 1) * M(0, 2));
  const auto inv_det = static_cast<value_type>(1.0) / det;
  specfem::datatype::TensorPointViewType<T, 3, 3, UseSIMD> result;
  result(0, 0) = (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)) * inv_det;
  result(0, 1) = -(M(0, 1) * M(2, 2) - M(2, 1) * M(0, 2)) * inv_det;
  result(0, 2) = (M(0, 1) * M(1, 2) - M(1, 1) * M(0, 2)) * inv_det;
  result(1, 0) = -(M(1, 0) * M(2, 2) - M(2, 0) * M(1, 2)) * inv_det;
  result(1, 1) = (M(0, 0) * M(2, 2) - M(2, 0) * M(0, 2)) * inv_det;
  result(1, 2) = -(M(0, 0) * M(1, 2) - M(1, 0) * M(0, 2)) * inv_det;
  result(2, 0) = (M(1, 0) * M(2, 1) - M(2, 0) * M(1, 1)) * inv_det;
  result(2, 1) = -(M(0, 0) * M(2, 1) - M(2, 0) * M(0, 1)) * inv_det;
  result(2, 2) = (M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1)) * inv_det;
  return result;
}

} // namespace algorithms
} // namespace specfem
