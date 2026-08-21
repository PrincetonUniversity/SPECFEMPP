#pragma once

#include <array>

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Map a SPECFEM3D_GLOBE hex27 anchor index to SPECFEM++ hex27 order.
 *
 * Both conventions currently use corners 0--7, edges 8--19, faces 20--25,
 * and the center at 26. Keeping the identity permutation explicit makes an
 * upstream ordering change visible and testable instead of silently changing
 * element Jacobians.
 */
inline constexpr std::array<int, 27> globe_to_specfem_hex27 = {
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
  14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
};

/** @brief Inverse of @ref globe_to_specfem_hex27. */
constexpr std::array<int, 27> specfem_to_globe_hex27() {
  std::array<int, 27> inverse{};
  for (int globe_index = 0; globe_index < 27; ++globe_index) {
    inverse[globe_to_specfem_hex27[globe_index]] = globe_index;
  }
  return inverse;
}

} // namespace specfem::io::mesh::impl::fortran::dim3
