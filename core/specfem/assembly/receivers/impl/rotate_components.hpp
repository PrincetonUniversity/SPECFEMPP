#pragma once

#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::receivers_impl {

/**
 * @brief Apply receiver @p irec's stored rotation to one set of seismogram
 * components.
 *
 * Computes \f$ out_i = \sum_j R_{ij}(irec)\, in_j \f$ with \f$ R \f$ read from
 * @p rotation_matrices (layout `(nrec, ncomponents, ncomponents)`). Callable
 * from device kernels and from host code with a host view.
 *
 * @tparam NComponents Number of seismogram components (2 or 3)
 * @tparam RotationViewType Rank-3 Kokkos view of the rotation matrices
 * @param rotation_matrices Per-receiver rotation matrices
 * @param irec Receiver index into @p rotation_matrices
 * @param in Mesh-frame components
 * @param out Rotated components
 */
template <int NComponents, typename RotationViewType>
KOKKOS_INLINE_FUNCTION void
rotate_components(const RotationViewType &rotation_matrices, const int irec,
                  const type_real (&in)[NComponents],
                  type_real (&out)[NComponents]) {
  for (int i = 0; i < NComponents; ++i) {
    type_real sum = 0;
    for (int j = 0; j < NComponents; ++j) {
      sum += rotation_matrices(irec, i, j) * in[j];
    }
    out[i] = sum;
  }
}

} // namespace specfem::assembly::receivers_impl
