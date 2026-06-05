#pragma once

#include "dim2/impl_prefetch.hpp"
#include "dim3/impl_prefetch.hpp"
#include "specfem/cache_line.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @defgroup JacobianMatrixDataAccess
 *
 */

/**
 * @brief Prefetch Jacobian matrix data into device cache.
 *
 * Issues prefetch hints for all Jacobian matrix components at the given index.
 * Call before load_on_device() to ensure data is cache-resident. No-op on CUDA.
 *
 * @tparam LocalityHint Cache locality hint (default: High)
 * @tparam IndexType Point index (@ref specfem::point::index)
 * @tparam ContainerType Jacobian matrix container
 *
 * @param index Quadrature point indices (ispec, iz, [iy,] ix)
 * @param container Global Jacobian matrix
 *
 * @code
 * specfem::assembly::prefetch_on_device(next_idx, jacobian);
 * specfem::assembly::load_on_device(idx, jacobian, point_jac);
 * @endcode
 *
 * @ingroup JacobianMatrixDataAccess
 */
template <
    typename LocalityHint = specfem::cache_line::High, typename IndexType,
    typename ContainerType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
prefetch_on_device(const IndexType &index, const ContainerType &container) {
  impl_prefetch<true, LocalityHint>(index, container);
}

} // namespace specfem::assembly
