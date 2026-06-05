#pragma once

#include "specfem/cache_line.hpp"
#include "specfem/data_access.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

/**
 * @brief Implementation: prefetch 2D Jacobian matrix components.
 *
 * Issues prefetch hints for the 5 Jacobian matrix terms (xix, xiz, gammax,
 * gammaz, jacobian) at a given 2D quadrature point, either on device or host.
 *
 * @tparam on_device If true, prefetch device data; otherwise host mirror
 * @tparam LocalityHint Cache locality hint (default: High)
 * @tparam IndexType 2D point index (ispec, iz, ix)
 * @tparam ContainerType Jacobian matrix container
 *
 * @param index Quadrature point location
 * @param container Global Jacobian matrix
 *
 * @ingroup JacobianMatrixDataAccess
 */
template <
    bool on_device, typename LocalityHint = specfem::cache_line::High,
    typename IndexType, typename ContainerType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            IndexType::dimension_tag == specfem::element::dimension_tag::dim2 &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_prefetch(const IndexType &index,
                                               const ContainerType &container) {

  const auto &mapping = container.xix.get_mapping();
  const std::size_t _index = mapping(index.ispec, index.iz, index.ix);

  if constexpr (on_device) {
    specfem::cache_line::prefetch(&container.xix[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.xiz[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.gammax[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.gammaz[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.jacobian[_index], LocalityHint{});
  } else {
    specfem::cache_line::prefetch(&container.h_xix[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.h_xiz[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.h_gammax[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.h_gammaz[_index], LocalityHint{});
    specfem::cache_line::prefetch(&container.h_jacobian[_index],
                                  LocalityHint{});
  }
}

} // namespace specfem::assembly
