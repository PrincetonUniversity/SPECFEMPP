#pragma once

#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {
/**
 * @defgroup MeshDataAccess
 *
 */

/**
 * @brief Load weights for a GLL point on host or device
 *
 * @ingroup MeshDataAccess
 *
 * @tparam IndexType Index type. Needs to be a point index type
 * @tparam ViewType View type for mesh.weights.
 * @param index GLL point index
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <bool on_device, typename IndexType, typename ViewType>
KOKKOS_INLINE_FUNCTION void
impl_load(const IndexType &index, const ViewType &weights,
          specfem::point::weights<IndexType::dimension_tag> &point_weights) {
  point_weights.wz = weights(index.iz);
  if constexpr (IndexType::dimension_tag ==
                specfem::element::dimension_tag::dim3) {
    point_weights.wy = weights(index.iy);
  }
  point_weights.wx = weights(index.ix);
}

/**
 * @brief Load weights for a GLL point on device
 *
 * @ingroup MeshDataAccess
 *
 * @tparam IndexType Index type. Needs to be a point index type
 * @tparam ViewType View type for mesh.weights.
 * @param index GLL point index
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <typename IndexType, typename ViewType,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               specfem::data_access::is_point<IndexType>::value),
              int> = 0>
KOKKOS_INLINE_FUNCTION void load_on_device(
    const IndexType &index, const ViewType &weights,
    specfem::point::weights<IndexType::dimension_tag> &point_weights) {
  impl_load<true>(index, weights, point_weights);
}

/**
 * @brief Load weights for a GLL point on the host
 *
 * @ingroup MeshDataAccess
 *
 * @tparam IndexType Index type. Needs to be a point index type
 * @tparam ViewType View type for mesh.weights.
 * @param index GLL point index
 * @param mesh Mesh data
 * @param lagrange_derivative Quadrature data for the element (output)
 */
template <typename IndexType, typename ViewType,
          typename std::enable_if_t<
              (specfem::data_access::is_index_type<IndexType>::value &&
               specfem::data_access::is_point<IndexType>::value),
              int> = 0>
void load_on_host(
    const IndexType &index, const ViewType &weights,
    specfem::point::weights<IndexType::dimension_tag> &point_weights) {
  impl_load<false>(index, weights, point_weights);
}
} // namespace specfem::assembly
