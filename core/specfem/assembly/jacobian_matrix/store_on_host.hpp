#pragma once

#include "dim2/impl_store.hpp"
#include "dim3/impl_store.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @defgroup JacobianMatrixDataAccess
 *
 */

/**
 * @brief Store Jacobian matrix data from point structure to container on host
 * functions
 *
 * @ingroup JacobianMatrixDataAccess
 *
 * This function writes Jacobian transformation matrix components from a local
 * point structure back to the global container during host-based computations.
 *
 * **Storage Components:**
 * For 2D elements: xix, xiz, gammax, gammaz, jacobian
 * For 3D elements: xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz,
 * jacobian
 *
 * @tparam IndexType Point index type (specfem::point::index)
 * @tparam ContainerType Jacobian matrix container type for the mesh
 * @tparam PointType Local point Jacobian matrix type
 *
 * @param index Quadrature point indices (ispec, iz, iy, ix)
 * @param container Global Jacobian matrix container (modified)
 * @param point Local point Jacobian matrix structure containing data to store
 *
 * @pre IndexType must satisfy is_index_type constraint
 * @pre ContainerType and PointType must satisfy is_jacobian_matrix constraint
 * @pre Types must be compatible according to CheckCompatibility
 *
 * @code
 * // Example usage during mesh initialization
 * specfem::point::index<specfem::element::dimension_tag::dim3> idx(ispec, iz,
 * iy, ix);
 * specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim3>
 * computed_jac;
 *
 * // Compute Jacobian from mesh coordinates
 * compute_jacobian_from_coordinates(mesh_coords, computed_jac);
 *
 * // Store to host container for later device synchronization
 * specfem::assembly::store_on_host(idx, jacobian_container, computed_jac);
 *
 * // Later synchronize to device
 * jacobian_container.sync_to_device();
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value &&
            specfem::data_access::is_jacobian_matrix<PointType>::value,
        int> = 0>
void store_on_host(const IndexType &index, ContainerType &container,
                   const PointType &point) {

  using compatibility =
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>;
  static_assert(compatibility::value, "Incompatible types");

  impl_store<false>(index, container, point);
}

} // namespace specfem::assembly
