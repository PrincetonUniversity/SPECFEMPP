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
 * @brief Store Jacobian matrix data from point structure to container on GPU
 * device
 *
 * @ingroup JacobianMatrixDataAccess
 *
 * This function writes Jacobian transformation matrix components from a local
 * point structure back to the global container during GPU-based computations.
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
 * // Example usage in GPU kernel after Jacobian computation
 * specfem::point::index<specfem::element::dimension_tag::dim2> idx(ispec, iz,
 * ix); specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2>
 * computed_jac;
 *
 * // Compute Jacobian values (simplified)
 * computed_jac.xix() = compute_xix(mesh_coords);
 * computed_jac.jacobian() = compute_jacobian_determinant(mesh_coords);
 *
 * // Store back to global container
 * specfem::assembly::store_on_device(idx, jacobian_container, computed_jac);
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value &&
            specfem::data_access::is_jacobian_matrix<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void store_on_device(const IndexType &index,
                                                 ContainerType &container,
                                                 const PointType &point) {

  using compatibility =
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>;
  static_assert(compatibility::value, "Incompatible types");

  impl_store<true>(index, container, point);
}

} // namespace specfem::assembly
