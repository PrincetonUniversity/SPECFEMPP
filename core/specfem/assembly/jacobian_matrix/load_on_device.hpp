#pragma once

#include "dim2/impl_load.hpp"
#include "dim3/impl_load.hpp"
#include "specfem/data_access.hpp"

namespace specfem::assembly {

/**
 * @defgroup JacobianMatrixDataAccess
 *
 */

/**
 * @brief Load Jacobian matrix data from container to point structure on GPU
 * device
 *
 * @ingroup JacobianMatrixDataAccess
 *
 * This function transfers Jacobian transformation matrix components from a
 * global container to a local point structure for GPU-based computations.
 *
 * **Data Components Loaded:**
 * - **2D:** xix, xiz, gammax, gammaz, jacobian (determinant)
 * - **3D:** xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, jacobian
 *
 * @tparam IndexType Point index type (@ref specfem::point::index)
 * @tparam ContainerType Jacobian matrix container type for the mesh
 * @tparam PointType Local point Jacobian matrix type (@ref
 * specfem::point::jacobian_matrix)
 *
 * @param index Quadrature point indices (ispec, iz, iy, ix)
 * @param container Global Jacobian matrix container
 * @param point Local point Jacobian matrix structure (output)
 *
 * @pre IndexType must satisfy is_index_type constraint
 * @pre ContainerType and PointType must satisfy is_jacobian_matrix constraint
 * @pre Types must be compatible according to CheckCompatibility
 *
 *
 * @code
 * // Example usage in GPU kernel
 * specfem::point::index<specfem::element::dimension_tag::dim2> idx(ispec, iz,
 * ix); specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2>
 * point_jac; specfem::assembly::load_on_device(idx, jacobian_container,
 * point_jac);
 *
 * // Access loaded values
 * type_real xix_val = point_jac.xix;          // ∂ξ/∂x
 * type_real det_jac = point_jac.jacobian;     // |J|
 * @endcode
 */
template <
    typename IndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<IndexType>::value &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value &&
            specfem::data_access::is_jacobian_matrix<PointType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const IndexType &index,
                                                const ContainerType &container,
                                                PointType &point) {

  using compatibility =
      specfem::data_access::CheckCompatibility<IndexType, ContainerType,
                                               PointType>;
  static_assert(compatibility::value, "Incompatible types");

  impl_load<true>(index, container, point);
}

} // namespace specfem::assembly
