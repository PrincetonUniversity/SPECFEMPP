#pragma once

#include "kokkos_abstractions.h"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Compute Lagrange interpolant-weighted source contributions at GLL
 * quadrature points for a 2D source.
 *
 * Computes the source array for a given source by evaluating Lagrange
 * interpolants at the source location and distributing source contributions to
 * GLL points. Supports both vector and tensor sources:
 * - Vector sources: Direct force components multiplied by Lagrange interpolants
 * - Tensor sources: Moment tensor components transformed via spatial
 * derivatives
 *
 * @tparam SourceArrayViewType Kokkos view type (must be rank-3)
 * @param source Source object (vector_source or tensor_source)
 * @param mesh Mesh containing quadrature information
 * @param jacobian_matrix Jacobian matrix for coordinate transformations
 * @param source_array Output array of shape (ncomponents, ngllz, ngllx)
 *
 * @code
 * // Vector source example
 * auto force = std::make_shared<specfem::sources::force<dim2>>(
 *     x, z, angle, stf, wavefield_type);
 * Kokkos::View<type_real***> source_array("src", 2, ngllz, ngllx);
 * compute_source_array(force, mesh, jacobian_matrix, source_array);
 *
 * // Tensor source example
 * auto moment_tensor = std::make_shared<specfem::sources::moment_tensor<dim2>>(
 *     x, z, Mxx, Mzz, Mxz, stf, wavefield_type);
 * compute_source_array(moment_tensor, mesh, jacobian_matrix, source_array);
 * @endcode
 */
template <typename SourceArrayViewType>
void compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::element::dimension_tag::dim2> >
        &source,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<
        specfem::element::dimension_tag::dim2> &jacobian_matrix,
    SourceArrayViewType &source_array);

/**
 * @brief Compute Lagrange interpolant-weighted source contributions at GLL
 * quadrature points for a 3D source.
 *
 * 3D version supporting vector and tensor sources with trilinear Lagrange
 * interpolation across xi, eta, and gamma coordinates. Supports both vector
 * and tensor sources:
 * - Vector sources: Direct force components multiplied by Lagrange interpolants
 * - Tensor sources: Moment tensor components transformed via spatial
 * derivatives
 *
 * @tparam SourceArrayViewType Kokkos view type (must be rank-4)
 * @param source Source object (vector_source or tensor_source)
 * @param mesh Mesh containing quadrature information
 * @param jacobian_matrix Jacobian matrix for coordinate transformations
 * @param source_array Output array of shape (ncomponents, ngllz, nglly, ngllx)
 *
 * @code
 * // Vector source example
 * auto force = std::make_shared<specfem::sources::force<dim3>>(
 *     x, y, z, angle, stf, wavefield_type);
 * Kokkos::View<type_real****> source_array("src", 3, ngllz, nglly, ngllx);
 * compute_source_array(force, mesh, jacobian_matrix, source_array);
 *
 * // Tensor source example
 * auto moment_tensor = std::make_shared<specfem::sources::moment_tensor<dim3>>(
 *     x, y, z, Mxx, Myy, Mzz, Mxy, Mxz, Myz, stf, wavefield_type);
 * compute_source_array(moment_tensor, mesh, jacobian_matrix, source_array);
 * @endcode
 */
template <typename SourceArrayViewType>
void compute_source_array(
    const std::shared_ptr<
        specfem::sources::source<specfem::element::dimension_tag::dim3> >
        &source,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::jacobian_matrix<
        specfem::element::dimension_tag::dim3> &jacobian_matrix,
    SourceArrayViewType &source_array);

} // namespace specfem::assembly

#include "specfem/assembly/compute_source_array/dim2/compute_source_array.tpp"
#include "specfem/assembly/compute_source_array/dim3/compute_source_array.tpp"
