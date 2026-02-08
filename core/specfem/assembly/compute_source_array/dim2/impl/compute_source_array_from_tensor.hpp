#pragma once

#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

using PointJacobianMatrix2D =
    specfem::point::jacobian_matrix<specfem::element::dimension_tag::dim2,
                                    false, false>;
using JacobianViewType2D = Kokkos::View<PointJacobianMatrix2D **,
                                        Kokkos::LayoutRight, Kokkos::HostSpace>;

/**
 * @brief Compute source array for a 2D tensor source using spatial derivatives.
 *
 * For moment tensor sources, computes source contributions by transforming
 * tensor components via spatial derivatives of Lagrange basis functions:
 * @f$ S_{i,jz,jx} = M_{i,0} \frac{\partial L}{\partial x} + M_{i,1}
 * \frac{\partial L}{\partial z} @f$
 *
 * Derivatives are computed using the chain rule with Jacobian matrix elements.
 *
 * @param source Tensor source containing moment tensor components
 * @param mesh Mesh providing quadrature information
 * @param jacobian_matrix Coordinate transformation Jacobians at GLL points
 * @param source_array Output array of shape (ncomponents, ngllz, ngllx)
 */
void from_tensor(
    const specfem::sources::tensor_source<specfem::element::dimension_tag::dim2>
        &source,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<
        specfem::element::dimension_tag::dim2> &jacobian_matrix,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

/**
 * @brief Helper function computing tensor source array with precomputed
 * Jacobians.
 *
 * Algorithm:
 * 1. Compute Lagrange polynomials and derivatives at source location
 * 2. Build source polynomial field from interpolants
 * 3. Transform to physical space using Jacobian chain rule:
 *    @f$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial \xi}
 * \frac{\partial \xi}{\partial x} + \frac{\partial L}{\partial \gamma}
 * \frac{\partial \gamma}{\partial x} @f$
 * 4. Apply moment tensor: @f$ S_i = M_{i,0} \frac{\partial L}{\partial x} +
 * M_{i,1} \frac{\partial L}{\partial z} @f$
 *
 * Separates Jacobian extraction from computation for testability.
 *
 * @param tensor_source Tensor source object
 * @param element_jacobian_matrix Precomputed Jacobian matrices for the element
 * @param quadrature Quadrature containing GLL points
 * @param source_array Output array
 *
 * @note The derivatives are computed using element Jacobian matrices to map
 * from reference coordinates (xi, gamma) to physical coordinates (x, z).
 */
void compute_source_array_from_tensor_and_element_jacobian(
    const specfem::sources::tensor_source<specfem::element::dimension_tag::dim2>
        &tensor_source,
    const JacobianViewType2D &element_jacobian_matrix,
    const specfem::assembly::mesh_impl::quadrature<
        specfem::element::dimension_tag::dim2> &quadrature,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
