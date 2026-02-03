#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file composite_stacey_dirichlet.hpp
 * @brief Composite Stacey absorbing and Dirichlet boundary conditions
 *
 */

namespace specfem {
namespace boundary_conditions {

using composite_stacey_dirichlet_type = std::integral_constant<
    specfem::element::boundary_tag,
    specfem::element::boundary_tag::composite_stacey_dirichlet>;

/**
 * @brief Apply composite Stacey and Dirichlet boundary conditions
 *
 * Handles boundary conditions for points situated at the intersection
 * of an Absorbing (Stacey) boundary and a Fixed (Dirichlet) boundary,
 * such as the corner of a domain.
 *
 * This function applies the Stacey absorbing condition (using `t_traction`)
 * to the components normal to the absorbing boundary, while enforcing
 * the Dirichlet condition (zeroing out acceleration/velocity) on the
 * components constrained by the fixed boundary.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param property Property object
 * @param field Field object
 * @param acceleration Acceleration object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(
    const composite_stacey_dirichlet_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration);

/**
 * @brief Compute mass matrix terms for composite Stacey and Dirichlet boundary
 * conditions
 *
 * Modifies the mass matrix for points at the intersection of Stacey and
 * Dirichlet boundaries (e.g. domain corners).
 *
 * Adds the contribution from the Stacey absorbing boundary (related to time
 * step `dt` and material properties) to the mass matrix diagonal for the
 * absorbing components, ensuring proper damping at the corner.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointMassMatrixType Point mass matrix type
 * @param dt Time step
 * @param boundary Boundary object
 * @param property Property object
 * @param mass_matrix Mass matrix object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FUNCTION void impl_compute_mass_matrix_terms(
    const composite_stacey_dirichlet_type &, const type_real dt,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    PointMassMatrixType &mass_matrix);

} // namespace boundary_conditions
} // namespace specfem
