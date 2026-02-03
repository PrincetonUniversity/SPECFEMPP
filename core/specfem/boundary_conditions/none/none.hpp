#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file none.hpp
 * @brief No boundary conditions
 *
 */

namespace specfem {
namespace boundary_conditions {

using none_type = std::integral_constant<specfem::element::boundary_tag,
                                         specfem::element::boundary_tag::none>;

/**
 * @brief Apply no boundary conditions.
 *
 * Calling this function results in a no-op.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void impl_apply_boundary_conditions(
    const none_type &, const PointBoundaryType &, const PointPropertyType &,
    const PointFieldType &, PointAccelerationType &) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::none,
                "Boundary tag must be none");

  // Do nothing
  return;
}

/**
 * @brief Compute mass matrix terms for no boundary conditions
 *
 * Calling this function results in a no-op.
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
KOKKOS_INLINE_FUNCTION void impl_compute_mass_matrix_terms(
    const none_type &, const type_real dt, const PointBoundaryType &boundary,
    const PointPropertyType &property, PointMassMatrixType &mass_matrix) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::none,
                "Boundary tag must be none");

  // Do nothing
  return;
}

/**
 * @brief Apply no boundary conditions
 *
 * Calling this function results in a no-op.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointAccelerationType Point acceleration type
 */
template <typename PointBoundaryType, typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
impl_apply_boundary_conditions(const none_type &, const PointBoundaryType &,
                               PointAccelerationType &) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::none,
                "Boundary tag must be none");

  // Do nothing
  return;
}

} // namespace boundary_conditions
} // namespace specfem
