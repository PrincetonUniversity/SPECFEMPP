#pragma once

#include "boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "boundary_conditions/composite_stacey_dirichlet/composite_stacey_dirichlet.tpp"
#include "boundary_conditions/dirichlet/dirichlet.hpp"
#include "boundary_conditions/none/none.hpp"
#include "boundary_conditions/stacey/stacey.hpp"
#include "boundary_conditions/stacey/stacey.tpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {

/**
 * @brief Boundary conditions
 *
 * This namespace contains functions to apply boundary conditions.
 */
namespace boundary_conditions {

/**
 * @brief Apply boundary conditions for Dirichlet or None boundaries (time
 * independent)
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param acceleration Acceleration object
 */
template <typename PointBoundaryType,
          typename PointAccelerationType
          // \cond
          ,
          typename std::enable_if_t<
              ((PointBoundaryType::boundary_tag ==
                specfem::element::boundary_tag::none) ||
               (PointBoundaryType::boundary_tag ==
                specfem::element::boundary_tag::acoustic_free_surface)),
              int> = 0
          // \endcond
          >
KOKKOS_FORCEINLINE_FUNCTION void
apply_boundary_conditions(const PointBoundaryType &boundary,
                          PointAccelerationType &acceleration) {

  static_assert(specfem::data_access::is_point<PointBoundaryType>::value &&
                    specfem::data_access::is_boundary<PointBoundaryType>::value,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_acceleration<PointAccelerationType>::value,
      "PointAccelerationType must be a PointAccelerationType");

  using boundary_tag_type =
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>;

  impl_apply_boundary_conditions(
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>(),
      boundary, acceleration);

  return;
}

// \cond
// for some reason, the nonconforming kernel doesn't compile on cuda 11.8
// without an instance of a stacey and composite overload.
// this should not be called at runtime.
template <typename PointBoundaryType, typename PointAccelerationType,
          typename std::enable_if_t<
              ((PointBoundaryType::boundary_tag ==
                specfem::element::boundary_tag::stacey) ||
               (PointBoundaryType::boundary_tag ==
                specfem::element::boundary_tag::composite_stacey_dirichlet)),
              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
apply_boundary_conditions(const PointBoundaryType &boundary,
                          PointAccelerationType &acceleration) {
  return;
}
// \endcond

/**
 * @brief Apply boundary conditions (time dependent/absorbing)
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointVelocityType Point velocity type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param property Property object
 * @param field Field object
 * @param acceleration Acceleration object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointVelocityType, typename PointAccelerationType>
KOKKOS_FORCEINLINE_FUNCTION void apply_boundary_conditions(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &field, PointAccelerationType &acceleration) {

  static_assert(specfem::data_access::is_point<PointBoundaryType>::value &&
                    specfem::data_access::is_boundary<PointBoundaryType>::value,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(specfem::data_access::is_point<PointVelocityType>::value &&
                    specfem::data_access::is_field<PointVelocityType>::value,
                "PointFieldType must be a PointFieldType");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_field<PointAccelerationType>::value,
      "PointAccelerationType must be a PointFieldType");

  static_assert(
      std::is_same_v<typename PointVelocityType::simd,
                     typename PointAccelerationType::simd>,
      "PointFieldType and PointAccelerationType must have the same SIMD type");

  using boundary_tag_type =
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>;

  impl_apply_boundary_conditions(boundary_tag_type(), boundary, property, field,
                                 acceleration);
}

/**
 * @brief Compute mass matrix terms for boundary conditions
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
KOKKOS_FORCEINLINE_FUNCTION void
compute_mass_matrix_terms(const type_real dt, const PointBoundaryType &boundary,
                          const PointPropertyType &property,
                          PointMassMatrixType &mass_matrix) {

  static_assert(specfem::data_access::is_point<PointBoundaryType>::value &&
                    specfem::data_access::is_boundary<PointBoundaryType>::value,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(specfem::data_access::is_point<PointMassMatrixType>::value &&
                    specfem::data_access::is_field<PointMassMatrixType>::value,
                "PointMassMatrixType must be a PointFieldType");

  using boundary_tag_type =
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>;

  impl_compute_mass_matrix_terms(boundary_tag_type(), dt, boundary, property,
                                 mass_matrix);

  return;
}

} // namespace boundary_conditions
} // namespace specfem
