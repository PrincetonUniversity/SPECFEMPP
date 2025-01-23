#pragma once

#include "composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "dirichlet/dirichlet.hpp"
#include "enumerations/boundary.hpp"
#include "none/none.hpp"
#include "stacey/stacey.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {

namespace boundary_conditions {
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FORCEINLINE_FUNCTION void apply_boundary_conditions(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::isPointBoundaryType,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(PointFieldType::isPointFieldType,
                "PointFieldType must be a PointFieldType");

  static_assert(PointFieldType::store_velocity,
                "PointFieldType must store velocity");

  static_assert(PointAccelerationType::isPointFieldType,
                "PointAccelerationType must be a PointFieldType");

  static_assert(PointAccelerationType::store_acceleration,
                "PointAccelerationType must store acceleration");

  static_assert(
      std::is_same_v<typename PointFieldType::simd,
                     typename PointAccelerationType::simd>,
      "PointFieldType and PointAccelerationType must have the same SIMD type");

  static_assert(
      std::is_same_v<typename PointFieldType::simd,
                     typename PointAccelerationType::simd>,
      "PointFieldType and PointAccelerationType must have the same SIMD type");

  using boundary_tag_type =
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>;

  impl_apply_boundary_conditions(boundary_tag_type(), boundary, property, field,
                                 acceleration);
}

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FORCEINLINE_FUNCTION void
compute_mass_matrix_terms(const type_real dt, const PointBoundaryType &boundary,
                          const PointPropertyType &property,
                          PointMassMatrixType &mass_matrix) {

  static_assert(PointBoundaryType::isPointBoundaryType,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(PointMassMatrixType::isPointFieldType,
                "PointMassMatrixType must be a PointFieldType");

  static_assert(PointMassMatrixType::store_mass_matrix,
                "PointMassMatrixType must store mass matrix");

  using boundary_tag_type =
      std::integral_constant<specfem::element::boundary_tag,
                             PointBoundaryType::boundary_tag>;

  impl_compute_mass_matrix_terms(boundary_tag_type(), dt, boundary, property,
                                 mass_matrix);

  return;
}

} // namespace boundary_conditions
} // namespace specfem
