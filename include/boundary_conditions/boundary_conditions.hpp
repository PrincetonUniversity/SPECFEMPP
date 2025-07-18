#pragma once

#include "composite_stacey_dirichlet/composite_stacey_dirichlet.hpp"
#include "dirichlet/dirichlet.hpp"
#include "enumerations/interface.hpp"
#include "none/none.hpp"
#include "specfem/data_access.hpp"
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

  static_assert(specfem::data_access::is_point<PointBoundaryType>::value &&
                    specfem::data_access::is_boundary<PointBoundaryType>::value,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(specfem::data_access::is_point<PointFieldType>::value &&
                    specfem::data_access::is_field<PointFieldType>::value,
                "PointFieldType must be a PointFieldType");

  static_assert(PointFieldType::store_velocity,
                "PointFieldType must store velocity");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_field<PointAccelerationType>::value,
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

  static_assert(specfem::data_access::is_point<PointBoundaryType>::value &&
                    specfem::data_access::is_boundary<PointBoundaryType>::value,
                "PointBoundaryType must be a PointBoundaryType");

  static_assert(specfem::data_access::is_point<PointMassMatrixType>::value &&
                    specfem::data_access::is_field<PointMassMatrixType>::value,
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
