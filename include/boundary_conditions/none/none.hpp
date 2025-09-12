#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace boundary_conditions {

using none_type = std::integral_constant<specfem::element::boundary_tag,
                                         specfem::element::boundary_tag::none>;

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
