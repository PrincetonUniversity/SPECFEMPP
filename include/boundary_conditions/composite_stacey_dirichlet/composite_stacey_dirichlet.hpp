#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace boundary_conditions {

using composite_stacey_dirichlet_type = std::integral_constant<
    specfem::element::boundary_tag,
    specfem::element::boundary_tag::composite_stacey_dirichlet>;

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(
    const composite_stacey_dirichlet_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration);

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FUNCTION void impl_compute_mass_matrix_terms(
    const composite_stacey_dirichlet_type &, const type_real dt,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    PointMassMatrixType &mass_matrix);

} // namespace boundary_conditions
} // namespace specfem
