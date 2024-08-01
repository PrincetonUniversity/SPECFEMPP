#pragma once

#include "stacey.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include <Kokkos_Core.hpp>

template <typename PointBoundaryType, typename PointFieldType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  // Do nothing
  return;
}
