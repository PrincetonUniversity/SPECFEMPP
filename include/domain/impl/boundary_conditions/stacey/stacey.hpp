#pragma once

#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace boundary_conditions {

using stacey_type =
    std::integral_constant<specfem::element::boundary_tag,
                           specfem::element::boundary_tag::stacey>;

template <typename PointBoundaryType, typename PointFieldType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration);

} // namespace boundary_conditions
} // namespace impl
} // namespace domain
} // namespace specfem
