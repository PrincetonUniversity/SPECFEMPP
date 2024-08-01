#pragma once

#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace boundary_conditions {

using none_type = std::integral_constant<specfem::element::boundary_tag,
                                         specfem::element::boundary_tag::none>;

template <typename PointBoundaryType, typename PointFieldType,
          typename PointAccelerationType>
KOKKOS_FORCEINLINE_FUNCTION void impl_apply_boundary_conditions(
    const none_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration){};

} // namespace boundary_conditions
} // namespace impl
} // namespace domain
} // namespace specfem
