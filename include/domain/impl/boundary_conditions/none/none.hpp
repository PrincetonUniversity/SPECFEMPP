#pragma once

#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace boundary_conditions {

using none_type = std::integral_constant<specfem::element::boundary_tag,
                                         specfem::element::boundary_tag::none>;

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(const none_type &,
                                                    const PointBoundaryType &,
                                                    const PointPropertyType &,
                                                    const PointFieldType &,
                                                    PointAccelerationType &) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::none,
                "Boundary tag must be none");

  // Do nothing
  return;
}

} // namespace boundary_conditions
} // namespace impl
} // namespace domain
} // namespace specfem
