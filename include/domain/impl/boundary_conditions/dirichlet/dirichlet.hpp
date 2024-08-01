#pragma once

#include "enumerations/boundary.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace boundary_conditions {

using acoustic_free_surface_type = std::integral_constant<
    specfem::element::boundary_tag,
    specfem::element::boundary_tag::acoustic_free_surface>;

template <
    typename PointBoundaryType, typename PointFieldType,
    typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_apply_boundary_conditions(
    const acoustic_free_surface_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  if (boundary.tags[0] != PointBoundaryType::boundary_tag)
    return;

  constexpr int components = PointFieldType::components;

  for (int icomp = 0; icomp < components; ++icomp)
    acceleration.acceleration(icomp) = 0.0;

  return;
};

template <
    typename PointBoundaryType, typename PointFieldType,
    typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_apply_boundary_conditions(
    const acoustic_free_surface_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  constexpr int components = PointFieldType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tags[lane] == tag; });

  for (int icomp = 0; icomp < components; ++icomp)
    Kokkos::Experimental::where(mask, acceleration.acceleration(icomp)) = 0.0;

  return;
};

} // namespace boundary_conditions
} // namespace impl
} // namespace domain
} // namespace specfem
