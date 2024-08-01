#pragma once

#include "composite_stacey_dirichlet.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "domain/impl/boundary_conditions/dirichlet/dirichlet.hpp"
#include "domain/impl/boundary_conditions/stacey/stacey.hpp"
#include "enumerations/boundary.hpp"
#include "point/boundary.hpp"
#include <Kokkos_Core.hpp>

template <typename PointBoundaryType, typename PointFieldType,
          typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions(
    const composite_stacey_dirichlet_type &, const PointBoundaryType &boundary,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::composite_stacey_dirichlet,
                "Boundary tag must be composite_stacey_dirichlet");

  constexpr bool using_simd = PointBoundaryType::simd::using_simd;

  const specfem::point::boundary<using_simd,
                                 specfem::element::boundary_tag::stacey>
      stacey_boundary(std::move(boundary));

  impl_apply_boundary_conditions(
      std::integral_constant<specfem::element::boundary_tag,
                             specfem::element::boundary_tag::stacey>(),
      stacey_boundary, field, acceleration);

  const specfem::point::boundary<
      using_simd, specfem::element::boundary_tag::acoustic_free_surface>
      acoustic_free_surface_boundary(std::move(stacey_boundary));

  impl_apply_boundary_conditions(
      std::integral_constant<
          specfem::element::boundary_tag,
          specfem::element::boundary_tag::acoustic_free_surface>(),
      acoustic_free_surface_boundary, field, acceleration);

  return;
}
