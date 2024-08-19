#pragma once

#include "composite_stacey_dirichlet.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "domain/impl/boundary_conditions/dirichlet/dirichlet.hpp"
#include "domain/impl/boundary_conditions/stacey/stacey.hpp"
#include "enumerations/boundary.hpp"
#include "point/boundary.hpp"
#include <Kokkos_Core.hpp>

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions(
    const composite_stacey_dirichlet_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::composite_stacey_dirichlet,
                "Boundary tag must be composite_stacey_dirichlet");

  const auto &stacey_boundary = static_cast<
      const specfem::point::boundary<specfem::element::boundary_tag::stacey,
                                     PointBoundaryType::simd::using_simd> &>(
      boundary);

  impl_apply_boundary_conditions(
      std::integral_constant<specfem::element::boundary_tag,
                             specfem::element::boundary_tag::stacey>(),
      stacey_boundary, property, field, acceleration);

  const auto &acoustic_free_surface_boundary =
      static_cast<const specfem::point::boundary<
          specfem::element::boundary_tag::acoustic_free_surface,
          PointBoundaryType::simd::using_simd> &>(boundary);

  impl_apply_boundary_conditions(
      std::integral_constant<
          specfem::element::boundary_tag,
          specfem::element::boundary_tag::acoustic_free_surface>(),
      acoustic_free_surface_boundary, property, field, acceleration);

  return;
}

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_compute_mass_matrix_terms(
    const composite_stacey_dirichlet_type &, const type_real dt,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    PointMassMatrixType &mass_matrix) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::composite_stacey_dirichlet,
                "Boundary tag must be composite_stacey_dirichlet");

  const auto &stacey_boundary = static_cast<
      const specfem::point::boundary<specfem::element::boundary_tag::stacey,
                                     PointBoundaryType::simd::using_simd> &>(
      boundary);

  impl_compute_mass_matrix_terms(
      std::integral_constant<specfem::element::boundary_tag,
                             specfem::element::boundary_tag::stacey>(),
      dt, stacey_boundary, property, mass_matrix);

  const auto &acoustic_free_surface_boundary =
      static_cast<const specfem::point::boundary<
          specfem::element::boundary_tag::acoustic_free_surface,
          PointBoundaryType::simd::using_simd> &>(boundary);

  impl_compute_mass_matrix_terms(
      std::integral_constant<
          specfem::element::boundary_tag,
          specfem::element::boundary_tag::acoustic_free_surface>(),
      dt, acoustic_free_surface_boundary, property, mass_matrix);

  return;
}
