#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

/**
 * @file dirichlet.hpp
 * @brief Dirichlet boundary conditions
 *
 */

namespace specfem {
namespace boundary_conditions {

using acoustic_free_surface_type = std::integral_constant<
    specfem::element::boundary_tag,
    specfem::element::boundary_tag::acoustic_free_surface>;

/**
 * @brief Apply Dirichlet boundary conditions (non-SIMD)
 *
 * Enforces homogeneous Dirichlet boundary conditions by setting the
 * acceleration to zero on the boundary.
 *
 * \f[
 * \mathbf{a} = \mathbf{0} \quad \text{on } \Gamma
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param acceleration Acceleration object
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_apply_boundary_conditions(
    const acoustic_free_surface_type &, const PointBoundaryType &boundary,
    const PointPropertyType &, const PointFieldType &,
    PointAccelerationType &acceleration) {

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  if (boundary.tag != tag)
    return;

  constexpr int components = PointFieldType::components;

  for (int icomp = 0; icomp < components; ++icomp)
    acceleration(icomp) = 0.0;

  return;
};

/**
 * @brief Apply Dirichlet boundary conditions (SIMD)
 *
 * Enforces homogeneous Dirichlet boundary conditions by setting the
 * acceleration to zero on the boundary.
 *
 * \f[
 * \mathbf{a} = \mathbf{0} \quad \text{on } \Gamma
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param acceleration Acceleration object
 */
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_apply_boundary_conditions(
    const acoustic_free_surface_type &, const PointBoundaryType &boundary,
    const PointPropertyType &, const PointFieldType &,
    PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  constexpr int components = PointFieldType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointAccelerationType::simd::mask_type;

  using simd_type = typename PointAccelerationType::simd::datatype;

  for (std::size_t icomp = 0; icomp < components; ++icomp) {
    simd_type result([&](std::size_t lane) {
      return (boundary.tag[lane] == tag) ? 0.0 : acceleration(icomp)[lane];
    });

    acceleration(icomp) = result;
  }

  return;
};

/**
 * @brief Compute mass matrix terms for Dirichlet boundary conditions
 *
 * No additional mass matrix terms are required for Dirichlet boundaries.
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointMassMatrixType Point mass matrix type
 * @param boundary Boundary object
 * @param mass_matrix Mass matrix object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FORCEINLINE_FUNCTION void impl_compute_mass_matrix_terms(
    const acoustic_free_surface_type &, const type_real,
    const PointBoundaryType &boundary, const PointPropertyType &,
    PointMassMatrixType &mass_matrix) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  // Do nothing
  return;
};

/**
 * @brief Apply Dirichlet boundary conditions (non-SIMD)
 *
 * Enforces homogeneous Dirichlet boundary conditions by setting the
 * acceleration to zero on the boundary.
 *
 * \f[
 * \mathbf{a} = \mathbf{0} \quad \text{on } \Gamma
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param acceleration Acceleration object
 */
template <
    typename PointBoundaryType, typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_INLINE_FUNCTION void
impl_apply_boundary_conditions(const acoustic_free_surface_type &,
                               const PointBoundaryType &boundary,
                               PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  constexpr int components = PointAccelerationType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  for (int icomp = 0; icomp < components; ++icomp)
    acceleration(icomp) = 0.0;

  return;
}

/**
 * @brief Apply Dirichlet boundary conditions (SIMD)
 *
 * Enforces homogeneous Dirichlet boundary conditions by setting the
 * acceleration to zero on the boundary.
 *
 * \f[
 * \mathbf{a} = \mathbf{0} \quad \text{on } \Gamma
 * \f]
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param acceleration Acceleration object
 */
template <
    typename PointBoundaryType, typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_INLINE_FUNCTION void
impl_apply_boundary_conditions(const acoustic_free_surface_type &,
                               const PointBoundaryType &boundary,
                               PointAccelerationType &acceleration) {
  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::acoustic_free_surface,
                "Boundary tag must be acoustic_free_surface");

  constexpr int components = PointAccelerationType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointAccelerationType::simd::mask_type;
  using simd_type = typename PointAccelerationType::simd::datatype;
  for (std::size_t icomp = 0; icomp < components; ++icomp) {
    simd_type result([&](std::size_t lane) {
      return (boundary.tag[lane] == tag) ? 0.0 : acceleration(icomp)[lane];
    });

    acceleration(icomp) = result;
  }

  return;
}

} // namespace boundary_conditions
} // namespace specfem
