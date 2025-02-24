#pragma once

#include "algorithms/dot.hpp"
#include "boundary_conditions/boundary_conditions.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "stacey.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace {
using elastic_sv_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_sv>;

using acoustic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::acoustic>;

using isotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::isotropic>;

using anisotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::anisotropic>;

// Elastic Isotropic Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const acoustic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::acoustic,
                "Medium tag must be acoustic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  traction(0) += static_cast<type_real>(-1.0) * factor *
                 property.rho_vpinverse * field.velocity(0);

  // Do nothing
  return;
}

// Elastic Isotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const acoustic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::acoustic,
                "Medium tag must be acoustic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr int components = PointFieldType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor *
                        property.rho_vpinverse * field.velocity(0);

  return;
}

// Elastic Isotropic Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sv_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto vn =
      specfem::algorithms::dot(field.velocity, boundary.edge_normal);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp - property.rho_vs)) +
                    field.velocity(icomp) * property.rho_vs;
  }

  traction(0) += static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                 boundary.edge_weight;
  traction(1) += static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                 boundary.edge_weight;

  return;
}

// Elastic Isotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sv_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr int components = PointFieldType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto vn =
      specfem::algorithms::dot(field.velocity, boundary.edge_normal);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp - property.rho_vs)) +
                    field.velocity(icomp) * property.rho_vs;
  }

  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                        boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(1)) =
      traction(1) + static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                        boundary.edge_weight;

  return;
}

// Elastic Anisotropic stacey boundary conditions not using SIMD typess
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sv_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto vn =
      specfem::algorithms::dot(field.velocity, boundary.edge_normal);

  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp - property.rho_vs)) +
                    field.velocity(icomp) * property.rho_vs;
  }

  traction(0) += static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                 boundary.edge_weight;
  traction(1) += static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                 boundary.edge_weight;

  return;
}

// Elastic Anisotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sv_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  constexpr int components = PointFieldType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto vn =
      specfem::algorithms::dot(field.velocity, boundary.edge_normal);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp - property.rho_vs)) +
                    field.velocity(icomp) * property.rho_vs;
  }

  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                        boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(1)) =
      traction(1) + static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                        boundary.edge_weight;

  return;
}
} // namespace

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::boundary_conditions::impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration) {

  constexpr auto MediumTag = PointPropertyType::medium_tag;
  constexpr auto PropertyTag = PointPropertyType::property_tag;

  impl_enforce_traction(
      std::integral_constant<specfem::element::medium_tag, MediumTag>{},
      std::integral_constant<specfem::element::property_tag, PropertyTag>{},
      boundary, property, field, acceleration.acceleration);

  return;
}

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FORCEINLINE_FUNCTION void
specfem::boundary_conditions::impl_compute_mass_matrix_terms(
    const stacey_type &, const type_real dt, const PointBoundaryType &boundary,
    const PointPropertyType &property, PointMassMatrixType &mass_matrix) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  constexpr static auto MediumTag = PointPropertyType::medium_tag;
  constexpr static auto PropertyTag = PointPropertyType::property_tag;
  constexpr static bool using_simd = PointPropertyType::simd::using_simd;

  using PointVelocityType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            true, false, false, using_simd>;

  using PointAccelerationType =
      specfem::point::field<specfem::dimension::type::dim2, MediumTag, false,
                            false, true, false, using_simd>;

  using ViewType = typename PointVelocityType::ViewType;

  using datatype = typename ViewType::value_type;

  const datatype velocity_factor(static_cast<type_real>(-1.0) * dt *
                                 static_cast<type_real>(0.5));

  PointVelocityType velocity(velocity_factor);

  impl_enforce_traction(
      std::integral_constant<specfem::element::medium_tag, MediumTag>{},
      std::integral_constant<specfem::element::property_tag, PropertyTag>{},
      boundary, property, velocity, mass_matrix.mass_matrix);
}
