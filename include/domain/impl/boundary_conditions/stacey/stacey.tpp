#pragma once

#include "algorithms/dot.hpp"
#include "domain/impl/boundary_conditions/boundary_conditions.hpp"
#include "enumerations/medium.hpp"
#include "stacey.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace {
using elastic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic>;

using acoustic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::acoustic>;

using isotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::isotropic>;

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_apply_boundary_conditions_stacey(
    const acoustic_type &, const isotropic_type &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::acoustic,
                "Medium tag must be acoustic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  if (boundary.tag != PointBoundaryType::boundary_tag)
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  acceleration.acceleration(0) += static_cast<type_real>(-1.0) * factor *
                                  property.rho_vpinverse * field.velocity(0);

  // Do nothing
  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_apply_boundary_conditions_stacey(
    const acoustic_type &, const isotropic_type &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointFieldType &field, PointAccelerationType &acceleration) {

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
  Kokkos::Experimental::where(mask, acceleration.acceleration(0)) =
      acceleration.acceleration(0) + static_cast<type_real>(-1.0) * factor *
                                         property.rho_vpinverse *
                                         field.velocity(0);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_apply_boundary_conditions_stacey(
    const elastic_type &, const isotropic_type &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  if (boundary.tag != PointBoundaryType::boundary_tag)
    return;

  const auto vn =
      specfem::algorithms::dot(field.velocity, boundary.edge_normal);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = typename PointAccelerationType::simd::datatype;

  datatype traction[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    traction[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                       (property.rho_vp - property.rho_vs)) +
                      field.velocity(icomp) * property.rho_vs;
  }

  acceleration.acceleration(0) += static_cast<type_real>(-1.0) * traction[0] *
                                  jacobian1d * boundary.edge_weight;
  acceleration.acceleration(1) += static_cast<type_real>(-1.0) * traction[1] *
                                  jacobian1d * boundary.edge_weight;

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename PointAccelerationType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void impl_apply_boundary_conditions_stacey(
    const elastic_type &, const isotropic_type &,
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointFieldType &field, PointAccelerationType &acceleration) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic,
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

  using datatype = typename PointAccelerationType::simd::datatype;

  datatype traction[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    traction[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                       (property.rho_vp - property.rho_vs)) +
                      field.velocity(icomp) * property.rho_vs;
  }

  Kokkos::Experimental::where(mask, acceleration.acceleration(0)) =
      acceleration.acceleration(0) + static_cast<type_real>(-1.0) *
                                         traction[0] * jacobian1d *
                                         boundary.edge_weight;

  Kokkos::Experimental::where(mask, acceleration.acceleration(1)) =
      acceleration.acceleration(1) + static_cast<type_real>(-1.0) *
                                         traction[1] * jacobian1d *
                                         boundary.edge_weight;

  return;
}
} // namespace

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::domain::impl::boundary_conditions::impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration) {

  constexpr auto medium_tag = PointPropertyType::medium_tag;
  constexpr auto property_tag = PointPropertyType::property_tag;

  impl_apply_boundary_conditions_stacey(
      std::integral_constant<specfem::element::medium_tag, medium_tag>{},
      std::integral_constant<specfem::element::property_tag, property_tag>{},
      boundary, property, field, acceleration);

  return;
}
