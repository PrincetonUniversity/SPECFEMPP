#pragma once

#include "specfem/boundary_conditions.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include "stacey.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace boundary_conditions {
namespace impl {

using elastic_psv_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_psv>;

using elastic_sh_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_sh>;

using elastic_psv_t_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::elastic_psv_t>;

using acoustic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::acoustic>;

using poroelastic_type =
    std::integral_constant<specfem::element::medium_tag,
                           specfem::element::medium_tag::poroelastic>;

using isotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::isotropic>;

using anisotropic_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::anisotropic>;

using isotropic_cosserat_type =
    std::integral_constant<specfem::element::property_tag,
                           specfem::element::property_tag::isotropic_cosserat>;

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void impl_base_elastic_psv_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction) {

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto vn = velocity.get_data() * boundary.edge_normal;
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp() - property.rho_vs())) +
                    velocity(icomp) * property.rho_vs();
  }

  traction(0) += static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                 boundary.edge_weight;
  traction(1) += static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                 boundary.edge_weight;

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void impl_base_elastic_psv_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction) {

  constexpr int components = PointVelocityType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto vn = velocity.get_data() * boundary.edge_normal;
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp() - property.rho_vs())) +
                    velocity(icomp) * property.rho_vs();
  }

  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                        boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(1)) =
      traction(1) + static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                        boundary.edge_weight;

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void impl_base_elastic_sh_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction) {

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  traction(0) +=
      static_cast<type_real>(-1.0) * factor * property.rho_vs() * velocity(0);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void impl_base_elastic_sh_traction(
    const PointBoundaryType &boundary, const PointPropertyType &property,
    const PointVelocityType &velocity, ViewType &traction) {

  constexpr int components = PointVelocityType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) +
      static_cast<type_real>(-1.0) * factor * property.rho_vs() * velocity(0);

  return;
}

// Elastic PSV (Cosserat) Boundary Conditions not using SIMD types

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void
impl_base_elastic_psv_t_traction(const PointBoundaryType &boundary,
                              const PointPropertyType &property,
                              const PointVelocityType &velocity, ViewType &traction) {

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;
  // can't use specfem::algorithms::dot here because
  // velocity is a 3-component vector, and edge_normal is a 2-component
  const auto vn =
      velocity(0) * boundary.edge_normal(0) +
      velocity(1) * boundary.edge_normal(1);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();
    // these are the high frequency limit group velocities for \beta_\pm
    // because of the mode switching phenomenon, they are correct at high
    // frequencies (though it's not guaranteed which branch corresponds to
    // displacement and which to rotation), but they don't necessarily
    // work well at low frequency
  const auto rho_vs = property.rho() * Kokkos::sqrt((property.mu() +
                        property.nu()) / property.rho());
  const auto j_vt = property.j() * Kokkos::sqrt((property.mu_c() +
                        property.nu_c()) / property.j());
  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp() - rho_vs)) +
                    velocity(icomp) * rho_vs;
  }

  traction(0) += static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                 boundary.edge_weight;
  traction(1) += static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                 boundary.edge_weight;
  traction(2) += static_cast<type_real>(-1.0) * j_vt *
                 velocity(2) * jacobian1d * boundary.edge_weight;

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void
impl_base_elastic_psv_t_traction(const PointBoundaryType &boundary,
                              const PointPropertyType &property,
                              const PointVelocityType &velocity, ViewType &traction) {

  constexpr int components = PointVelocityType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;
  // can't use specfem::algorithms::dot here because
  // velocity is a 3-component vector, and edge_normal is a 2-component
  const auto vn =
        velocity(0) * boundary.edge_normal(0) +
        velocity(1) * boundary.edge_normal(1);
  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

    // these are the high frequency limit group velocities for \beta_\pm
    // because of the mode switching phenomenon, they are correct at high
    // frequencies (though it's not guaranteed which branch corresponds to
    // displacement and which to rotation), but they don't necessarily
    // work well at low frequency
  const auto rho_vs = property.rho() * Kokkos::sqrt((property.mu() +
                            property.nu()) / property.rho());
  const auto j_vt = property.j() * Kokkos::sqrt((property.mu_c() +
                            property.nu_c()) / property.j());

  using datatype = std::remove_const_t<decltype(jacobian1d)>;

  datatype factor[2];

  for (int icomp = 0; icomp < 2; ++icomp) {
    factor[icomp] = ((vn * dn(icomp) / (jacobian1d * jacobian1d)) *
                     (property.rho_vp() - rho_vs)) +
                    velocity(icomp) * rho_vs;
  }

  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor[0] * jacobian1d *
                        boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(1)) =
      traction(1) + static_cast<type_real>(-1.0) * factor[1] * jacobian1d *
                        boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(2)) =
      traction(2) + static_cast<type_real>(-1.0) * j_vt *
                    velocity(2) * jacobian1d * boundary.edge_weight;

  return;
}

// Acoustic Isotropic Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void
impl_enforce_traction(const acoustic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

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
                 property.rho_vpinverse() * velocity(0);

  // Do nothing
  return;
}

// Acoustic Isotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int>>
KOKKOS_FUNCTION void
impl_enforce_traction(const acoustic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::acoustic,
                "Medium tag must be acoustic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr int components = PointVelocityType::components;
  constexpr auto tag = PointBoundaryType::boundary_tag;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto factor = boundary.edge_weight * boundary.edge_normal.l2_norm();

  // Apply Stacey boundary condition
  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) + static_cast<type_real>(-1.0) * factor *
                        property.rho_vpinverse() * velocity(0);

  return;
}

// Poroelastic Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const poroelastic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::poroelastic,
                "Medium tag must be poroelastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  if (boundary.tag != tag)
    return;

  const auto rho = (property.rho_bar() -
                    property.phi() / property.tortuosity() * property.rho_f());
  const auto rho_vpI = property.vpI() * rho;   // @f$ \rho v_{pI} @f$
  const auto rho_vpII = property.vpII() * rho; // @f$ \rho v_{pII} @f$
  const auto rho_vs = property.vs() * rho;     // @f$ \rho v_{s} @f$

  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  const auto vn = velocity(0) * dn(0) + velocity(1) * dn(1);
  const auto vnf = velocity(1) * dn(0) + velocity(2) * dn(1);

  const auto tsx =
      rho_vpI * vn * dn(0) +
      rho_vs * (velocity(0) - vn * dn(0)); /// Solid traction X component
  const auto tsz =
      rho_vpI * vn * dn(1) +
      rho_vs * (velocity(1) - vn * dn(1)); /// Solid traction Z component

  const auto tfx =
      rho_vpII * vnf * dn(0) -
      rho_vs * (velocity(2) - vn * dn(0)); /// Fluid traction X component
  const auto tfz =
      rho_vpII * vnf * dn(1) -
      rho_vs * (velocity(3) - vn * dn(1)); /// Fluid traction Z component

  traction(0) +=
      static_cast<type_real>(-1.0) * tsx * jacobian1d * boundary.edge_weight;
  traction(1) +=
      static_cast<type_real>(-1.0) * tsz * jacobian1d * boundary.edge_weight;

  traction(2) +=
      static_cast<type_real>(-1.0) * tfx * jacobian1d * boundary.edge_weight;
  traction(3) +=
      static_cast<type_real>(-1.0) * tfz * jacobian1d * boundary.edge_weight;

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const poroelastic_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::poroelastic,
                "Medium tag must be poroelastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  constexpr static auto tag = PointBoundaryType::boundary_tag;

  constexpr int components = PointVelocityType::components;

  using mask_type = typename PointBoundaryType::simd::mask_type;

  mask_type mask([&](std::size_t lane) { return boundary.tag[lane] == tag; });

  if (Kokkos::Experimental::none_of(mask))
    return;

  const auto rho = (property.rho_bar() -
                    property.phi() / property.tortuosity() * property.rho_f());
  const auto rho_vpI = property.vpI() * rho;   // @f$ \rho v_{pI} @f$
  const auto rho_vpII = property.vpII() * rho; // @f$ \rho v_{pII} @f$
  const auto rho_vs = property.vs() * rho;     // @f$ \rho v_{s} @f$

  const auto &dn = boundary.edge_normal;

  const auto jacobian1d = dn.l2_norm();

  const auto vn = velocity(0) * dn(0) + velocity(1) * dn(1);
  const auto vnf = velocity(1) * dn(0) + velocity(2) * dn(1);

  const auto tsx =
      rho_vpI * vn * dn(0) +
      rho_vs * (velocity(0) - vn * dn(0)); /// Solid traction X component
  const auto tsz =
      rho_vpI * vn * dn(1) +
      rho_vs * (velocity(1) - vn * dn(1)); /// Solid traction Z component

  const auto tfx =
      rho_vpII * vnf * dn(0) -
      rho_vs * (velocity(2) - vn * dn(0)); /// Fluid traction X component
  const auto tfz =
      rho_vpII * vnf * dn(1) -
      rho_vs * (velocity(3) - vn * dn(1)); /// Fluid traction Z component

  // Apply Stacey boundary condition
  Kokkos::Experimental::where(mask, traction(0)) =
      traction(0) +
      static_cast<type_real>(-1.0) * tsx * jacobian1d * boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(1)) =
      traction(1) +
      static_cast<type_real>(-1.0) * tsz * jacobian1d * boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(2)) =
      traction(2) +
      static_cast<type_real>(-1.0) * tfx * jacobian1d * boundary.edge_weight;

  Kokkos::Experimental::where(mask, traction(3)) =
      traction(3) +
      static_cast<type_real>(-1.0) * tfz * jacobian1d * boundary.edge_weight;

  return;
}

// Elastic Isotropic Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  impl_base_elastic_psv_traction(boundary, property, velocity, traction);

  return;
}

// Elastic Isotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  impl_base_elastic_psv_traction(boundary, property, velocity, traction);

  return;
}

// Elastic Anisotropic stacey boundary conditions not using SIMD typess
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  impl_base_elastic_psv_traction(boundary, property, velocity, traction);

  return;
}

// Elastic Anisotropic Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  impl_base_elastic_psv_traction(boundary, property, velocity, traction);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sh_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sh,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  impl_base_elastic_sh_traction(boundary, property, velocity, traction);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sh_type &, const isotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sh,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic,
                "Property tag must be isotropic");

  impl_base_elastic_sh_traction(boundary, property, velocity, traction);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sh_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sh,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  impl_base_elastic_sh_traction(boundary, property, velocity, traction);

  return;
}

template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointVelocityType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_sh_type &, const anisotropic_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointVelocityType &velocity, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_sh,
                "Medium tag must be elastic");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::anisotropic,
                "Property tag must be anisotropic");

  impl_base_elastic_sh_traction(boundary, property, velocity, traction);

  return;
}

// Elastic Isotropic Cosserat Stacey Boundary Conditions not using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<!PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_t_type &, const isotropic_cosserat_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv_t,
                "Medium tag must be elastic cosserat");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic_cosserat,
                "Property tag must be isotropic cosserat");

  impl_base_elastic_psv_t_traction(boundary, property, field, traction);

  return;
}

// Elastic Isotropic Cosserat Stacey Boundary Conditions using SIMD types
template <
    typename PointBoundaryType, typename PointPropertyType,
    typename PointFieldType, typename ViewType,
    typename std::enable_if_t<PointBoundaryType::simd::using_simd, int> = 0>
KOKKOS_FUNCTION void
impl_enforce_traction(const elastic_psv_t_type &, const isotropic_cosserat_type &,
                      const PointBoundaryType &boundary,
                      const PointPropertyType &property,
                      const PointFieldType &field, ViewType &traction) {

  static_assert(PointBoundaryType::boundary_tag ==
                    specfem::element::boundary_tag::stacey,
                "Boundary tag must be stacey");

  static_assert(PointPropertyType::medium_tag ==
                    specfem::element::medium_tag::elastic_psv_t,
                "Medium tag must be elastic cosserat");

  static_assert(PointPropertyType::property_tag ==
                    specfem::element::property_tag::isotropic_cosserat,
                "Property tag must be isotropic cosserat");

  impl_base_elastic_psv_t_traction(boundary, property, field, traction);

  return;
}
} // namespace impl
} // namespace boundary_conditions
} // namespace specfem

template <typename PointBoundaryType, typename PointPropertyType,
          typename PointVelocityType, typename PointAccelerationType>
KOKKOS_FUNCTION void
specfem::boundary_conditions::impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointVelocityType &velocity,
    PointAccelerationType &acceleration) {

  constexpr auto MediumTag = PointPropertyType::medium_tag;
  constexpr auto PropertyTag = PointPropertyType::property_tag;

  impl::impl_enforce_traction(
      std::integral_constant<specfem::element::medium_tag, MediumTag>{},
      std::integral_constant<specfem::element::property_tag, PropertyTag>{},
      boundary, property, velocity, acceleration);

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
      specfem::point::velocity<specfem::dimension::type::dim2, MediumTag,
                               using_simd>;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2, MediumTag,
                                   using_simd>;

  using ViewType = typename PointVelocityType::value_type;

  using datatype = typename ViewType::value_type;

  const datatype velocity_factor(static_cast<type_real>(-1.0) * dt *
                                 static_cast<type_real>(0.5));

  PointVelocityType velocity(velocity_factor);

  impl::impl_enforce_traction(
      std::integral_constant<specfem::element::medium_tag, MediumTag>{},
      std::integral_constant<specfem::element::property_tag, PropertyTag>{},
      boundary, property, velocity, mass_matrix);
}
