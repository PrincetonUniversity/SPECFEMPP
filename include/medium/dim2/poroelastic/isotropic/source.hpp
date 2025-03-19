#pragma once

#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourcesType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourcesType &point_source,
    const PointPropertiesType &point_properties) {

  using PointAccelerationType =
      specfem::point::field<PointPropertiesType::dimension,
                            PointPropertiesType::medium_tag, false, false, true,
                            false, PointPropertiesType::simd::using_simd>;

  const auto rho_bar = point_properties.rho_bar();
  const auto phi = point_properties.phi();
  const auto rho_f = point_properties.rho_f();
  const auto tort = point_properties.tortuosity();

  const type_real solid_factor = static_cast<type_real>(1.0) - phi / tort;
  const type_real fluid_factor = static_cast<type_real>(1.0) - rho_f / rho_bar;

  PointAccelerationType result;

  result.acceleration(0) =
      solid_factor * point_source.lagrange_interpolant(0) * point_source.stf(0);
  result.acceleration(1) =
      solid_factor * point_source.lagrange_interpolant(1) * point_source.stf(1);
  result.acceleration(2) =
      fluid_factor * point_source.lagrange_interpolant(2) * point_source.stf(2);
  result.acceleration(3) =
      fluid_factor * point_source.lagrange_interpolant(3) * point_source.stf(3);

  return result;
}

} // namespace medium
} // namespace specfem
