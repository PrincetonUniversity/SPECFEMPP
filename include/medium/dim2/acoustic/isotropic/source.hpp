#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {

  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::acoustic,
                                   using_simd>;

  PointAccelerationType result;

  /* note: for acoustic medium, the source is a pressure source and gets divided
   *       by Kappa of the fluid. The sign is negative because pressure p = -
   *       Chi_dot_dot therefore we need to add minus the source to Chi_dot_dot
   *       to get plus the source in pressure
   */
  result(0) = -point_source.stf(0) * point_source.lagrange_interpolant(0) /
              point_properties.kappa();

  return result;
}

} // namespace medium
} // namespace specfem
