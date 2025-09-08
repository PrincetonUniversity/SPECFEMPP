#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim3>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim3,
                                   specfem::element::medium_tag::elastic,
                                   using_simd>;

  PointAccelerationType result;

  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0);
  result(1) = point_source.stf(1) * point_source.lagrange_interpolant(1);
  result(2) = point_source.stf(2) * point_source.lagrange_interpolant(2);

  return result;
}

} // namespace medium
} // namespace specfem
