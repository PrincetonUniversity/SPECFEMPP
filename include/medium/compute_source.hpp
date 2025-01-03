#pragma once

#include "dim2/acoustic/isotropic/source.hpp"
#include "dim2/elastic/anisotropic/source.hpp"
#include "dim2/elastic/isotropic/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourcesType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto
compute_source_contribution(const PointSourcesType &point_source,
                            const PointPropertiesType &point_properties) {

  static_assert(PointSourcesType::is_point_source,
                "point_source is not a point source type");

  static_assert(PointPropertiesType::is_point_properties,
                "point_properties is not a point properties type");

  static_assert(PointSourcesType::dimension == PointPropertiesType::dimension,
                "point_source and point_properties have different dimensions");

  static_assert(PointSourcesType::medium_tag == PointPropertiesType::medium_tag,
                "point_source and point_properties have different medium tags");

  static_assert(!PointPropertiesType::simd::using_simd,
                "point_properties should be a non SIMD type for this function");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             PointSourcesType::dimension>;

  using medium_dispatch = std::integral_constant<specfem::element::medium_tag,
                                                 PointSourcesType::medium_tag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag,
                             PointPropertiesType::property_tag>;

  return specfem::medium::impl_compute_source_contribution(
      dimension_dispatch(), medium_dispatch(), property_dispatch(),
      point_source, point_properties);
}

} // namespace medium
} // namespace specfem
