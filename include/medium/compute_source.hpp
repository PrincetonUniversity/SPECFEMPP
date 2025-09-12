#pragma once

#include "dim2/acoustic/isotropic/source.hpp"
#include "dim2/elastic/anisotropic/source.hpp"
#include "dim2/elastic/isotropic/source.hpp"
#include "dim2/elastic/isotropic_cosserat/source.hpp"
#include "dim2/poroelastic/isotropic/source.hpp"
#include "dim3/elastic/isotropic/source.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto
compute_source_contribution(const PointSourceType &point_source,
                            const PointPropertiesType &point_properties) {

  static_assert(specfem::data_access::is_point<PointSourceType>::value &&
                    specfem::data_access::is_source<PointSourceType>::value,
                "point_source is not a point source type");

  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "point_properties is not a point properties type");

  static_assert(PointSourceType::dimension_tag ==
                    PointPropertiesType::dimension_tag,
                "point_source and point_properties have different dimensions");

  static_assert(PointSourceType::medium_tag == PointPropertiesType::medium_tag,
                "point_source and point_properties have different medium tags");

  static_assert(!PointPropertiesType::simd::using_simd,
                "point_properties should be a non SIMD type for this function");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             PointSourceType::dimension_tag>;

  using medium_dispatch = std::integral_constant<specfem::element::medium_tag,
                                                 PointSourceType::medium_tag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag,
                             PointPropertiesType::property_tag>;

  return specfem::medium::impl_compute_source_contribution(
      dimension_dispatch(), medium_dispatch(), property_dispatch(),
      point_source, point_properties);
}

} // namespace medium
} // namespace specfem
