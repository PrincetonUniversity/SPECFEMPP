#pragma once

#include "specfem/data_access.hpp"
#include "specfem/medium/dim2/acoustic/isotropic/source.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/source.hpp"
#include "specfem/medium/dim2/elastic/isotropic/source.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/source.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/source.hpp"
#include "specfem/medium/dim3/acoustic/isotropic/source.hpp"
#include "specfem/medium/dim3/elastic/isotropic/source.hpp"
#include "specfem/medium/dim3/elastic/isotropic_cosserat/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {
// clang-format off
/**
 * @brief Compute source contribution for seismic wave simulation.
 *
 * Generic source computation interface that dispatches to medium-specific
 * implementations. Computes force/moment contributions from seismic sources
 * (earthquakes, explosions, etc.) at GLL nodes based on source type and
 * material properties.
 *
 * @tparam PointSourceType Point-wise source parameters container
 * @tparam PointPropertiesType Point-wise material properties container
 * @param point_source Source parameters at quadrature point (specifies source
 * time function and lagrange interpolants)
 * @param point_properties Material properties at quadrature point
 * @return Source force/moment contribution for wave equation
 *
 * @code{.cpp}
 * // Example usage for 2D elastic isotropic medium
 * using Source     = specfem::point::source<dim2, elastic, false>;
 * using Properties = specfem::point::properties<specfem::tags::Tags<dim2, elastic, isotropic, false>>;
 *
 * Source     src   = ...;  // Initialize source parameters
 * Properties props = ...;  // Initialize material properties
 *
 * auto source_contribution =
 *     specfem::medium_physics::compute_source_contribution(src, props);
 * @endcode
 */
// clang-format on
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
      std::integral_constant<specfem::element::dimension_tag,
                             PointSourceType::dimension_tag>;

  using medium_dispatch = std::integral_constant<specfem::element::medium_tag,
                                                 PointSourceType::medium_tag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag,
                             PointPropertiesType::property_tag>;

  return specfem::medium_physics::impl_compute_source_contribution(
      dimension_dispatch(), medium_dispatch(), property_dispatch(),
      point_source, point_properties);
}

} // namespace medium_physics
} // namespace specfem
