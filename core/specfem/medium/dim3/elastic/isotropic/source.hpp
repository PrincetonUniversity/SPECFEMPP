#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim3_compute_source_contribution_elastic_isotropic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_source_contribution_elastic_isotropic
 * @brief Compute source contribution for 3D elastic isotropic media.
 *
 * Implements force source contribution for 3D elastic wave propagation
 * in isotropic media. Sources inject body forces that generate both
 * P and S waves with uniform propagation in all directions.
 *
 * **Source equations:**
 * - \f$ \ddot{u}_x = S_x(t) \cdot L_x(\mathbf{x}) \f$
 * - \f$ \ddot{u}_y = S_y(t) \cdot L_y(\mathbf{x}) \f$
 * - \f$ \ddot{u}_z = S_z(t) \cdot L_z(\mathbf{x}) \f$
 *
 * @param point_source Source parameters (STF components, interpolants)
 * @param point_properties Material properties (unused for force sources)
 * @return Acceleration contributions [\f$\ddot{u}_x, \ddot{u}_y, \ddot{u}_z\f$]
 */
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
