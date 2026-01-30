#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup
 * specfem_medium_dim2_compute_source_contribution_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup
 * specfem_medium_dim2_compute_source_contribution_elastic_isotropic_cosserat
 * @brief Compute source contribution for 2D elastic isotropic Cosserat media.
 *
 * Implements force and moment source contribution for Cosserat (micropolar)
 * elastic media with rotational degrees of freedom. Sources inject both
 * body forces and body couples to generate extended wave phenomena.
 *
 * **Source equations:**
 * - \f$ \ddot{u}_x = S_x(t) \cdot L_x(\mathbf{x}) \f$
 * - \f$ \ddot{u}_z = S_z(t) \cdot L_z(\mathbf{x}) \f$
 * - \f$ \ddot{\omega}_y = M_y(t) \cdot L_{\omega}(\mathbf{x}) \f$
 *
 * where:
 * - \f$ S_x(t), S_z(t) \f$: body force components
 * - \f$ M_y(t) \f$: body couple (moment about y-axis)
 * - \f$ \omega_y \f$: rotational degree of freedom
 *
 * @param point_source Source parameters (STF components, interpolants)
 * @param point_properties Material properties (unused for force/moment sources)
 * @return Acceleration contributions [\f$\ddot{u}_x, \ddot{u}_z,
 * \ddot{\omega}_y\f$]
 */
template <typename PointSourcesType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointSourcesType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv_t,
                                   using_simd>;

  PointAccelerationType result;

  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0);
  result(1) = point_source.stf(1) * point_source.lagrange_interpolant(1);
  result(2) = point_source.stf(2) * point_source.lagrange_interpolant(2);

  return result;
}

} // namespace medium
} // namespace specfem
