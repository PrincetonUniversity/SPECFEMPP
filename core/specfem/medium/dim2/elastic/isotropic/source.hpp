#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim2_compute_source_contribution_elastic_isotropic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_source_contribution_elastic_isotropic
 * @brief Compute source contribution for 2D elastic isotropic P-SV media.
 *
 * Implements force source contribution for P-SV wave propagation in
 * isotropic elastic media. Sources inject body forces that generate
 * both P and S waves with uniform propagation velocities.
 *
 * **Source equations:**
 * - \f$ \ddot{u}_x = S_x(t) \cdot L_x(\mathbf{x}) \f$
 * - \f$ \ddot{u}_z = S_z(t) \cdot L_z(\mathbf{x}) \f$
 *
 * where:
 * - \f$ S_x(t), S_z(t) \f$: source time functions (x,z components)
 * - \f$ L_x(\mathbf{x}), L_z(\mathbf{x}) \f$: Lagrange interpolants
 * - \f$ u_x, u_z \f$: displacement components in P-SV plane
 *
 * @param point_source Source parameters (STF components, interpolants)
 * @param point_properties Material properties (unused for force sources)
 * @return Acceleration contributions [\f$\ddot{u}_x, \ddot{u}_z\f$]
 */
template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_psv,
                                   using_simd>;

  PointAccelerationType result;

  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0);
  result(1) = point_source.stf(1) * point_source.lagrange_interpolant(1);

  return result;
}

/**
 * @ingroup specfem_medium_dim2_compute_source_contribution_elastic_isotropic
 * @brief Compute source contribution for 2D elastic isotropic SH media.
 *
 * Implements force source contribution for SH wave propagation in
 * isotropic elastic media. Sources inject anti-plane body forces
 * that generate shear horizontal waves.
 *
 * **Source equation:**
 * - \f$ \ddot{u}_y = S_y(t) \cdot L_y(\mathbf{x}) \f$
 *
 * where:
 * - \f$ S_y(t) \f$: source time function (y component)
 * - \f$ L_y(\mathbf{x}) \f$: Lagrange interpolant
 * - \f$ u_y \f$: anti-plane displacement
 *
 * @param point_source Source parameters (STF, interpolant)
 * @param point_properties Material properties (unused for force sources)
 * @return Acceleration contribution [\f$\ddot{u}_y\f$]
 */
template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_sh>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim2,
                                   specfem::element::medium_tag::elastic_sh,
                                   using_simd>;

  PointAccelerationType result;

  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0);

  return result;
}

} // namespace medium
} // namespace specfem
