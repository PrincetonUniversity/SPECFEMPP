#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim2_compute_source_contribution_acoustic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_source_contribution_acoustic
 * @brief Compute source contribution for 2D acoustic isotropic media.
 *
 * Implements pressure source contribution for acoustic wave propagation.
 *
 * **Source equation:**
 * \f$ \ddot{\chi} = -\frac{S(t) \cdot L(\mathbf{x})}{\kappa} \f$
 *
 * where:
 * - \f$ \chi \f$: acoustic potential (related to pressure: \f$ p = -\ddot{\chi}
 * \f$)
 * - \f$ S(t) \f$: source time function
 * - \f$ L(\mathbf{x}) \f$: Lagrange interpolant for spatial localization
 * - \f$ \kappa \f$: bulk modulus of fluid
 *
 * **Sign convention:**
 * Negative sign ensures \f$ +S(t) \f$ produces \f$ +p \f$ in pressure field.
 *
 * @tparam PointSourceType Point-wise source parameters
 * @tparam PointPropertiesType Point-wise material properties
 * @param point_source Source parameters (STF, interpolant)
 * @param point_properties Material properties (κ)
 * @return Acceleration contribution for acoustic potential
 */
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
