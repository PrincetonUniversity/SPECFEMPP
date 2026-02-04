#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_physics {

/**
 * @defgroup specfem_medium_dim3_compute_source_contribution_acoustic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_source_contribution_acoustic
 * @brief Compute source contribution for 3D acoustic isotropic media.
 *
 * Calculates acceleration contribution from point sources in acoustic media.
 * The source term is scaled by inverse bulk modulus and interpolated to
 * quadrature points:
 *
 * \f$a = \frac{S \cdot L}{\kappa}\f$
 *
 * where \f$S\f$ is the source time function, \f$L\f$ is the Lagrange
 * interpolant, and \f$\kappa\f$ is the bulk modulus.
 *
 * @tparam PointSourceType Source data structure containing STF and interpolants
 * @tparam PointPropertiesType Material properties containing bulk modulus
 * @param point_source Source parameters (STF, interpolants)
 * @param point_properties Material properties (\f$\kappa\f$)
 * @return Acceleration contribution from source
 */
template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim3> /*unused*/,
    const std::integral_constant<
        specfem::element::medium_tag,
        specfem::element::medium_tag::acoustic> /*unused*/,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic> /*unused*/,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {
  constexpr bool using_simd = PointPropertiesType::simd::using_simd;

  using PointAccelerationType =
      specfem::point::acceleration<specfem::dimension::type::dim3,
                                   specfem::element::medium_tag::acoustic,
                                   using_simd>;

  PointAccelerationType result;
  result(0) = point_source.stf(0) * point_source.lagrange_interpolant(0) /
              point_properties.kappa();
  return result;
}

} // namespace specfem::medium_physics
