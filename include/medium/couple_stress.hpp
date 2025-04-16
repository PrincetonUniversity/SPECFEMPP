#pragma once

#include "medium/dim2/acoustic/isotropic/couple_stress.hpp"
#include "medium/dim2/elastic/anisotropic/couple_stress.hpp"
#include "medium/dim2/elastic/isotropic/couple_stress.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/couple_stress.hpp"
#include "medium/dim2/poroelastic/isotropic/couple_stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup MediumPhysics
 */

/**
 * @brief Compute the damping term at a quadrature point
 *
 * @ingroup MediumPhysics
 *
 * @tparam PointPropertiesType Material properties at the quadrature point
 * specfem::point::properties
 * @tparam PointVelocityType Velocity at the quadrature point
 * specfem::point::field
 * @tparam PointAccelerationType Acceleration at the quadrature point
 * specfem::point::field
 * @param factor Prefactor for the damping term ($wx * wz * jacobian)
 * @param point_properties Material properties at the quadrature point
 * @param velocity Velocity at the quadrature point
 * @param acceleration Acceleration at the quadrature point
 */
template <typename T, typename PointPartialDerivativesType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void compute_couple_stress(
    const PointPartialDerivativesType &point_partial_derivatives,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  static_assert(PointAccelerationType::isPointFieldType,
                "acceleration is not a point field type");

  static_assert(PointAccelerationType::store_acceleration,
                "acceleration must store acceleration");

  static_assert(
      PointPartialDerivativesType::simd::using_simd ==
          PointAccelerationType::simd::using_simd,
      "point_properties and acceleration have different SIMD settings");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             specfem::dimension::type::dim2>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  impl_compute_couple_stress(dimension_dispatch(), medium_dispatch(),
                             property_dispatch(), point_partial_derivatives,
                             point_properties, factor, F, acceleration);
}

} // namespace medium
} // namespace specfem
