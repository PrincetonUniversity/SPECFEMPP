#pragma once

#include "medium/dim2/elastic/isotropic_cosserat/couple_stress.hpp"
#include "utilities/errors.hpp"
#include <Kokkos_Core.hpp>

template <typename T, typename PointPartialDerivativesType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, true>) {

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

  return;
}

template <typename T, typename PointPartialDerivativesType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, false>) {
  // If the implementation is not available, we do nothing
  return;
}

namespace specfem {
namespace medium {

template <typename T, typename PointPartialDerivativesType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_couple_stress(
    const std::true_type, const DimensionTagType dimesion_dispatch,
    const MediumTagType medium_dispatch,
    const PropertyTagType property_dispatch,
    const PointPartialDerivativesType &point_partial_derivatives,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  using ActualDimensionTag = typename dimesion_dispatch::type;
  using ActualMediumTag = typename medium_dispatch::type;
  using ActualPropertyTag = typename property_dispatch::type;

  static_assert(specfem::utilities::always_false<ActualDimensionTag::value,
                                                 ActualMediumTag::value,
                                                 ActualPropertyTag::value>,
                "\n\nCosserat Couple Stress is not implemented for "
                "this dimension, medium, and property.\n"
                "    --> Either deactivate has_cosserate_couple force in "
                " enumerations/medium.hpp or \n"
                "        implement the damping force in "
                "medium/<dim>/<medium>/<property>/damping.hpp\n");
}

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

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type,
                             specfem::dimension::type::dim2>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  using couple_stress_dispatch =
      std::integral_constant<bool, has_cosserat_couple>;

  impl_compute_couple_stress(couple_stress_dispatch(), dimension_dispatch(),
                             medium_dispatch(), property_dispatch(),
                             point_partial_derivatives, point_properties,
                             factor, F, acceleration);
}

} // namespace medium
} // namespace specfem
