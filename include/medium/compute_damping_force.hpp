#pragma once

#include "enumerations/medium.hpp"
#include "medium/dim2/poroelastic/isotropic/damping.hpp"
#include "utilities/errors.hpp"
#include <Kokkos_Core.hpp>

// Function that is called when the implementation is available
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void assert_types(const std::true_type) {

  constexpr auto DimensionTag = PointPropertiesType::dimension;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  // Check that the types are compatible
  static_assert(std::is_same_v<T, typename PointPropertiesType::simd::datatype>,
                "factor must have the same SIMD type as point_properties");

  static_assert(PointPropertiesType::is_point_properties,
                "point_properties is not a point properties type");

  static_assert(PointVelocityType::isPointFieldType,
                "velocity is not a point field type");

  static_assert(PointAccelerationType::isPointFieldType,
                "acceleration is not a point field type");

  static_assert(PointVelocityType::store_velocity,
                "velocity must store velocity");

  static_assert(PointAccelerationType::store_acceleration,
                "acceleration must store acceleration");

  static_assert(PointPropertiesType::dimension == PointVelocityType::dimension,
                "point_properties and velocity have different dimensions");

  static_assert(PointPropertiesType::dimension ==
                    PointAccelerationType::dimension,
                "point_properties and acceleration have different dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointVelocityType::medium_tag,
                "point_properties and velocity have different medium tags");

  static_assert(PointPropertiesType::medium_tag ==
                    PointAccelerationType::medium_tag,
                "point_properties and acceleration have different medium tags");

  static_assert(PointPropertiesType::simd::using_simd ==
                    PointVelocityType::simd::using_simd,
                "point_properties and velocity have different SIMD settings");

  static_assert(
      PointPropertiesType::simd::using_simd ==
          PointAccelerationType::simd::using_simd,
      "point_properties and acceleration have different SIMD settings");
}

// Function that is called when the implementation is not available
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void assert_types(const std::false_type) {
  // If the implementation is not available, we do nothing
  return;
}

namespace specfem {
namespace medium {

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_damping_force(
    std::false_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {
  // If the implementation is not available, we do nothing
  return;
}

template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_damping_force(
    std::true_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  // Extract actual tag types for the static_assert message
  using ActualDimensionTag = typename DimensionTagType::type;
  using ActualMediumTag = typename MediumTagType::type;
  using ActualPropertyTag = typename PropertyTagType::type;

  // The enumeration is set to true for damping force, but there is
  // no implementation available for this dimension, medium and property
  static_assert(
      specfem::utilities::always_false<ActualDimensionTag::value,
                                       ActualMediumTag::value,
                                       ActualPropertyTag::value>,
      "Damping force is not implemented for this medium and dimension.\n"
      "    --> Either deactivate damping force in "
      " enumerations/medium.hpp or \n"
      "        implement the damping force in "
      "medium/<dim>/<medium>/<property>/damping.hpp");
  //  If the implementation is not available, we do nothing
  return;
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
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void compute_damping_force(
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  constexpr auto DimensionTag = PointPropertiesType::dimension;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr bool has_damping_force =
      specfem::element::attributes<DimensionTag, MediumTag>::has_damping_force;

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, DimensionTag>;

  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  using damping_force_dispatch =
      std::integral_constant<bool, has_damping_force>;

  // Check that the types are compatible
  assert_types<T, PointPropertiesType, PointVelocityType,
               PointAccelerationType>(damping_force_dispatch());

  // If damping force is not available call empty function, else call the
  // implementation
  // Compute the damping force
  specfem::medium::impl_compute_damping_force(
      damping_force_dispatch(), dimension_dispatch(), medium_dispatch(),
      property_dispatch(), factor, point_properties, velocity, acceleration);
}

} // namespace medium
} // namespace specfem
