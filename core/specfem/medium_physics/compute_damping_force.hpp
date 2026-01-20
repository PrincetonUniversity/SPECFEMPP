#pragma once

#include "enumerations/medium.hpp"
#include "medium/dim2/poroelastic/isotropic/damping.hpp"
#include "specfem/data_access.hpp"
#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>

// Function that is called when the implementation is available
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void assert_types(const std::true_type) {

  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "point_properties is not a point properties type");

  // Check that the types are compatible
  static_assert(std::is_same_v<T, typename PointPropertiesType::simd::datatype>,
                "factor must have the same SIMD type as point_properties");

  static_assert(specfem::data_access::is_point<PointVelocityType>::value &&
                    specfem::data_access::is_field<PointVelocityType>::value,
                "velocity is not a point field type");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_field<PointAccelerationType>::value,
      "acceleration is not a point field type");

  static_assert(PointPropertiesType::dimension_tag ==
                    PointVelocityType::dimension_tag,
                "point_properties and velocity have different dimensions");

  static_assert(PointPropertiesType::dimension_tag ==
                    PointAccelerationType::dimension_tag,
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
  static_assert(specfem::utilities::always_false<ActualDimensionTag::value,
                                                 ActualMediumTag::value,
                                                 ActualPropertyTag::value>,
                "\n\nDamping force is not implemented for this dimension, "
                "medium, and property.\n"
                "    --> Either deactivate damping force in "
                " enumerations/medium.hpp or \n"
                "        implement the damping force in "
                "medium/<dim>/<medium>/<property>/damping.hpp\n");
  //  If the implementation is not available, we do nothing
  return;
}

/**
 * @brief Compute damping force for wave attenuation.
 *
 * Generic damping force computation interface that adds viscous damping
 * to wave propagation equations. Provides compile-time dispatch to
 * medium-specific implementations based on element attributes.
 *
 * @note Only medium types with damping force support will modify the
 * acceleration. Medium types without damping force support will result a no-op.
 *
 * @tparam T Scalar type for damping factor
 * @tparam PointPropertiesType Point-wise material properties
 * @tparam PointVelocityType Point-wise velocity field
 * @tparam PointAccelerationType Point-wise acceleration field
 * @param factor Integration factor (e.g., product of quadrature weight(s) and
 * Jacobian determinant) \f$ J \, w_q \f$
 * @param point_properties Material properties at point
 * @param velocity Velocity field at point
 * @param acceleration[in,out] Acceleration field (modified by damping)
 */
template <typename T, typename PointPropertiesType, typename PointVelocityType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void compute_damping_force(
    const T factor, const PointPropertiesType &point_properties,
    const PointVelocityType &velocity, PointAccelerationType &acceleration) {

  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;
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
