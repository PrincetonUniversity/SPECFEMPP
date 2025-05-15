#pragma once

#include "enumerations/medium.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/cosserat_stress.hpp"
#include "utilities/errors.hpp"
#include <Kokkos_Core.hpp>

// Function that is called when the implementation is available
template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void assert_types(const std::true_type) {

  constexpr auto DimensionTag = PointPropertiesType::dimension;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  static_assert(PointPropertiesType::is_point_properties,
                "point_properties is not a point properties type");

  static_assert(PointDisplacementType::isPointFieldType,
                "displacement is not a point field type");

  static_assert(PointDisplacementType::store_displacement,
                "displacement must store displacement");

  static_assert(PointPropertiesType::dimension ==
                    PointDisplacementType::dimension,
                "point_properties and velocity have different dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointDisplacementType::medium_tag,
                "point_properties and velocity have different medium tags");

  static_assert(PointPropertiesType::simd::using_simd ==
                    PointDisplacementType::simd::using_simd,
                "point_properties and velocity have different SIMD settings");

  static_assert(PointStressType::dimension == PointDisplacementType::dimension,
                "point_stress and displacement have different dimensions");

  static_assert(PointStressType::medium_tag ==
                    PointDisplacementType::medium_tag,
                "point_stress and displacement have different medium tags");

  static_assert(
      PointStressType::simd::using_simd ==
          PointDisplacementType::simd::using_simd,
      "point_properties and displacement have different SIMD settings");
}

// Function that is called when the implementation is not available
template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void assert_types(const std::false_type) {
  // If the implementation is not available, we do nothing
  return;
}

namespace specfem {
namespace medium {

template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_stress(
    std::false_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointPropertiesType &point_properties,
    const PointDisplacementType &point_displacement,
    PointStressType &point_stress) {
  // If the implementation is not available, we do nothing
  return;
}

template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_stress(
    std::true_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointPropertiesType &point_properties,
    const PointDisplacementType &point_displacement,
    PointStressType &point_stress) {

  // Extract actual tag types for the static_assert message
  using ActualDimensionTag = typename DimensionTagType::type;
  using ActualMediumTag = typename MediumTagType::type;
  using ActualPropertyTag = typename PropertyTagType::type;

  // The enumeration is set to true for damping force, but there is
  // no implementation available for this dimension, medium and property
  static_assert(specfem::utilities::always_false<ActualDimensionTag::value,
                                                 ActualMediumTag::value,
                                                 ActualPropertyTag::value>,
                "\n\nCosserat Stress Contribution is not implemented for "
                "this dimension, medium, and property.\n"
                "    --> Either deactivate damping force in "
                " enumerations/medium.hpp or \n"
                "        implement the cosserat stress computation in "
                "medium/<dim>/<medium>/<property>/cosserat_stress.hpp\n");
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
template <typename PointPropertiesType, typename PointDisplacementType,
          typename PointStressType>
KOKKOS_INLINE_FUNCTION void
compute_cosserat_stress(const PointPropertiesType &point_properties,
                        const PointDisplacementType &point_displacement,
                        PointStressType &point_stress) {

  constexpr auto DimensionTag = PointPropertiesType::dimension;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr bool has_cosserat_stress =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::has_cosserat_stress;

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, DimensionTag>;

  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  using cosserat_stress_dispatch =
      std::integral_constant<bool, has_cosserat_stress>;

  // Check that the types are compatible
  assert_types<PointPropertiesType, PointDisplacementType, PointStressType>(
      cosserat_stress_dispatch());

  // If damping force is not available call empty function, else call the
  // implementation
  // Compute the damping force
  specfem::medium::impl_compute_cosserat_stress(
      cosserat_stress_dispatch(), dimension_dispatch(), medium_dispatch(),
      property_dispatch(), point_properties, point_displacement, point_stress);
}

} // namespace medium
} // namespace specfem
