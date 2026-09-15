#pragma once

#include "specfem/data_access.hpp"
#include "specfem/element.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/cosserat_couple_stress.hpp"
#include "specfem/medium/dim3/elastic/isotropic_cosserat/cosserat_couple_stress.hpp"
#include "specfem/point.hpp"
#include "specfem/utilities.hpp"
#include <Kokkos_Core.hpp>

template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, true>) {

  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "point_properties is not a point properties type");

  static_assert(specfem::data_access::is_point<PointStressType>::value &&
                    specfem::data_access::is_stress<PointStressType>::value,
                "point_stress is not a point stress type");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_field<PointAccelerationType>::value,
      "point_acceleration is not a point field type");

  static_assert(PointPropertiesType::dimension_tag ==
                    PointStressType::dimension_tag,
                "point_properties and point_stress have different dimensions");

  static_assert(PointPropertiesType::medium_tag == PointStressType::medium_tag,
                "point_properties and point_stress have different medium tags");

  static_assert(PointPropertiesType::medium_tag ==
                    PointAccelerationType::medium_tag,
                "point_properties and acceleration have different medium tags");

  static_assert(PointStressType::simd::using_simd ==
                    PointAccelerationType::simd::using_simd,
                "point_stress and acceleration have different SIMD settings");

  static_assert(
      PointPropertiesType::simd::using_simd ==
          PointAccelerationType::simd::using_simd,
      "point_properties and acceleration have different SIMD settings");

  return;
}

template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, false>) {
  // If the implementation is not available, we do nothing
  return;
}

namespace specfem {
namespace medium_physics {

template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::false_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressType &point_stress,
    PointAccelerationType &acceleration) {};

template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressType &point_stress, PointAccelerationType &acceleration) {

  using ActualDimensionTag = typename DimensionTagType::type;
  using ActualMediumTag = typename MediumTagType::type;
  using ActualPropertyTag = typename PropertyTagType::type;

  static_assert(specfem::utilities::always_false<ActualDimensionTag::value,
                                                 ActualMediumTag::value,
                                                 ActualPropertyTag::value>,
                "\n\nCosserat Couple Stress is not implemented for "
                "this dimension, medium, and property.\n"
                "    --> Either deactivate has_cosserate_couple force in "
                " specfem/enums.hpp or \n"
                "        implement the damping force in "
                "medium/<dim>/<medium>/<property>/damping.hpp\n");
}

/**
 * @brief Compute Cosserat couple stress contribution for micropolar elastic
 * media.
 *
 * Provides compile-time dispatch to medium-specific implementations. Adds the
 * moment equilibrium contribution to @p acceleration based on stress tensor
 * asymmetry. Only modifies the spin component for Cosserat media; for all
 * other media this is a no-op.
 *
 * @note The caller is responsible for zero-initialising @p acceleration before
 *       calling this function. The contribution is accumulated with @c +=,
 *       consistent with the scatter pattern where the epilogue applies the
 *       global sign negation.
 *
 * @tparam T             Scalar type for the weight product
 * @tparam PointPropertiesType    Point-wise material properties
 * @tparam PointStressType        Point-wise physical stress tensor
 * @tparam PointAccelerationType  Point-wise acceleration field
 * @param point_properties  Cosserat material properties
 * @param factor             Integration factor w(iz)*w(ix)*J for this GLL point
 * @param point_stress      Physical stress tensor at this GLL point
 * @param acceleration[in,out]  Acceleration delta (spin component modified)
 */
template <typename T, typename PointPropertiesType, typename PointStressType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void compute_cosserat_couple_stress(
    const PointPropertiesType &point_properties, const T factor,
    const PointStressType &point_stress, PointAccelerationType &acceleration) {

  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr bool has_cosserat_couple_stress =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::has_cosserat_couple_stress;

  using dimension_dispatch =
      std::integral_constant<specfem::element::dimension_tag, DimensionTag>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  using cosserat_couple_stress_dispatch =
      std::integral_constant<bool, has_cosserat_couple_stress>;

  assert_types<T, PointPropertiesType, PointStressType, PointAccelerationType>(
      cosserat_couple_stress_dispatch());

  impl_compute_cosserat_couple_stress(cosserat_couple_stress_dispatch(),
                                      dimension_dispatch(), medium_dispatch(),
                                      property_dispatch(), point_properties,
                                      factor, point_stress, acceleration);
}

} // namespace medium_physics
} // namespace specfem
