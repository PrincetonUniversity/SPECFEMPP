#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/cosserat_couple_stress.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point.hpp"
#include "utilities/errors.hpp"
#include <Kokkos_Core.hpp>

template <typename T, typename PointJacobianMatrixType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, true>) {

  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;

  static_assert(
      specfem::data_access::is_point<PointJacobianMatrixType>::value &&
          specfem::data_access::is_jacobian_matrix<
              PointJacobianMatrixType>::value,
      "point_jacobian_matrix is not a point Jacobian matrix type");

  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "point_properties is not a point properties type");

  static_assert(
      specfem::data_access::is_point<PointAccelerationType>::value &&
          specfem::data_access::is_field<PointAccelerationType>::value,
      "point_acceleration is not a point field type");

  static_assert(
      PointJacobianMatrixType::simd::using_simd ==
          PointAccelerationType::simd::using_simd,
      "point_properties and acceleration have different SIMD settings");

  static_assert(PointPropertiesType::dimension_tag ==
                    PointJacobianMatrixType::dimension_tag,
                "point_properties and point_jacobian_matrix have different "
                "dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointAccelerationType::medium_tag,
                "point_properties and acceleration have different medium tags");

  static_assert(PointPropertiesType::simd::using_simd ==
                    PointJacobianMatrixType::simd::using_simd,
                "point_properties and point_jacobian_matrix have different "
                "SIMD settings");

  // Check the PointStressIntegrandViewType, which is a kokkos view for its
  //  extent
  static_assert(PointStressIntegrandViewType::rank() == 2,
                "PointStressIntegrandViewType must be a 2D view");
  static_assert(
      PointStressIntegrandViewType::static_extent(0) ==
          specfem::element::attributes<DimensionTag, MediumTag>::components,
      "PointStressIntegrandViewType must have the same number of "
      "components as the medium");
  static_assert(
      PointStressIntegrandViewType::static_extent(1) ==
          specfem::element::attributes<DimensionTag, MediumTag>::dimension,
      "PointStressIntegrandViewType must have the same number of "
      "dimensions as the medium");

  return;
}

template <typename T, typename PointJacobianMatrixType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
assert_types(const std::integral_constant<bool, false>) {
  // If the implementation is not available, we do nothing
  return;
}

namespace specfem {
namespace medium {

template <typename T, typename PointJacobianMatrixType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::false_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointJacobianMatrixType &point_jacobian_matrix,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {};

template <typename T, typename PointJacobianMatrixType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType, typename DimensionTagType,
          typename MediumTagType, typename PropertyTagType>
KOKKOS_INLINE_FUNCTION void impl_compute_cosserat_couple_stress(
    const std::true_type, const DimensionTagType dimension_tag,
    const MediumTagType medium_tag, const PropertyTagType property_tag,
    const PointJacobianMatrixType &point_jacobian_matrix,
    const PointPropertiesType &point_properties, const T factor,
    const PointStressIntegrandViewType &F,
    PointAccelerationType &acceleration) {

  // Extract actual tag types for the static_assert message
  using ActualDimensionTag = typename DimensionTagType::type;
  using ActualMediumTag = typename MediumTagType::type;
  using ActualPropertyTag = typename PropertyTagType::type;

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
 * @param PointJacobianMatrixType Jacobian matrix at the quadrature point
 * @param point_properties Material properties at the quadrature point
 * @param factor Prefactor for the damping term ($wx * wz * jacobian)
 * @param F Stress integrand at the quadrature point
 * @param acceleration Acceleration at the quadrature point
 */
template <typename T, typename PointJacobianMatrixType,
          typename PointPropertiesType, typename PointStressIntegrandViewType,
          typename PointAccelerationType>
KOKKOS_INLINE_FUNCTION void
compute_couple_stress(const PointJacobianMatrixType &point_jacobian_matrix,
                      const PointPropertiesType &point_properties,
                      const T factor, const PointStressIntegrandViewType &F,
                      PointAccelerationType &acceleration) {

  constexpr auto DimensionTag = PointPropertiesType::dimension_tag;
  constexpr auto MediumTag = PointPropertiesType::medium_tag;
  constexpr auto PropertyTag = PointPropertiesType::property_tag;
  constexpr bool has_cosserat_couple_stress =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::has_cosserat_couple_stress;

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, DimensionTag>;
  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, MediumTag>;
  using property_dispatch =
      std::integral_constant<specfem::element::property_tag, PropertyTag>;

  using cosserat_couple_stress_dispatch =
      std::integral_constant<bool, has_cosserat_couple_stress>;

  // Check that the types are compatible
  assert_types<T, PointJacobianMatrixType, PointPropertiesType,
               PointStressIntegrandViewType, PointAccelerationType>(
      cosserat_couple_stress_dispatch());

  impl_compute_cosserat_couple_stress(
      cosserat_couple_stress_dispatch(), dimension_dispatch(),
      medium_dispatch(), property_dispatch(), point_jacobian_matrix,
      point_properties, factor, F, acceleration);
}

} // namespace medium
} // namespace specfem
