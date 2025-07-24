#pragma once

#include "dim2/acoustic/isotropic/frechet_derivative.hpp"
#include "dim2/elastic/anisotropic/frechet_derivative.hpp"
#include "dim2/elastic/isotropic/frechet_derivative.hpp"
#include "dim2/poroelastic/isotropic/frechet_derivative.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointPropertiesType, typename AdjointPointVelocityType,
          typename AdjointPointAccelerationType,
          typename BackwardPointDisplacementType,
          typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto compute_frechet_derivatives(
    const PointPropertiesType &properties,
    const AdjointPointVelocityType &adjoint_velocity,
    const AdjointPointAccelerationType &adjoint_acceleration,
    const BackwardPointDisplacementType &backward_displacement,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "properties is not a point properties type");

  static_assert(
      specfem::data_access::is_point<PointFieldDerivativesType>::value &&
          specfem::data_access::is_field_derivatives<
              PointFieldDerivativesType>::value,
      "field_derivatives is not a point field derivatives type");

  static_assert(
      specfem::data_access::is_point<AdjointPointVelocityType>::value &&
          specfem::data_access::is_field_l<AdjointPointVelocityType>::value,
      "adjoint_velocity is not a point field type");

  static_assert(
      specfem::data_access::is_point<AdjointPointAccelerationType>::value &&
          specfem::data_access::is_field_l<AdjointPointAccelerationType>::value,
      "adjoint_acceleration is not a point field type");

  static_assert(
      specfem::data_access::is_point<BackwardPointDisplacementType>::value &&
          specfem::data_access::is_field_l<
              BackwardPointDisplacementType>::value,
      "backward_displacement is not a point field type");

  constexpr auto dimension = PointPropertiesType::dimension_tag;

  static_assert(
      (dimension == AdjointPointVelocityType::dimension_tag &&
       dimension == AdjointPointAccelerationType::dimension_tag &&
       dimension == BackwardPointDisplacementType::dimension_tag &&
       dimension == PointFieldDerivativesType::dimension_tag),
      "Dimension inconsistency between properties, fields, and derivatives");

  constexpr auto using_simd = PointPropertiesType::simd::using_simd;

  static_assert(
      (using_simd == AdjointPointVelocityType::simd::using_simd &&
       using_simd == AdjointPointAccelerationType::simd::using_simd &&
       using_simd == BackwardPointDisplacementType::simd::using_simd &&
       using_simd == PointFieldDerivativesType::simd::using_simd),
      "SIMD inconsistency between properties, fields, and derivatives");

  constexpr auto medium_tag = PointPropertiesType::medium_tag;

  static_assert(
      (medium_tag == AdjointPointVelocityType::medium_tag &&
       medium_tag == AdjointPointAccelerationType::medium_tag &&
       medium_tag == BackwardPointDisplacementType::medium_tag &&
       medium_tag == PointFieldDerivativesType::medium_tag),
      "Medium tag inconsistency between properties, fields, and derivatives");

  using dimension_dispatch =
      std::integral_constant<specfem::dimension::type, dimension>;

  using medium_dispatch =
      std::integral_constant<specfem::element::medium_tag, medium_tag>;

  using property_dispatch =
      std::integral_constant<specfem::element::property_tag,
                             PointPropertiesType::property_tag>;

  return specfem::medium::impl_compute_frechet_derivatives(
      dimension_dispatch(), medium_dispatch(), property_dispatch(), properties,
      adjoint_velocity, adjoint_acceleration, backward_displacement,
      adjoint_derivatives, backward_derivatives, dt);
}

} // namespace medium
} // namespace specfem
