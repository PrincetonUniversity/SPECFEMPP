#pragma once

#include "dim2/acoustic/isotropic/frechet_derivative.hpp"
#include "dim2/elastic/anisotropic/frechet_derivative.hpp"
#include "dim2/elastic/isotropic/frechet_derivative.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointPropertiesType, typename AdjointPointFieldType,
          typename BackwardPointFieldType, typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto compute_frechet_derivatives(
    const PointPropertiesType &properties,
    const AdjointPointFieldType &adjoint_field,
    const BackwardPointFieldType &backward_field,
    const PointFieldDerivativesType &adjoint_derivatives,
    const PointFieldDerivativesType &backward_derivatives,
    const type_real &dt) {

  static_assert(PointPropertiesType::is_point_properties,
                "properties is not a point properties type");
  static_assert(PointFieldDerivativesType::is_point_field_derivatives,
                "field_derivatives is not a point field derivatives type");

  static_assert(AdjointPointFieldType::isPointFieldType,
                "adjoint_field is not a point field type");

  static_assert(BackwardPointFieldType::isPointFieldType,
                "backward_field is not a point field type");

  static_assert(AdjointPointFieldType::store_acceleration,
                "adjoint_field does not store acceleration");

  static_assert(BackwardPointFieldType::store_displacement,
                "backward_field does not store displacement");

  constexpr auto dimension = PointPropertiesType::dimension;

  static_assert(
      (dimension == AdjointPointFieldType::dimension &&
       dimension == BackwardPointFieldType::dimension &&
       dimension == PointFieldDerivativesType::dimension),
      "Dimension inconsistency between properties, fields, and derivatives");

  constexpr auto using_simd = PointPropertiesType::simd::using_simd;

  static_assert(
      (using_simd == AdjointPointFieldType::simd::using_simd &&
       using_simd == BackwardPointFieldType::simd::using_simd &&
       using_simd == PointFieldDerivativesType::simd::using_simd),
      "SIMD inconsistency between properties, fields, and derivatives");

  constexpr auto medium_tag = PointPropertiesType::medium_tag;

  static_assert(
      (medium_tag == AdjointPointFieldType::medium_tag &&
       medium_tag == BackwardPointFieldType::medium_tag &&
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
      adjoint_field, backward_field, adjoint_derivatives, backward_derivatives,
      dt);
}

} // namespace medium
} // namespace specfem
