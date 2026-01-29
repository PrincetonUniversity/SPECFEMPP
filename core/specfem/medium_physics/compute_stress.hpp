#pragma once

#include "medium/dim2/acoustic/isotropic/stress.hpp"
#include "medium/dim2/elastic/anisotropic/stress.hpp"
#include "medium/dim2/elastic/isotropic/stress.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/stress.hpp"
#include "medium/dim2/poroelastic/isotropic/stress.hpp"
#include "medium/dim3/acoustic/isotropic/stress.hpp"
#include "medium/dim3/elastic/isotropic/stress.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

// clang-format off
/**
 * @brief Compute stress tensor from material properties and field derivatives.
 *
 * Generic stress computation interface that dispatches to medium-specific
 * implementations based on dimension, medium type, and property tags.
 * Provides compile-time type safety through static assertions.
 *
 * @tparam PointPropertiesType Point-wise material properties container
 * @tparam PointFieldDerivativesType Point-wise displacement derivatives container
 * @param properties Material properties at quadrature point
 * @param field_derivatives Displacement field derivatives at point
 * @return Stress tensor computed using medium-specific constitutive relations
 *
 * @code{.cpp}
 * // Example usage for 2D elastic isotropic medium
 * using Properties = specfem::point::properties<dim2, elastic, isotropic, false>;
 * using FieldDerivatives = specfem::point::field_derivatives<dim2, elastic, false>;
 * Properties props = ...; // Initialize material properties
 * FieldDerivatives derivs = ...; // Initialize field derivatives
 * auto stress = specfem::medium::compute_stress(props, derivs);
 * @endcode
 */
// clang-format on
template <typename PointPropertiesType, typename PointFieldDerivativesType>
KOKKOS_INLINE_FUNCTION auto
compute_stress(const PointPropertiesType &properties,
               const PointFieldDerivativesType &field_derivatives)
    -> decltype(specfem::medium::impl_compute_stress(properties,
                                                     field_derivatives)) {

  // Check whether the point is of properties type
  static_assert(
      specfem::data_access::is_point<PointPropertiesType>::value &&
          specfem::data_access::is_properties<PointPropertiesType>::value,
      "properties is not a point properties type");

  static_assert(
      specfem::data_access::is_point<PointFieldDerivativesType>::value &&
          +specfem::data_access::is_field_derivatives<
              PointFieldDerivativesType>::value,
      "field_derivatives is not a point field derivatives type");

  static_assert(PointPropertiesType::dimension_tag ==
                    PointFieldDerivativesType::dimension_tag,
                "properties and field_derivatives have different dimensions");

  static_assert(PointPropertiesType::medium_tag ==
                    PointFieldDerivativesType::medium_tag,
                "properties and field_derivatives have different medium tags");

  static_assert(
      PointPropertiesType::simd::using_simd ==
          PointFieldDerivativesType::simd::using_simd,
      "properties and field_derivatives have different SIMD settings");

  return specfem::medium::impl_compute_stress(properties, field_derivatives);
}

} // namespace medium
} // namespace specfem
