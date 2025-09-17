#pragma once

#include "dim2/acoustic/isotropic/stress.hpp"
#include "dim2/elastic/anisotropic/stress.hpp"
#include "dim2/elastic/isotropic/stress.hpp"
#include "dim2/elastic/isotropic_cosserat/stress.hpp"
#include "dim2/poroelastic/isotropic/stress.hpp"
#include "dim3/elastic/isotropic/stress.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @defgroup MediumPhysics
 */

/**
 * @brief Compute the stress tensor at a quadrature point
 *
 * @ingroup MediumPhysics
 *
 * @tparam PointPropertiesType Material properties at the quadrature point
 * specfem::point::properties
 * @tparam PointFieldDerivativesType Field derivatives at the quadrature point
 * specfem::point::field_derivatives
 * @param properties Material properties at the quadrature point
 * @param field_derivatives Field derivatives at the quadrature point
 * @return specfem::point::stress The stress tensor at the quadrature point
 */
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
