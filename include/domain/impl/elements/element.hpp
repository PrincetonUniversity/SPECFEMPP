#pragma once

#include "acoustic/acoustic2d.hpp"
#include "elastic/elastic2d.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress_integrand.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace elements {

/**
 * @brief Compute stress integrands at a given quadrature point given the
 * derivatives of the field at that point
 *
 * @tparam DimensionType Dimension of the element (2D or 3D)
 * @tparam MediumTag Medium tag for the element
 * @tparam PropertyTag Property tag for the element
 * @tparam UseSIMD Use SIMD instructions
 * @param partial_derivatives Spatial derivatives of basis functions at the
 * quadrature point
 * @param properties Material properties at the quadrature point
 * @param field_derivatives Derivatives of the field at the quadrature point
 * @return specfem::point::stress_integrand<DimensionType, MediumTag, UseSIMD>
 * Stress integrands at the quadrature point
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress_integrand<DimensionType, MediumTag, UseSIMD>
    compute_stress_integrands(
        const specfem::point::partial_derivatives<DimensionType, false, UseSIMD>
            &partial_derivatives,
        const specfem::point::properties<DimensionType, MediumTag, PropertyTag,
                                         UseSIMD> &properties,
        const specfem::point::field_derivatives<DimensionType, MediumTag,
                                                UseSIMD> &field_derivatives) {
  return impl_compute_stress_integrands(partial_derivatives, properties,
                                        field_derivatives);
}

/**
 * @brief Compute the contribution to mass matrix at a given quadrature point
 * within an element
 *
 * @tparam DimensionType Dimension of the element (2D or 3D)
 * @tparam MediumTag Medium tag for the element
 * @tparam PropertyTag Property tag for the element
 * @tparam UseSIMD Use SIMD instructions
 * @param properties Material properties at the quadrature point
 * @param partial_derivatives Spatial derivatives of basis functions at the
 * quadrature point
 * @return specfem::point::field<DimensionType, MediumTag, false, false, false,
 * true, UseSIMD> Contribution to mass matrix at the quadrature point
 */
template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::field<DimensionType, MediumTag, false,
                                             false, false, true, UseSIMD>
mass_matrix_component(
    const specfem::point::properties<DimensionType, MediumTag, PropertyTag,
                                     UseSIMD> &properties,
    const specfem::point::partial_derivatives<DimensionType, true, UseSIMD>
        &partial_derivatives) {
  return impl_mass_matrix_component(properties, partial_derivatives);
}

} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem
