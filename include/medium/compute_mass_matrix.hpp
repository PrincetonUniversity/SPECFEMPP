#pragma once

#include "dim2/acoustic/isotropic/mass_matrix.tpp"
#include "dim2/elastic/anisotropic/mass_matrix.tpp"
#include "dim2/elastic/isotropic/mass_matrix.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/partial_derivatives.hpp"
#include "point/properties.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

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

} // namespace medium
} // namespace specfem
