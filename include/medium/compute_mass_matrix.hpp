#pragma once

#include "dim2/acoustic/isotropic/mass_matrix.tpp"
#include "dim2/elastic/anisotropic/mass_matrix.tpp"
#include "dim2/elastic/isotropic/mass_matrix.tpp"
#include "dim2/elastic/isotropic_cosserat/mass_matrix.tpp"
#include "dim2/poroelastic/isotropic/mass_matrix.tpp"
#include "dim3/elastic/isotropic/mass_matrix.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/material_definitions.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

/**
 * @brief Compute the contribution to mass matrix at a given quadrature point
 * within an element
 *
 * @tparam DimensionTag Dimension of the element (2D or 3D)
 * @tparam MediumTag Medium tag for the element
 * @tparam PropertyTag Property tag for the element
 * @tparam UseSIMD Use SIMD instructions
 * @param properties Material properties at the quadrature point
 * @param jacobian_matrix Spatial derivatives of basis functions at the
 * quadrature point
 * @return specfem::point::mass_matrix<DimensionTag, MediumTag,
 * UseSIMD> Contribution to mass matrix at the quadrature point
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::mass_inverse<DimensionTag, MediumTag, UseSIMD>
    mass_matrix_component(
        const specfem::point::properties<DimensionTag, MediumTag, PropertyTag,
                                         UseSIMD> &properties) {
  return impl_mass_matrix_component(properties);
}

} // namespace medium
} // namespace specfem
