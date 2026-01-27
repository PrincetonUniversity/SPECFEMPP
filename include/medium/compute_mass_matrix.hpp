#pragma once

#include "dim2/acoustic/isotropic/mass_matrix.tpp"
#include "dim2/elastic/anisotropic/mass_matrix.tpp"
#include "dim2/elastic/isotropic/mass_matrix.tpp"
#include "dim2/elastic/isotropic_cosserat/mass_matrix.tpp"
#include "dim2/poroelastic/isotropic/mass_matrix.tpp"
#include "dim3/acoustic/isotropic/mass_matrix.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {
// clang-format off
/**
 * @brief Compute mass matrix from material properties.
 *
 * Generic mass matrix computation interface that dispatches to medium-specific
 * implementations.
 *
 * @tparam DimensionTag Spatial dimension (dim2/dim3)
 * @tparam MediumTag Medium type (acoustic, elastic, poroelastic)
 * @tparam PropertyTag Property type (isotropic, anisotropic, etc.)
 * @tparam UseSIMD Enable SIMD vectorization
 * @param properties Material properties at quadrature point
 * @return Inverse mass matrix components for time integration
 *
 * @code{.cpp}
 * // Example usage for 2D elastic isotropic medium
 * using Properties = specfem::point::properties<dim2, elastic, isotropic, false>;
 * Properties props = ...; // Initialize material properties
 * auto mass_inv = specfem::medium::mass_matrix_component(props);
 * @endcode
 */
// clang-format on
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
