#pragma once

#include "specfem/element.hpp"
#include "specfem/macros.hpp"
#include "specfem/medium/dim2/acoustic/isotropic/mass_matrix.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/mass_matrix.hpp"
#include "specfem/medium/dim2/elastic/isotropic/mass_matrix.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/mass_matrix.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/mass_matrix.hpp"
#include "specfem/medium/dim3/elastic/isotropic/mass_matrix.hpp"
#include "specfem/medium/dim3/elastic/isotropic_cosserat/mass_matrix.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {
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
 * using Properties = specfem::point::properties<specfem::tags::Tags<dim2, elastic, isotropic, false>>;
 * Properties props = ...; // Initialize material properties
 * auto mass_inv = specfem::medium_physics::mass_matrix_component<Tags>(props);
 * @endcode
 */
// clang-format on
template <typename Tags>
KOKKOS_INLINE_FUNCTION specfem::point::mass_inverse<Tags>
mass_matrix_component(const specfem::point::properties<Tags> &properties) {
  return impl_mass_matrix_component<Tags>(properties);
}

} // namespace medium_physics
} // namespace specfem
