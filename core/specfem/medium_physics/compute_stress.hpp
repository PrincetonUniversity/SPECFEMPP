#pragma once

#include "specfem/data_access.hpp"
#include "specfem/medium/dim2/acoustic/isotropic/stress.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/stress.hpp"
#include "specfem/medium/dim2/elastic/isotropic/stress.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/stress.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/stress.hpp"
#include "specfem/medium/dim3/acoustic/isotropic/stress.hpp"
#include "specfem/medium/dim3/elastic/isotropic/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium_physics {

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
 * using Properties = specfem::point::properties<specfem::tags::Tags<dim2, elastic, isotropic, false>>;
 * using FieldDerivatives = specfem::point::field_derivatives<specfem::tags::Tags<dim2, elastic, isotropic, false>>;
 * Properties props = ...; // Initialize material properties
 * FieldDerivatives derivs = ...; // Initialize field derivatives
 * auto stress = specfem::medium_physics::compute_stress<Tags>(props, derivs);
 * @endcode
 */
// clang-format on
template <typename Tags>
KOKKOS_INLINE_FUNCTION specfem::point::stress<Tags> compute_stress(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::field_derivatives<Tags> &field_derivatives) {

  return specfem::medium_physics::impl_compute_stress<Tags>(properties,
                                                            field_derivatives);
}

} // namespace medium_physics
} // namespace specfem
