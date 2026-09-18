#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_dim3_compute_mass_matrix_elastic_isotropic_cosserat
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_mass_matrix_elastic_isotropic_cosserat
 * @brief Compute mass matrix inverse for 3D elastic isotropic Cosserat media.
 *
 * Implements mass matrix for Cosserat (micropolar) elastic media with
 * rotational degrees of freedom. Extends classical elasticity by including
 * rotational inertia for microstructural effects.
 *
 * **Mass matrix components:**
 * - Translation: \f$ M_{trans} = \rho \f$ (displacement DOF)
 * - Rotation: \f$ M_{rot} = j \f$ (rotational DOF)
 *
 * **Physical parameters:**
 * - \f$ \rho \f$: Mass density
 * - \f$ j \f$: Rotational inertia (microinertia)
 *
 * Returns 6 components for 3D Cosserat system: [ρ, ρ, ρ, j, j, j]
 * corresponding to [u_x, u_y, u_z, ω_x, ω_y, ω_z] degrees of freedom where ω_y
 * is the rotation about the y-axis (out-of-plane).
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param properties Cosserat material properties
 * @return Inverse mass matrix components for explicit time integration
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic_spin &&
            Tags::property_tag ==
                specfem::element::property_tag::isotropic_cosserat,
        int> = 0>
KOKKOS_FUNCTION specfem::point::mass_inverse<Tags>
impl_mass_matrix_component(const specfem::point::properties<Tags> &properties) {

  return { properties.rho(), properties.rho(), properties.rho(),
           properties.j(),   properties.j(),   properties.j() };
}

} // namespace medium_physics
} // namespace specfem
