#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim2_compute_mass_matrix_elastic_cosserat
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_mass_matrix_elastic_cosserat
 * @brief Compute mass matrix inverse for 2D elastic isotropic Cosserat media.
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
 * Returns 3 components for 2D Cosserat system: [ρ, ρ, j]
 * corresponding to [u_x, u_z, ω_y] degrees of freedom where ω_y
 * is the rotation about the y-axis (out-of-plane).
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param properties Cosserat material properties
 * @return Inverse mass matrix components for explicit time integration
 */
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv_t,
    UseSIMD>
impl_mass_matrix_component(const specfem::point::properties<
                           specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_psv_t,
                           specfem::element::property_tag::isotropic_cosserat,
                           UseSIMD> &properties);

} // namespace medium
} // namespace specfem
