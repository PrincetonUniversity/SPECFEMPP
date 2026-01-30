#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

/**
 * @defgroup specfem_medium_dim2_compute_mass_matrix_elastic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_mass_matrix_elastic
 * @brief Compute mass matrix inverse for 2D elastic isotropic P-SV waves.
 *
 * Implements mass matrix for pressure-shear vertical wave propagation.
 * P-SV waves involve displacement in the x-z plane with coupling between
 * normal and shear motions.
 *
 * **Mass matrix:**
 * \f$ M = \rho \f$ (for both u_x and u_z components)
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @tparam PropertyTag Property type (isotropic, anisotropic)
 * @param properties Material properties (density)
 * @return Mass inverse components [ρ, ρ] for [u_x, u_z]
 */
template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_psv,
                                     PropertyTag, UseSIMD> &properties);

/**
 * @ingroup specfem_medium_dim2_compute_mass_matrix_elastic
 * @brief Compute mass matrix inverse for 2D elastic isotropic SH waves.
 *
 * Implements mass matrix for shear horizontal wave propagation.
 * SH waves involve anti-plane motion (u_y displacement only) perpendicular
 * to the propagation plane.
 *
 * **Mass matrix:**
 * \f$ M = \rho \f$ (for u_y component only)
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @tparam PropertyTag Property type (isotropic, anisotropic)
 * @param properties Material properties (density)
 * @return Mass inverse component [ρ] for [u_y]
 */
template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>
impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     PropertyTag, UseSIMD> &properties);

} // namespace medium
} // namespace specfem
