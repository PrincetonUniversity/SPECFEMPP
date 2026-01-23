#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium {

/**
 * @defgroup specfem_medium_dim3_compute_mass_matrix_acoustic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_mass_matrix_acoustic
 * @brief Compute mass matrix inverse for 3D acoustic isotropic media.
 *
 * Implements mass matrix for acoustic wave propagation in fluids.
 * Acoustic media only support compressional waves (P-waves) with no
 * shear wave propagation, leading to a simplified mass matrix formulation.
 *
 * **Mass matrix:**
 * \f$ M = \frac{1}{\kappa} \f$
 *
 * where \f$ \kappa \f$ is the bulk modulus.
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @param properties Acoustic material properties (\f$ \rho^{-1}, \kappa \f$)
 * @return Mass inverse component [\f$ \kappa^{-1} \f$] for pressure wavefield
 */
template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION
    specfem::point::mass_inverse<specfem::dimension::type::dim3,
                                 specfem::element::medium_tag::acoustic,
                                 UseSIMD>
    impl_mass_matrix_component(
        const specfem::point::properties<specfem::dimension::type::dim3,
                                         specfem::element::medium_tag::acoustic,
                                         PropertyTag, UseSIMD> &properties);

} // namespace specfem::medium
