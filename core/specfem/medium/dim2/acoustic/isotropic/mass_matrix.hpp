#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_dim2_compute_mass_matrix_acoustic
 *
 */

/**
 * @ingroup specfem_medium_dim2_compute_mass_matrix_acoustic
 * @brief Compute mass matrix inverse for 2D acoustic isotropic media.
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
template <
    typename Tags,
    std::enable_if_t<
        Tags::medium_tag == specfem::element::medium_tag::acoustic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_FUNCTION specfem::point::mass_inverse<Tags>
impl_mass_matrix_component(const specfem::point::properties<Tags> &properties) {

  return { static_cast<type_real>(1.0) / properties.kappa() };
}

} // namespace medium_physics
} // namespace specfem
