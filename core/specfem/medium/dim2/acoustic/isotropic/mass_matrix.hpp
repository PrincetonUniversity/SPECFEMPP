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
template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<specfem::tags::Tags<
    DimensionTag, specfem::element::medium_tag::acoustic, UseSIMD> >
impl_mass_matrix_component(
    const specfem::point::properties<
        DimensionTag, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties);

} // namespace medium_physics
} // namespace specfem
