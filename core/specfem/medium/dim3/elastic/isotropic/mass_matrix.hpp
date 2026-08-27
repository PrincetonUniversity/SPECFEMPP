#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"

namespace specfem {
namespace medium_physics {

/**
 * @defgroup specfem_medium_dim3_compute_mass_matrix_elastic
 *
 */

/**
 * @ingroup specfem_medium_dim3_compute_mass_matrix_elastic
 * @brief Compute mass matrix inverse for 3D elastic isotropic media.
 *
 * **Mass matrix:**
 * \f$ M_{ij} = \rho \delta_{ij} \f$
 *
 * **Components:**
 * \f$ [M_x, M_y, M_z] = [\rho, \rho, \rho] \f$
 *
 * @tparam UseSIMD Enable SIMD vectorization
 * @tparam PropertyTag Property type (isotropic, anisotropic)
 * @param properties Material properties
 * @return Mass inverse components for \f$ [u_x, u_y, u_z] \f$
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::isotropic,
        int> = 0>
KOKKOS_FUNCTION specfem::point::mass_inverse<Tags>
impl_mass_matrix_component(const specfem::point::properties<Tags> &properties) {
  return { properties.rho(), properties.rho(), properties.rho() };
}
} // namespace medium_physics
} // namespace specfem
