#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"

namespace specfem::medium_physics {

/**
 * @brief Compute mass-matrix components for 3D anisotropic elastic media.
 *
 * @tparam Tags Point tags selecting 3D elastic anisotropic properties.
 * @param properties Material properties at the quadrature point.
 * @return The three diagonal mass components \f$[\rho,\rho,\rho]\f$.
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::anisotropic,
        int> = 0>
KOKKOS_FUNCTION specfem::point::mass_inverse<Tags>
impl_mass_matrix_component(const specfem::point::properties<Tags> &properties) {
  return { properties.rho(), properties.rho(), properties.rho() };
}

} // namespace specfem::medium_physics
