#pragma once

#include "specfem/element.hpp"
#include "specfem/point.hpp"

namespace specfem::medium_physics {

/**
 * @brief Reject stress evaluation until 3D anisotropic physics is implemented.
 *
 * @tparam Tags Point tags selecting 3D elastic anisotropic properties.
 * @param properties Material properties at the quadrature point.
 * @param field_derivatives Displacement derivatives at the quadrature point.
 * @return No value; instantiation fails with a diagnostic.
 */
template <
    typename Tags,
    std::enable_if_t<
        Tags::dimension_tag == specfem::element::dimension_tag::dim3 &&
            Tags::medium_tag == specfem::element::medium_tag::elastic &&
            Tags::property_tag == specfem::element::property_tag::anisotropic,
        int> = 0>
KOKKOS_INLINE_FUNCTION specfem::point::stress<Tags> compute_stress(
    const specfem::point::properties<Tags> &properties,
    const specfem::point::field_derivatives<Tags> &field_derivatives) {
  static_cast<void>(properties);
  static_cast<void>(field_derivatives);
  static_assert(
      Tags::property_tag != specfem::element::property_tag::anisotropic,
      "3D elastic anisotropic stress is not implemented; see the follow-up "
      "3D anisotropic medium-physics issue");
}

} // namespace specfem::medium_physics
