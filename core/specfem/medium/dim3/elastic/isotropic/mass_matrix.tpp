#pragma once

#include "mass_matrix.hpp"

template <specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::tags::Tags<specfem::element::dimension_tag::dim3, specfem::element::medium_tag::elastic, UseSIMD>>
specfem::medium_physics::impl_mass_matrix_component(
    const specfem::point::properties<specfem::element::dimension_tag::dim3,
                                     specfem::element::medium_tag::elastic,
                                     PropertyTag, UseSIMD> &properties) {
  return { properties.rho(), properties.rho(), properties.rho() };
}
