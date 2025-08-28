#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
    UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim3,
                                     specfem::element::medium_tag::elastic,
                                     PropertyTag, UseSIMD> &properties) {

  return { properties.rho(), properties.rho(), properties.rho() };
}
