#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
    UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_psv,
                                     PropertyTag, UseSIMD> &properties) {

  return { properties.rho(), properties.rho() };
}

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     PropertyTag, UseSIMD> &properties) {

  return { properties.rho() }; ///< Mass matrix for SH waves is isotropic
}
