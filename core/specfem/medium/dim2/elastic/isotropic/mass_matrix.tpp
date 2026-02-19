#pragma once

#include "mass_matrix.hpp"

template <specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::tags::Tags<specfem::element::dimension_tag::dim2, specfem::element::medium_tag::elastic_psv,
    UseSIMD>>
specfem::medium_physics::impl_mass_matrix_component(
    const specfem::point::properties<specfem::tags::Tags<specfem::element::dimension_tag::dim2, specfem::element::medium_tag::elastic_psv, PropertyTag, UseSIMD>> &properties) {

  return { properties.rho(), properties.rho() };
}

template <specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::tags::Tags<specfem::element::dimension_tag::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>>
specfem::medium_physics::impl_mass_matrix_component(
    const specfem::point::properties<specfem::tags::Tags<specfem::element::dimension_tag::dim2, specfem::element::medium_tag::elastic_sh, PropertyTag, UseSIMD>> &properties) {

  return { properties.rho() }; ///< Mass matrix for SH waves is isotropic
}
