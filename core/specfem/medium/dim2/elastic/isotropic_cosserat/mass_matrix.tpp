#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION
    specfem::point::mass_inverse<specfem::element::dimension_tag::dim2,
                                 specfem::element::medium_tag::elastic_psv_t,
                                 UseSIMD>
    specfem::medium_physics::impl_mass_matrix_component(
        const specfem::point::properties<
            specfem::element::dimension_tag::dim2,
            specfem::element::medium_tag::elastic_psv_t,
            specfem::element::property_tag::isotropic_cosserat, UseSIMD>
            &properties) {

  return { properties.rho(), properties.rho(), properties.j() };
}
