#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic_psv_t,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv_t,
        specfem::element::property_tag::isotropic_cosserat, UseSIMD> &properties) {

  return specfem::datatype::VectorPointViewType<type_real, 3, UseSIMD>(properties.rho(), properties.rho(), properties.j());
}
