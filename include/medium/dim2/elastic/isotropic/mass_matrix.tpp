#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic_psv,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_psv,
                                     PropertyTag, UseSIMD> &properties) {

  return specfem::datatype::VectorPointViewType<type_real, 2, UseSIMD>(
      properties.rho(), properties.rho());
}

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic_sh,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     PropertyTag, UseSIMD> &properties) {

  return specfem::datatype::VectorPointViewType<type_real, 1, UseSIMD>(
      properties.rho());
}
