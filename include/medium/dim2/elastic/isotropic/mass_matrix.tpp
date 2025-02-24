#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic_sv,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
        PropertyTag, UseSIMD> &properties,
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives) {

  if constexpr (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>(
        partial_derivatives.jacobian * properties.rho,
        partial_derivatives.jacobian * properties.rho);
  } else if constexpr (specfem::globals::simulation_wave == specfem::wave::sh) {
    return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>(
        partial_derivatives.jacobian * properties.rho, 0);
  } else {
    static_assert("Unknown wave type");
    return specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>(
        0.0, 0.0); // dummy return
  }
}
