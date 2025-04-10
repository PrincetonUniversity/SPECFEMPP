#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic_psv_t,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv_t,
        specfem::element::property_tag::isotropic_cosserat, UseSIMD> &properties,
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives) {

const auto elastic_component = partial_derivatives.jacobian * properties.rho();
const auto spin_component = partial_derivatives.jacobian * properties.j();

  return specfem::datatype::ScalarPointViewType<type_real, 3, UseSIMD>(elastic_component, elastic_component, spin_component);
}
