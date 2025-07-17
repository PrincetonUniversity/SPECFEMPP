#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::acoustic,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties) {

  return specfem::datatype::ScalarPointViewType<type_real, 1, UseSIMD>(
      static_cast<type_real>(1.0) / properties.kappa());
}
