#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mass_matrix.hpp"
#include "specfem/point.hpp"

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION
    specfem::point::mass_inverse<specfem::dimension::type::dim3,
                                 specfem::element::medium_tag::acoustic,
                                 UseSIMD>
    specfem::medium::impl_mass_matrix_component(
        const specfem::point::properties<specfem::dimension::type::dim3,
                                         specfem::element::medium_tag::acoustic,
                                         PropertyTag, UseSIMD> &properties) {

  return { static_cast<type_real>(1.0) / properties.kappa() };
}
