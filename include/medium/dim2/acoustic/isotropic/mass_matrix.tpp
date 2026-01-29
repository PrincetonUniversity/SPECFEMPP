#pragma once

#include "mass_matrix.hpp"

template <specfem::dimension::type DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    DimensionTag, specfem::element::medium_tag::acoustic, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<
        DimensionTag, specfem::element::medium_tag::acoustic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties) {

  return { static_cast<type_real>(1.0) / properties.kappa() };
}
