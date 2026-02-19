#pragma once

#include "mass_matrix.hpp"

template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::tags::Tags<DimensionTag, specfem::element::medium_tag::acoustic, UseSIMD>>
specfem::medium_physics::impl_mass_matrix_component(
    const specfem::point::properties<specfem::tags::Tags<DimensionTag, specfem::element::medium_tag::acoustic, specfem::element::property_tag::isotropic, UseSIMD>> &properties) {

  return { static_cast<type_real>(1.0) / properties.kappa() };
}
