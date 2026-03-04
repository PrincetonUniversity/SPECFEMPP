#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::mass_inverse<
    specfem::tags::Tags<specfem::element::dimension_tag::dim2, specfem::element::medium_tag::poroelastic, UseSIMD>>
specfem::medium_physics::impl_mass_matrix_component(
    const specfem::point::properties<specfem::element::dimension_tag::dim2,
                                     specfem::element::medium_tag::poroelastic,
                                     specfem::element::property_tag::isotropic,
                                     UseSIMD> &properties) {

  const auto solid_component =
      (properties.rho_bar() -
       properties.phi() * properties.rho_f() / properties.tortuosity());
  const auto fluid_component =
      (properties.rho_f() * properties.tortuosity() * properties.rho_bar() -
       properties.phi() * properties.rho_f() * properties.rho_f()) /
      (properties.phi() * properties.rho_bar());

  return { solid_component, solid_component, fluid_component, fluid_component };
}
