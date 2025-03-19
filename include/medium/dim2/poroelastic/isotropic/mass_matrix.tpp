#pragma once

#include "mass_matrix.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::poroelastic,
                                      false, false, false, true, UseSIMD>
specfem::medium::impl_mass_matrix_component(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::poroelastic,
                                     specfem::element::property_tag::isotropic,
                                     UseSIMD> &properties,
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives) {

  const auto jacobian = partial_derivatives.jacobian;
  const auto rho_bar = properties.rho_bar();
  const auto phi = properties.phi();
  const auto rho_f = properties.rho_f();
  const auto tort = properties.tortuosity();

  const auto solid_component = jacobian * (rho_bar - phi * rho_f / tort);
  const auto fluid_component = jacobian *
                               (rho_bar * rho_f * tort - phi * rho_f * rho_f) /
                               (rho_bar * phi);

  return specfem::datatype::ScalarPointViewType<type_real, 4, UseSIMD>(
      solid_component, solid_component, fluid_component, fluid_component);
}
