#pragma once

#include "elastic2d.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    UseSIMD>
specfem::domain::impl::elements::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, false, UseSIMD> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
        UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx =
        properties.lambdaplus2mu * du(0, 0) + properties.lambda * du(1, 1);

    // sigma_zz
    sigma_zz =
        properties.lambdaplus2mu * du(1, 1) + properties.lambda * du(0, 0);

    // sigma_xz
    sigma_xz = properties.mu * (du(0, 1) + du(1, 0));
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = properties.mu * du(0, 0); // would be sigma_xy in
                                         // CPU-version

    // sigma_xz
    sigma_xz = properties.mu * du(1, 0); // sigma_zy
  }

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> F;

  F(0, 0) =
      sigma_xx * partial_derivatives.xix + sigma_xz * partial_derivatives.xiz;
  F(0, 1) =
      sigma_xz * partial_derivatives.xix + sigma_zz * partial_derivatives.xiz;
  F(1, 0) = sigma_xx * partial_derivatives.gammax +
            sigma_xz * partial_derivatives.gammaz;
  F(1, 1) = sigma_xz * partial_derivatives.gammax +
            sigma_zz * partial_derivatives.gammaz;

  return { F };
}

template <bool UseSIMD, specfem::element::property_tag PropertyTag>
KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
                                      specfem::element::medium_tag::elastic,
                                      false, false, false, true, UseSIMD>
specfem::domain::impl::elements::impl_mass_matrix_component(
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
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
