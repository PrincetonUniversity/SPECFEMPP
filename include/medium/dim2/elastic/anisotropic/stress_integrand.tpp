#pragma once
#pragma once

#include "stress_integrand.hpp"

template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::stress_integrand<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    UseSIMD>
specfem::medium::impl_compute_stress_integrands(
    const specfem::point::partial_derivatives<
        specfem::dimension::type::dim2, false, UseSIMD> &partial_derivatives,
    const specfem::point::properties<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
        specfem::element::property_tag::anisotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
        UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  static_assert(specfem::globals::simulation_wave == specfem::wave::p_sv,
                "Only P-SV are supported for stress integrand computation.");

  datatype sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx = properties.c11 * du(0, 0) + properties.c13 * du(1, 1) +
              properties.c15 * (du(1, 0) + du(0, 1));

    // sigma_zz
    sigma_zz = properties.c13 * du(0, 0) + properties.c33 * du(1, 1) +
              properties.c35 * (du(1, 0) + du(0, 1));

    // sigma_xz
    sigma_xz = properties.c15 * du(0, 0) + properties.c35 * du(1, 1) +
              properties.c55 * (du(1, 0) + du(0, 1));

  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // // SH-case
    // // sigma_xx
    // sigma_xx = properties.mu * du(0, 0); // would be sigma_xy in
    //                                      // CPU-version

    // // sigma_xz
    // sigma_xz = properties.mu * du(1, 0); // sigma_zy
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
