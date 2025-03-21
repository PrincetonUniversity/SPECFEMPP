#pragma once

#include "enumerations/medium.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::poroelastic, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::poroelastic,
            specfem::element::property_tag::isotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::poroelastic, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz, sigma_p;

  const auto lambda_G = properties.lambda_G();
  const auto lambdaplus2mu_G = properties.lambdaplus2mu_G();
  const auto mu_G = properties.mu_G();
  const auto C_biot = properties.C_Biot();
  const auto M_biot = properties.M_Biot();
  const auto phi = properties.phi();
  const auto tort = properties.tortuosity();
  const auto rho_f = properties.rho_f();
  const auto rho_bar = properties.rho_bar();
  const auto dux_dxl = du(0, 0);
  const auto dux_dzl = du(1, 0);
  const auto duz_dxl = du(0, 1);
  const auto duz_dzl = du(1, 1);
  const auto dwx_dxl = du(0, 2);
  const auto dwx_dzl = du(1, 2);
  const auto dwz_dxl = du(0, 3);
  const auto dwz_dzl = du(1, 3);

  sigma_xx = (lambdaplus2mu_G)*dux_dxl + lambda_G * duz_dzl +
             C_biot * (dwx_dxl + dwz_dzl);
  sigma_xz = mu_G * (duz_dxl + dux_dzl);
  sigma_zz = (lambdaplus2mu_G)*duz_dzl + lambda_G * dux_dxl +
             C_biot * (dwx_dxl + dwz_dzl);

  sigma_p = C_biot * (dux_dxl + duz_dzl) + M_biot * (dwx_dxl + dwz_dzl);

  specfem::datatype::VectorPointViewType<type_real, 2, 4, UseSIMD> T;

  T(0, 0) = sigma_xx - phi / tort * sigma_p; // Ts_xx
  T(1, 0) = sigma_xz;                        // Ts_xz
  T(0, 1) = sigma_xz;                        // Ts_zx
  T(1, 1) = sigma_zz - phi / tort * sigma_p; // Ts_zz
  T(0, 2) = static_cast<type_real>(-1.0) * rho_f / rho_bar * sigma_xx +
            sigma_p;                                                   // Tf_xx
  T(1, 2) = static_cast<type_real>(-1.0) * rho_f / rho_bar * sigma_xz; // Tf_xz
  T(0, 3) = static_cast<type_real>(-1.0) * rho_f / rho_bar * sigma_xz; // Tf_zx
  T(1, 3) = static_cast<type_real>(-1.0) * rho_f / rho_bar * sigma_zz +
            sigma_p; // Tf_zz

  return { T };
}

} // namespace medium
} // namespace specfem
