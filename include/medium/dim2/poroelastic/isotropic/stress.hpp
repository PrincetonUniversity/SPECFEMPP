#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
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

  datatype sigma_xx, sigma_zz, sigma_xz, sigmap;

  // Poroelastic case
  //  sigma_xx = lambdalplus2mul_G*dux_dxl + lambdal_G*duz_dzl + C_biot*(dwx_dxl
  //  + dwz_dzl) sigma_xz = mu_G*(duz_dxl + dux_dzl) sigma_zz =
  //  lambdalplus2mul_G*duz_dzl + lambdal_G*dux_dxl + C_biot*(dwx_dxl + dwz_dzl)
  //  sigmap = C_biot*(dux_dxl + duz_dzl) + M_biot*(dwx_dxl + dwz_dzl)

  // sigma_xx
  sigma_xx = properties.lambdaplus2mu_G() * du(0, 0) +
             properties.lambda_G() * du(1, 1) +
             properties.C_Biot() * (du(2, 0) + du(3, 1));

  // sigma_zz
  sigma_zz = properties.lambdaplus2mu_G() * du(1, 1) +
             properties.lambda_G() * du(0, 0) +
             properties.C_Biot() * (du(2, 0) + du(3, 1));

  // sigma_xz
  sigma_xz = properties.mu_G() * (du(0, 1) + du(1, 0));

  // sigmap
  sigmap = properties.C_Biot() * (du(0, 0) + du(1, 1)) +
           properties.M_Biot() * (du(2, 0) + du(3, 1));

  specfem::datatype::VectorPointViewType<type_real, 4, 2, UseSIMD> T;

  T(0, 0) = sigma_xx - properties.phi() / properties.tortuosity() * sigmap;
  T(1, 0) = sigma_xz;
  T(0, 1) = sigma_xz;
  T(1, 1) = sigma_zz - properties.phi() / properties.tortuosity() * sigmap;
  T(2, 0) = sigmap - properties.rho_f() / properties.rho_bar() * sigma_xx;
  T(3, 0) = static_cast<type_real>(-1.0) * properties.rho_f() /
            properties.rho_bar() * sigma_xz;
  T(2, 1) = static_cast<type_real>(-1.0) * properties.rho_f() /
            properties.rho_bar() * sigma_xz;
  T(3, 1) = sigmap - properties.rho_f() / properties.rho_bar() * sigma_zz;

  return { T };
}

} // namespace medium
} // namespace specfem
