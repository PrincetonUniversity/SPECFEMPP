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

  datatype sigma_xx, sigma_zz, sigma_xz, sigmap, lambda_G, lambdaplus2mu_G;

  // Poroelastic case
  //  sigma_xx = lambdalplus2mul_G*dux_dxl + lambdal_G*duz_dzl + C_biot*(dwx_dxl
  //  + dwz_dzl) sigma_xz = mu_G*(duz_dxl + dux_dzl) sigma_zz =
  //  lambdalplus2mul_G*duz_dzl + lambdal_G*dux_dxl + C_biot*(dwx_dxl + dwz_dzl)
  //  sigmap = C_biot*(dux_dxl + duz_dzl) + M_biot*(dwx_dxl + dwz_dzl)

  // lambda_G
  lambda_G = properties.H_Biot() - 2 * properties.mu_G();

  // lambdaplus2mu_G
  lambdaplus2mu_G = lambda_G + 2 * properties.mu_G();

  // sigma_xx
  sigma_xx = lambdaplus2mu_G * du(0, 0) + lambda_G * du(1, 1) +
             properties.C_Biot() * (du(0, 2) + du(1, 3));

  // sigma_zz
  sigma_zz = lambdaplus2mu_G * du(1, 1) + lambda_G * du(0, 0) +
             properties.C_Biot() * (du(0, 2) + du(1, 3));

  // sigma_xz
  sigma_xz = properties.mu_G() * (du(0, 1) + du(1, 0));

  // sigmap
  sigmap = properties.C_Biot() * (du(0, 0) + du(1, 1)) +
           properties.M_Biot() * (du(0, 2) + du(1, 3));

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> T;
  specfem::datatype::ScalarPointViewType<type_real, 1, UseSIMD> P;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;
  P = sigmap;

  return { T, P };
}

} // namespace medium
} // namespace specfem
