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
                           specfem::element::medium_tag::elastic_psv_t, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv_t,
            specfem::element::property_tag::isotropic_cosserat, UseSIMD>
            &properties,
        const specfem::point::field<specfem::dimension::type::dim2,
                                    specfem::element::medium_tag::elastic_psv_t,
                                    true, false, false, false, UseSIMD> &u,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv_t, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_xz, sigma_zx, sigma_zz, sigma_c_yx, sigma_c_yz;

  const auto twothirds =
      static_cast<type_real>(2.0) / static_cast<type_real>(3.0);
  // P_SV case
  // sigma_xx         [lambda = kappa - 2/3*mu]
  sigma_xx = (properties.kappa() - twothirds * properties.mu()) *
                 (du(0, 0) + du(1, 1)) +
             properties.mu() * du(0, 0);

  sigma_zz = (properties.kappa() - twothirds * properties.mu()) *
                 (du(0, 0) + du(1, 1)) +
             properties.mu() * du(1, 1);

  // \sigma_{zx} = \mu (\partial_z s_x + \partial_x s_z )
  //               + 2 \nu ( -phi_y - .5 * (\partial_z s_x - \partial_x s_z) )
  sigma_zx = properties.mu() * (du(1, 0) + du(0, 1)) +
             static_cast<type_real>(2.0) * properties.nu() *
                 (-u(1) - static_cast<type_real>(0.5) * (du(1, 0) - du(0, 1)));

  // \sigma_{xz} = \mu (\partial_x s_z + \partial_z s_x )
  //               + 2 \nu ( phi_y - .5 * (\partial_x s_z - \partial_z s_x) )
  sigma_xz = properties.mu() * (du(0, 1), du(1, 0)) +
             static_cast<type_real>(2.0) * properties.nu() *
                 (u(1) - static_cast<type_real>(0.5) * (du(0, 1) - du(1, 0)));

  // I'm not sure about the sign in this expression
  // in notes on spin equation (122) suggest +, but my derivation gets to
  // -
  // It is important to note here that the dimension along which the divergence
  // operator is applied is crucial.
  // t^c_{yx} = (\mu_c + \nu_c) * \partial_x \phi_y
  // or
  // t^c_{yx} = (\mu_c - \nu_c) * \partial_x \phi_y
  // but then below as well...
  sigma_c_yx = (properties.mu_c() + properties.nu_c()) * du(0, 2);

  // t^c_{yz} = (\mu_c + \nu_c) * \partial_z \phi_y
  // or
  // t^c_{yz} = (\mu_c - \nu_c) * \partial_z \phi_y
  sigma_c_yz = (properties.mu_c() + properties.nu_c()) * du(1, 2);

  specfem::datatype::VectorPointViewType<type_real, 2, 3, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_zx;
  T(1, 1) = sigma_zz;
  T(2, 0) = sigma_c_yx;
  T(2, 1) = sigma_c_yz;

  return { T };
}

} // namespace medium
} // namespace specfem
