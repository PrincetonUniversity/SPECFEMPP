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
    impl_compute_stress(const specfem::point::properties<
                            specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_psv_t,
                            specfem::element::property_tag::isotropic_cosserat,
                            UseSIMD> &properties,
                        const specfem::point::field_derivatives<
                            specfem::dimension::type::dim2,
                            specfem::element::medium_tag::elastic_psv_t,
                            UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_xz, sigma_zx, sigma_zz, sigma_c_xy, sigma_c_zy;

  const auto twothirds =
      static_cast<type_real>(2.0) / static_cast<type_real>(3.0);

  sigma_xx = (properties.kappa() - twothirds * properties.mu()) *
                 (du(0, 0) + du(1, 1)) +
             static_cast<type_real>(2.0) * properties.mu() * du(0, 0);

  sigma_zz = (properties.kappa() - twothirds * properties.mu()) *
                 (du(0, 0) + du(1, 1)) +
             static_cast<type_real>(2.0) * properties.mu() * du(1, 1);

  // From Jeroen's spin notes:
  sigma_xz = properties.mu() * (du(0, 1) + du(1, 0)) +
             properties.nu() * (du(0, 1) - du(1, 0));

  sigma_zx = properties.mu() * (du(1, 0) + du(0, 1)) +
             properties.nu() * (du(1, 0) - du(0, 1));

  // Couple stress components for psv propagation
  sigma_c_xy = (properties.mu_c() + properties.nu_c()) * du(2, 0);

  sigma_c_zy = (properties.mu_c() + properties.nu_c()) * du(2, 1);

  specfem::datatype::VectorPointViewType<type_real, 3, 2, UseSIMD> T;

  // Note that the the spin notes have the divergence act on the first component
  // Komatitsch & Tromp (1999) which we are following here defines the
  // divergence as acting on the second component. so we have to implement the
  // transpose
  T(0, 0) = sigma_xx;
  T(1, 0) = sigma_xz;
  T(0, 1) = sigma_zx;
  T(1, 1) = sigma_zz;
  T(2, 0) = sigma_c_xy;
  T(2, 1) = sigma_c_zy;

  return { T };
}

} // namespace medium
} // namespace specfem
