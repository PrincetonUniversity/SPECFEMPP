#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<
        specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
        specfem::element::property_tag::isotropic, UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim3, specfem::element::medium_tag::elastic,
        UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz;

  // P_SV case
  // sigma_xx
  sigma_xx = properties.lambdaplus2mu() * du(0, 0) +
             properties.lambda() * (du(1, 1) + du(2, 2));

  // sigma_yy
  sigma_yy = properties.lambdaplus2mu() * du(1, 1) +
             properties.lambda() * (du(0, 0) + du(2, 2));

  // sigma_zz
  sigma_zz = properties.lambdaplus2mu() * du(2, 2) +
             properties.lambda() * (du(0, 0) + du(1, 1));

  // sigma_xy
  sigma_xy = properties.mu() * (du(0, 1) + du(1, 0));

  // sigma_xz
  sigma_xz = properties.mu() * (du(0, 2) + du(2, 0));

  // sigma_yz
  sigma_yz = properties.mu() * (du(1, 2) + du(2, 1));

  specfem::datatype::TensorPointViewType<type_real, 3, 3, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(1, 1) = sigma_yy;
  T(2, 2) = sigma_zz;
  T(0, 1) = sigma_xy;
  T(1, 0) = sigma_xy;
  T(0, 2) = sigma_xz;
  T(2, 0) = sigma_xz;
  T(1, 2) = sigma_yz;
  T(2, 1) = sigma_yz;

  return { T };
}

} // namespace medium
} // namespace specfem
