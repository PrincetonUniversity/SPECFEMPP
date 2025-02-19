#pragma once

#include "enumerations/medium.hpp"
#include "point/field_derivatives.hpp"
#include "point/properties.hpp"
#include "point/stress.hpp"

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_sv, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::anisotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_sv, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

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

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

  return { T };
}

} // namespace medium
} // namespace specfem
