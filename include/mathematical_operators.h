#ifndef MATHEMATICAL_OPERATORS_H
#define MATHEMATICAL_OPERATORS_H

#include "../include/config.h"
#include <Kokkos_Core.hpp>

// using simd_type = Kokkos::Experimental::native_simd<type_real>;
// using mask_type = Kokkos::Experimental::native_simd_mask<double>;
// using tag_type = Kokkos::Experimental::element_aligned_tag;

namespace specfem {
namespace mathematical_operators {

template <int NGLL>
KOKKOS_FUNCTION void compute_gradients_2D(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const int ispec, const specfem::kokkos::DeviceView3d<type_real> xix,
    const specfem::kokkos::DeviceView3d<type_real> xiz,
    const specfem::kokkos::DeviceView3d<type_real> gammax,
    const specfem::kokkos::DeviceView3d<type_real> gammaz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL>
        s_hprime_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL>
        s_hprime_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> field_x,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> field_z,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_duxdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_duxdz,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_duzdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL> s_duzdz) {

  const int NGLL2 = NGLL * NGLL;
  assert(xix.extent(1) == NGLL);
  assert(xix.extent(2) == NGLL);
  assert(xiz.extent(1) == NGLL);
  assert(xiz.extent(2) == NGLL);
  assert(gammax.extent(1) == NGLL);
  assert(gammax.extent(2) == NGLL);
  assert(gammaz.extent(1) == NGLL);
  assert(gammaz.extent(2) == NGLL);

  const type_real NGLL_INV = 1.0 / NGLL;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
        const int iz = xz * NGLL_INV;
        const int ix = xz - iz * NGLL;

        const simd_type xixl = xix(ispec, iz, ix);
        const simd_type xizl = xiz(ispec, iz, ix);
        const simd_type gammaxl = gammax(ispec, iz, ix);
        const simd_type gammazl = gammaz(ispec, iz, ix);

        simd_type sum_hprime_x1 = 0.0;
        simd_type sum_hprime_x3 = 0.0;
        simd_type sum_hprime_z1 = 0.0;
        simd_type sum_hprime_z3 = 0.0;

        for (int l = 0; l < NGLL; l++) {
          sum_hprime_x1 += s_hprime_xx(ix, l) * field_x(iz, l);
          sum_hprime_x3 += s_hprime_xx(ix, l) * field_z(iz, l);
          sum_hprime_z1 += s_hprime_zz(iz, l) * field_x(l, ix);
          sum_hprime_z3 += s_hprime_zz(iz, l) * field_z(l, ix);
        }
        // duxdx
        s_duxdx(iz, ix) = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

        // duxdz
        s_duxdz(iz, ix) = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

        // duzdx
        s_duzdx(iz, ix) = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

        // duzdz
        s_duzdz(iz, ix) = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;
      });

  return;
};

} // namespace mathematical_operators
} // namespace specfem

#endif
