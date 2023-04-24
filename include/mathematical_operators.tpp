#include "../include/kokkos_abstractions.h"
#include "../include/specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mathematical_operators {

template <int NGLL>
KOKKOS_FUNCTION void compute_gradients_2D(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const int ispec, const specfem::kokkos::DeviceView3d<type_real> xix,
    const specfem::kokkos::DeviceView3d<type_real> xiz,
    const specfem::kokkos::DeviceView3d<type_real> gammax,
    const specfem::kokkos::DeviceView3d<type_real> gammaz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_x,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_z,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duxdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duxdz,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duzdx,
    specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL> s_duzdz) {

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

        const type_real xixl = xix(ispec, iz, ix);
        const type_real xizl = xiz(ispec, iz, ix);
        const type_real gammaxl = gammax(ispec, iz, ix);
        const type_real gammazl = gammaz(ispec, iz, ix);

        type_real sum_hprime_x1 = 0.0;
        type_real sum_hprime_x3 = 0.0;
        type_real sum_hprime_z1 = 0.0;
        type_real sum_hprime_z3 = 0.0;

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

template <int NGLL>
KOKKOS_FUNCTION void add_contributions(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<int, NGLL, NGLL> s_iglob,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        integrand_1,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        integrand_2,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        integrand_3,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        integrand_4,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        field_dot_dot) {

  assert(wxgll.extent(0) == NGLL);
  assert(wzgll.extent(0) == NGLL);

  constexpr int NGLL2 = NGLL * NGLL;
  constexpr type_real NGLL_INV = 1.0 / NGLL;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, NGLL2), [&](const int xz) {
        const int iz = xz * NGLL_INV;
        const int ix = xz - iz * NGLL;

        type_real tempx1 = 0.0;
        type_real tempz1 = 0.0;
        type_real tempx3 = 0.0;
        type_real tempz3 = 0.0;

#pragma unroll
        for (int l = 0; l < NGLL; l++) {
          tempx1 += s_hprimewgll_xx(ix, l) * integrand_1(iz, l);
          tempz1 += s_hprimewgll_xx(ix, l) * integrand_2(iz, l);
          tempx3 += s_hprimewgll_zz(iz, l) * integrand_3(l, ix);
          tempz3 += s_hprimewgll_zz(iz, l) * integrand_4(l, ix);
        }

        const int iglob = s_iglob(iz, ix);
        const type_real sum_terms1 =
            -1.0 * (wzgll(iz) * tempx1) - (wxgll(ix) * tempx3);
        const type_real sum_terms3 =
            -1.0 * (wzgll(iz) * tempz1) - (wxgll(ix) * tempz3);
        Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
        Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
      });
}
} // namespace mathematical_operators
} // namespace specfem
