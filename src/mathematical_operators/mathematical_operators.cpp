#include "kokkos_abstractions.h"
#include "mathematical_operators/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

KOKKOS_FUNCTION void specfem::mathematical_operators::compute_gradients_2D(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const int ispec, const specfem::kokkos::DeviceView3d<type_real> xix,
    const specfem::kokkos::DeviceView3d<type_real> xiz,
    const specfem::kokkos::DeviceView3d<type_real> gammax,
    const specfem::kokkos::DeviceView3d<type_real> gammaz,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz,
    const specfem::kokkos::DeviceScratchView2d<type_real> field_x,
    const specfem::kokkos::DeviceScratchView2d<type_real> field_z,
    specfem::kokkos::DeviceScratchView2d<type_real> s_duxdx,
    specfem::kokkos::DeviceScratchView2d<type_real> s_duxdz,
    specfem::kokkos::DeviceScratchView2d<type_real> s_duzdx,
    specfem::kokkos::DeviceScratchView2d<type_real> s_duzdz) {

  const int ngllz = xix.extent(1);
  const int ngllx = xix.extent(2);
  const int ngll2 = ngllz * ngllx;

  assert(xiz.extent(1) == ngllz);
  assert(xiz.extent(2) == ngllx);
  assert(gammax.extent(1) == ngllz);
  assert(gammax.extent(2) == ngllx);
  assert(gammaz.extent(1) == ngllz);
  assert(gammaz.extent(2) == ngllx);

  assert(s_hprime_xx.extent(0) == ngllx);
  assert(s_hprime_xx.extent(1) == ngllx);

  assert(s_hprime_xx.extent(0) == ngllz);
  assert(s_hprime_xx.extent(1) == ngllz);

  assert(field_x.extent(0) == ngllz);
  assert(field_x.extent(1) == ngllx);
  assert(field_z.extent(0) == ngllz);
  assert(field_z.extent(1) == ngllx);

  assert(s_duxdx.extent(0) == ngllz);
  assert(s_duxdx.extent(1) == ngllx);
  assert(s_duxdz.extent(0) == ngllz);
  assert(s_duxdz.extent(1) == ngllx);
  assert(s_duzdx.extent(0) == ngllz);
  assert(s_duzdx.extent(1) == ngllx);
  assert(s_duzdz.extent(0) == ngllz);
  assert(s_duzdz.extent(1) == ngllx);

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, ngll2), [&](const int xz) {
        int iz, ix;
        sub2ind(xz, ngllx, iz, ix);

        const type_real xixl = xix(ispec, iz, ix);
        const type_real xizl = xiz(ispec, iz, ix);
        const type_real gammaxl = gammax(ispec, iz, ix);
        const type_real gammazl = gammaz(ispec, iz, ix);

        type_real sum_hprime_x1 = 0.0;
        type_real sum_hprime_x3 = 0.0;
        type_real sum_hprime_z1 = 0.0;
        type_real sum_hprime_z3 = 0.0;

        for (int l = 0; l < ngllx; l++) {
          sum_hprime_x1 += s_hprime_xx(ix, l) * field_x(iz, l);
          sum_hprime_x3 += s_hprime_xx(ix, l) * field_z(iz, l);
        }

        for (int l = 0; l < ngllz; l++) {
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

KOKKOS_FUNCTION void specfem::mathematical_operators::add_contributions(
    const specfem::kokkos::DeviceTeam::member_type &team_member,
    const specfem::kokkos::DeviceView1d<type_real> wxgll,
    const specfem::kokkos::DeviceView1d<type_real> wzgll,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz,
    const specfem::kokkos::DeviceScratchView2d<int> s_iglob,
    const specfem::kokkos::DeviceScratchView2d<type_real> integrand_1,
    const specfem::kokkos::DeviceScratchView2d<type_real> integrand_2,
    const specfem::kokkos::DeviceScratchView2d<type_real> integrand_3,
    const specfem::kokkos::DeviceScratchView2d<type_real> integrand_4,
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
        field_dot_dot) {

  const int ngllz = integrand_1.extent(0);
  const int ngllx = integrand_1.extent(1);

  assert(s_hprimewgll_xx.extent(0) == ngllx);
  assert(s_hprimewgll_xx.extent(1) == ngllx);

  assert(s_hprimewgll_zz.extent(0) == ngllz);
  assert(s_hprimewgll_zz.extent(1) == ngllz);

  assert(s_iglob.extent(0) == ngllz);
  assert(s_iglob.extent(1) == ngllx);

  assert(integrand_1.extent(0) == ngllz);
  assert(integrand_1.extent(1) == ngllx);

  assert(integrand_2.extent(0) == ngllz);
  assert(integrand_2.extent(1) == ngllx);

  assert(integrand_3.extent(0) == ngllz);
  assert(integrand_3.extent(1) == ngllx);

  assert(integrand_4.extent(0) == ngllz);
  assert(integrand_4.extent(1) == ngllx);

  assert(wxgll.extent(0) == ngllx);
  assert(wzgll.extent(0) == ngllz);

  const int ngll2 = ngllx * ngllz;

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, ngll2), [&](const int xz) {
        int iz, ix;
        sub2ind(xz, ngllz, iz, ix);

        type_real tempx1 = 0.0;
        type_real tempz1 = 0.0;
        type_real tempx3 = 0.0;
        type_real tempz3 = 0.0;

        for (int l = 0; l < ngllx; l++) {
          tempx1 += s_hprimewgll_xx(ix, l) * integrand_1(iz, l);
          tempz1 += s_hprimewgll_xx(ix, l) * integrand_2(iz, l);
        }

        for (int l = 0; l < ngllz; l++) {
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
