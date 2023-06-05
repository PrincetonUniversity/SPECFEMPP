#ifndef ELASTIC_UPDATE_ACCELERATION_2D_TPP
#define ELASTIC_UPDATE_ACCELERATION_2D_TPP

#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include "update_acceleration2d.hpp"
#include <Kokkos_Core.hpp>

template <int NGLL>
KOKKOS_FUNCTION void
specfem::Domain::elastic::impl::operators::update_acceleration2d::operator()(
    const int &xz, const int &ispec, const int &iglob, const type_real &wxglll,
    const type_real &wzglll,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_1,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_2,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_3,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        stress_integrand_4,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprimewgll_zz) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);
  type_real tempx1 = 0.0;
  type_real tempz1 = 0.0;
  type_real tempx3 = 0.0;
  type_real tempz3 = 0.0;

#pragma unroll
  for (int l = 0; l < NGLL; l++) {
    tempx1 += s_hprimewgll_xx(ix, l) * stress_integrand_1(iz, l);
    tempz1 += s_hprimewgll_xx(ix, l) * stress_integrand_2(iz, l);
    tempx3 += s_hprimewgll_zz(iz, l) * stress_integrand_3(l, ix);
    tempz3 += s_hprimewgll_zz(iz, l) * stress_integrand_4(l, ix);
  }

  const type_real sum_terms1 = -1.0 * (wzglll * tempx1) - (wxglll * tempx3);
  const type_real sum_terms3 = -1.0 * (wzglll * tempz1) - (wxglll * tempz3);
  Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
  Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
}

#endif
