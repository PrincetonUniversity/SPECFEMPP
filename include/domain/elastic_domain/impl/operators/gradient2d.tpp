#ifndef ELASTIC_GRADIENT_2D_TPP
#define ELASTIC_GRADIENT_2D_TPP

#include "compute/interface.hpp"
#include "gradient2d.hpp"
#include "kokkos_abstractions.h"

template <int NGLL>
KOKKOS_FUNCTION void
specfem::Domain::elastic::impl::operators::gradient2d::operator()(
    const int &xz, const int &ispec,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_xx,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        s_hprime_zz,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_x,
    const specfem::kokkos::StaticDeviceScratchView2d<type_real, NGLL, NGLL>
        field_z,
    type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
    type_real &duzdzl) const {

  assert(this->partial_derivatives.xix.extent(1) == NGLL);
  assert(this->partial_derivatives.xix.extent(2) == NGLL);
  assert(this->partial_derivatives.xiz.extent(1) == NGLL);
  assert(this->partial_derivatives.xiz.extent(2) == NGLL);
  assert(this->partial_derivatives.gammax.extent(1) == NGLL);
  assert(this->partial_derivatives.gammax.extent(2) == NGLL);
  assert(this->partial_derivatives.gammaz.extent(1) == NGLL);
  assert(this->partial_derivatives.gammaz.extent(2) == NGLL);

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real xixl = this->partial_derivatives.xix(ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(ispec, iz, ix);
  const type_real gammaxl = this->partial_derivatives.gammax(ispec, iz, ix);
  const type_real gammazl = this->partial_derivatives.gammaz(ispec, iz, ix);

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
  duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

  // duxdz
  duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  // duzdx
  duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

  // duzdz
  duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  return;
}

#endif
