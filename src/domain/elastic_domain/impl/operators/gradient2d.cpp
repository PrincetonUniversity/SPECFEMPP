#include "domain/elastic_domain/impl/operators/gradient2d.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

KOKKOS_FUNCTION
void specfem::Domain::elastic::impl::operators::gradient2d::operator()(
    const int &xz, const int &ispec,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_xx,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprime_zz,
    const specfem::kokkos::DeviceScratchView2d<type_real> field_x,
    const specfem::kokkos::DeviceScratchView2d<type_real> field_z,
    type_real &duxdxl, type_real &duxdzl, type_real &duzdxl,
    type_real &duzdzl) const {

  const int ngllz = this->partial_derivatives.xix.extent(1);
  const int ngllx = this->partial_derivatives.xix.extent(2);
  const int ngll2 = ngllz * ngllx;

  assert(this->partial_derivatives.xiz.extent(1) == ngllz);
  assert(this->partial_derivatives.xiz.extent(2) == ngllx);
  assert(this->partial_derivatives.gammax.extent(1) == ngllz);
  assert(this->partial_derivatives.gammax.extent(2) == ngllx);
  assert(this->partial_derivatives.gammaz.extent(1) == ngllz);
  assert(this->partial_derivatives.gammaz.extent(2) == ngllx);

  assert(s_hprime_xx.extent(0) == ngllx);
  assert(s_hprime_xx.extent(1) == ngllx);

  assert(s_hprime_xx.extent(0) == ngllz);
  assert(s_hprime_xx.extent(1) == ngllz);

  assert(field_x.extent(0) == ngllz);
  assert(field_x.extent(1) == ngllx);
  assert(field_z.extent(0) == ngllz);
  assert(field_z.extent(1) == ngllx);

  int iz, ix;
  sub2ind(xz, ngllz, iz, ix);

  const type_real xixl = this->partial_derivatives.xix(ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(ispec, iz, ix);
  const type_real gammaxl = this->partial_derivatives.gammax(ispec, iz, ix);
  const type_real gammazl = this->partial_derivatives.gammaz(ispec, iz, ix);

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
  duxdxl = xixl * sum_hprime_x1 + gammaxl * sum_hprime_x3;

  // duxdz
  duxdzl = xizl * sum_hprime_x1 + gammazl * sum_hprime_x3;

  // duzdx
  duzdxl = xixl * sum_hprime_z1 + gammaxl * sum_hprime_z3;

  // duzdz
  duzdzl = xizl * sum_hprime_z1 + gammazl * sum_hprime_z3;

  return;
}
