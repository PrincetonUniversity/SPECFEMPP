#include "domain/elastic_domain/impl/operators/update_acceleration2d.hpp"
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

KOKKOS_FUNCTION void
specfem::Domain::elastic::impl::operators::update_acceleration2d::operator()(
    const int &xz, const int &ispec, const int &iglob, const type_real &wxglll,
    const type_real &wzglll,
    const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_1,
    const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_2,
    const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_3,
    const specfem::kokkos::DeviceScratchView2d<type_real> stress_integrand_4,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_xx,
    const specfem::kokkos::DeviceScratchView2d<type_real> s_hprimewgll_zz)
    const {

  const int ngllz = stress_integrand_1.extent(0);
  const int ngllx = stress_integrand_1.extent(1);

  assert(s_hprimewgll_xx.extent(0) == ngllx);
  assert(s_hprimewgll_xx.extent(1) == ngllx);

  assert(s_hprimewgll_zz.extent(0) == ngllz);
  assert(s_hprimewgll_zz.extent(1) == ngllz);

  assert(stress_integrand_2.extent(0) == ngllz);
  assert(stress_integrand_2.extent(1) == ngllx);

  assert(stress_integrand_3.extent(0) == ngllz);
  assert(stress_integrand_3.extent(1) == ngllx);

  assert(stress_integrand_4.extent(0) == ngllz);
  assert(stress_integrand_4.extent(1) == ngllx);

  int ix, iz;
  sub2ind(xz, ngllx, iz, ix);
  type_real tempx1 = 0.0;
  type_real tempz1 = 0.0;
  type_real tempx3 = 0.0;
  type_real tempz3 = 0.0;

  for (int l = 0; l < ngllx; l++) {
    tempx1 += s_hprimewgll_xx(ix, l) * stress_integrand_1(iz, l);
    tempz1 += s_hprimewgll_xx(ix, l) * stress_integrand_2(iz, l);
  }

  for (int l = 0; l < ngllz; l++) {
    tempx3 += s_hprimewgll_zz(iz, l) * stress_integrand_3(l, ix);
    tempz3 += s_hprimewgll_zz(iz, l) * stress_integrand_4(l, ix);
  }

  const type_real sum_terms1 = -1.0 * (wzglll * tempx1) - (wxglll * tempx3);
  const type_real sum_terms3 = -1.0 * (wzglll * tempz1) - (wxglll * tempz3);
  Kokkos::atomic_add(&field_dot_dot(iglob, 0), sum_terms1);
  Kokkos::atomic_add(&field_dot_dot(iglob, 1), sum_terms3);
}
