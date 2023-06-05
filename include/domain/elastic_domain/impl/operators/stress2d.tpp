#ifndef ELASTIC_STRESS_2D_TPP
#define ELASTIC_STRESS_2D_TPP

#include "compute/interface.hpp"
#include "domain/elastic_domain/impl/operators/stress2d.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

template <int NGLL>
KOKKOS_FUNCTION void
specfem::Domain::elastic::impl::operators::stress2d::operator()(
    const int &xz, const int &ispec, const type_real &duxdxl,
    const type_real &duxdzl, const type_real &duzdxl, const type_real &duzdzl,
    type_real &stress_integrand_1l, type_real &stress_integrand_2l,
    type_real &stress_integrand_3l, type_real &stress_integrand_4l) const {

  int ix, iz;
  sub2ind(xz, NGLL, iz, ix);

  const type_real lambdaplus2mul =
      this->properties.lambdaplus2mu(ispec, iz, ix);
  const type_real mul = this->properties.mu(ispec, iz, ix);
  const type_real lambdal = lambdaplus2mul - 2.0 * mul;

  const type_real xixl = this->partial_derivatives.xix(ispec, iz, ix);
  const type_real xizl = this->partial_derivatives.xiz(ispec, iz, ix);
  const type_real gammaxl = this->partial_derivatives.gammax(ispec, iz, ix);
  const type_real gammazl = this->partial_derivatives.gammaz(ispec, iz, ix);
  const type_real jacobianl = this->partial_derivatives.jacobian(ispec, iz, ix);

  type_real sigma_xx, sigma_zz, sigma_xz;

  if (specfem::globals::simulation_wave == specfem::wave::p_sv) {
    // P_SV case
    // sigma_xx
    sigma_xx = lambdaplus2mul * duxdxl + lambdal * duzdzl;

    // sigma_zz
    sigma_zz = lambdaplus2mul * duzdzl + lambdal * duxdxl;

    // sigma_xz
    sigma_xz = mul * (duzdxl + duxdzl);
  } else if (specfem::globals::simulation_wave == specfem::wave::sh) {
    // SH-case
    // sigma_xx
    sigma_xx = mul * duxdxl; // would be sigma_xy in
                             // CPU-version

    // sigma_xz
    sigma_xz = mul * duxdzl; // sigma_zy
  }

  stress_integrand_1l = jacobianl * (sigma_xx * xixl + sigma_xz * xizl);
  stress_integrand_2l = jacobianl * (sigma_xz * xixl + sigma_zz * xizl);
  stress_integrand_3l = jacobianl * (sigma_xx * gammaxl + sigma_xz * gammazl);
  stress_integrand_4l = jacobianl * (sigma_xz * gammaxl + sigma_zz * gammazl);

  return;
}

#endif
