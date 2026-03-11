#pragma once

#include "compute_integration_factors.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

template <int N_SLS>
IntegrationFactors<N_SLS> compute_integration_factors(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real deltat) {

  IntegrationFactors<N_SLS> result;
  result.alpha =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "alpha_rk");
  result.beta =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "beta_rk");
  result.gamma =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "gamma_rk");

  const type_real dt = deltat;
  const type_real dt2 = dt * dt;
  const type_real dt3 = dt2 * dt;
  const type_real dt4 = dt3 * dt;

  for (int j = 0; j < N_SLS; ++j) {
    // Negative sign follows Fortran convention in get_attenuation_model.f90
    const type_real tauinv  = -1.0 / tau_sigma(j);
    const type_real tauinv2 = tauinv * tauinv;
    const type_real tauinv3 = tauinv2 * tauinv;
    const type_real tauinv4 = tauinv3 * tauinv;

    // Savage et al. BSSA 2010, eq. (11)
    result.alpha(j) = 1.0
                    + dt  * tauinv
                    + dt2 * tauinv2 / 2.0
                    + dt3 * tauinv3 / 6.0
                    + dt4 * tauinv4 / 24.0;

    result.beta(j)  = dt  / 2.0
                    + dt2 * tauinv  / 3.0
                    + dt3 * tauinv2 / 8.0
                    + dt4 * tauinv3 / 24.0;

    result.gamma(j) = dt  / 2.0
                    + dt2 * tauinv  / 6.0
                    + dt3 * tauinv2 / 24.0;
  }

  return result;
}

} // namespace attenuation
} // namespace specfem
