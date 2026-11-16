#pragma once

#include "compute_tau_eps.hpp"
#include "globe.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>

template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::attenuation::compute_tau_eps(
    type_real Q,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    const specfem::utilities::Band<T> &band) {
  const auto fitted =
      specfem::attenuation::globe::search_tau_eps<N_SLS>(Q, tau_sigma, band);

  // The 1/L model spreads the modulus over N_SLS mechanisms, so each defect
  // tau_eps - tau_sigma is N_SLS times the unnormalized model's.
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_epsilon");
  for (int i = 0; i < N_SLS; ++i) {
    const double sigma = tau_sigma(i);
    tau_eps(i) = sigma + N_SLS * (fitted(i) - sigma);
  }
  return tau_eps;
}
