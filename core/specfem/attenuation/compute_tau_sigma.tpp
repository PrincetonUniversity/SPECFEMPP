#pragma once
#include "compute_tau_sigma.hpp"
#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const type_real min_period, const type_real max_period) {
  static_assert(N_SLS > 1, "N_SLS must be greater than 1 to avoid division by zero");

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_sigma");

  // min/max frequencies
  const type_real f1 = 1.0 / max_period;
  const type_real f2 = 1.0 / min_period;

  // logarithms
  const type_real exp1 = std::log10(f1);
  const type_real exp2 = std::log10(f2);

  // equally spaced in log10 frequency
  const type_real dexpval =
      (exp2 - exp1) / (static_cast<type_real>(N_SLS) - 1);

  for (int i = 0; i < N_SLS; ++i) {
    tau_s(i) = 1.0 / (pi * 2.0 * std::pow(10.0, exp1 + i * dexpval));
  }

  return tau_s;
}

} // namespace attenuation
} // namespace specfem