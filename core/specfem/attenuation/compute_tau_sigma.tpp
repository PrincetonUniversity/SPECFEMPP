#pragma once
#include "compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const type_real min_frequency, const type_real max_frequency) {
  static_assert(N_SLS > 1, "N_SLS must be greater than 1 to avoid division by zero");

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> tau_s(
      "tau_sigma");

  // logarithms of the input frequencies
  const type_real exp1 = std::log10(min_frequency);
  const type_real exp2 = std::log10(max_frequency);

  // equally spaced in log10 frequency
  const type_real dexpval =
      (exp2 - exp1) / (static_cast<type_real>(N_SLS) - 1);

  for (int i = 0; i < N_SLS; ++i) {
    tau_s(i) = 1.0 / (specfem::constants::pi * 2.0 * std::pow(10.0, exp1 + i * dexpval));
  }

  return tau_s;
}

} // namespace attenuation
} // namespace specfem
