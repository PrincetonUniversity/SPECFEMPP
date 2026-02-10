#pragma once
#include "compute_tau_eps.hpp"
#include "maxwell.hpp"
#include "constants.hpp"
#include "specfem/optimization.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_eps(
    type_real Q,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real min_period, type_real max_period) {

  static_assert(N_SLS > 1,
                "N_SLS must be greater than 1 for tau_eps computation");

  // Set up evaluation frequencies equally spaced in log10
  // These span the same range as tau_sigma but with NF_ATTENUATION points
  Kokkos::View<type_real[NF_ATTENUATION], Kokkos::LayoutRight, Kokkos::HostSpace>
      f("frequencies");

  const type_real f1 = 1.0 / max_period; // min frequency
  const type_real f2 = 1.0 / min_period; // max frequency
  const type_real log_f1 = std::log10(f1);
  const type_real log_f2 = std::log10(f2);
  const type_real d_log_f =
      (log_f2 - log_f1) / (static_cast<type_real>(NF_ATTENUATION) - 1);

  for (int i = 0; i < NF_ATTENUATION; ++i) {
    f(i) = std::pow(10.0, log_f1 + i * d_log_f);
  }

  // Create the objective function
  AttenuationObjective<N_SLS> objective;
  objective.Q = Q;
  objective.iQ = 1.0 / Q;
  objective.f = f;
  objective.tau_sigma = tau_sigma;

  // Initial guess: tau_eps = tau_sigma * (1 + 2/Q)
  // This matches the Fortran SPECFEM3D implementation and provides a better
  // starting point for the optimization (derived from tan_delta ≈ 1/Q at
  // omega*tau = 1)
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> x0(
      "tau_eps_init");
  for (int j = 0; j < N_SLS; ++j) {
    x0(j) = tau_sigma(j) + (tau_sigma(j) * 2.0 / Q);
  }

  // Run Nelder-Mead optimization
  // Use default tolerances matching Fortran SPECFEM3D
  optimization::NelderMeadOptions<N_SLS> options;
  options.x0 = x0;
  options.max_iterations = -1;  // default max iterations
  options.tol_f = 1.0e-4;
  options.tol_x = 1.0e-4;

  auto result = optimization::optimize(optimization::NelderMeadSimplex{},
                                        objective, options);

  return result();
}

} // namespace attenuation
} // namespace specfem
