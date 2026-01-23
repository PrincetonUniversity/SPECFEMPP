#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

// TODO (Lucas : CPP20): and requires N_SLS>1

/**
 * @brief Compute stress relaxation times tau_sigma equally spaced in log10
 * frequency.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param min_period Minimum period
 * @param max_period Maximum period
 * @return Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
 */
template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const type_real min_period, const type_real max_period)

} // namespace attenuation
} // namespace specfem