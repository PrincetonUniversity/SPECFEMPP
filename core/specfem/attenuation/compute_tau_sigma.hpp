#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

// TODO (Lucas : CPP20): and requires N_SLS>1

/**
 * @brief Compute stress relaxation times \f$\tau_{\sigma}\f$ equally spaced in
 * log10 frequency.
 *
 * The routine constructs a set of relaxation times corresponding to
 * frequencies that are equally spaced in base-10 logarithmic scale. Let
 * \f$f_1 = 1/T_{\text{max}}\f$ and \f$f_2 = 1/T_{\text{min}}\f$ be the minimum and
 * maximum frequencies, where \f$T_{\text{min}}\f$ and \f$T_{\text{max}}\f$ are the input minimum and maximum periods.
 * and
 * \f[\Delta = \frac{\log_{10} f_2 - \log_{10} f_1}{N\_SLS - 1}.\f]
 * For index \f$i\in[0, N\_SLS-1]\f$ the relaxation time is
 * \f[\tau_{\sigma,i} = \frac{1}{2\pi\,10^{\log_{10} f_1 + i\Delta}}.\f]
 *
 * @tparam N_SLS Number of standard linear solids (requires \f$N\_SLS>1\f$)
 * @param min_period Minimum period (s)
 * @param max_period Maximum period (s)
 * @return Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
 *
 * @code
 * // Example: build 4 SLS between 0.1s and 10s
 * auto tau = specfem::attenuation::compute_tau_sigma<4>(0.1_rt, 10.0_rt);
 * for (int i = 0; i < 4; ++i) std::cout << tau(i) << "\n";
 * @endcode
 */
template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const type_real min_period, const type_real max_period);

} // namespace attenuation
} // namespace specfem