#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

/**
 * @brief Compute stress relaxation times \f$\tau_{\sigma}\f$ equally spaced in
 * log10 frequency.
 *
 * The routine constructs a set of relaxation times corresponding to
 * frequencies that are equally spaced in base-10 logarithmic scale. Let
 * \f$f_1\f$ and \f$f_2\f$ be the minimum and maximum frequencies (Hz).
 * \f[\Delta = \frac{\log_{10} f_2 - \log_{10} f_1}{N\_SLS - 1}.\f]
 * For index \f$i\in[0, N\_SLS-1]\f$ the relaxation time is
 * \f[\tau_{\sigma,i} = \frac{1}{2\pi\,10^{\log_{10} f_1 + i\Delta}}.\f]
 *
 * @tparam N_SLS Number of standard linear solids (requires \f$N\_SLS>1\f$)
 * @param min_frequency Minimum frequency (Hz)
 * @param max_frequency Maximum frequency (Hz)
 * @return Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
 *
 * Kokkos::HostSpace>
 *
 * @code
 * // Example: build 4 SLS between 0.1 Hz and 100 Hz
 * auto tau = specfem::attenuation::compute_tau_sigma<4>(0.1_rt, 100.0_rt);
 * for (int i = 0; i < 4; ++i) std::cout << tau(i) << "\n";
 * @endcode
 */
template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_sigma(const type_real min_frequency, const type_real max_frequency);

} // namespace attenuation
} // namespace specfem

extern template Kokkos::View<type_real[specfem::constants::N_SLS],
                             Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::attenuation::compute_tau_sigma<specfem::constants::N_SLS>(
    const type_real, const type_real);
