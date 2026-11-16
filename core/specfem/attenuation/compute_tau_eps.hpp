#pragma once

#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Strain relaxation times of the 1/L-normalized model for a target Q.
 *
 * Runs the code's single Q search,
 * specfem::attenuation::globe::search_tau_eps(), and converts its result to
 * the 1/L normalization of specfem::attenuation::maxwell():
 * \f[
 *   \tau_{\epsilon_i} = \tau_{\sigma_i}
 *     + N_\text{SLS}\,(\tau^{g}_{\epsilon_i} - \tau_{\sigma_i}),
 * \f]
 * where \f$\tau^{g}_\epsilon\f$ fits the unnormalized
 * \f$1-N_\text{SLS}+\sum_i\f$ model. Under this map both models have the
 * same loss tangent at every frequency, so the fitted Q is the same.
 *
 * Q is used as given: it is not clipped to specfem::constants::Q_MAXIMUM.
 *
 * @tparam N_SLS Number of standard linear solids (> 1)
 * @tparam T Unit of the band (frequency or period)
 * @param Q Target quality factor, positive and finite
 * @param tau_sigma Stress relaxation times from compute_tau_sigma()
 * @param band Attenuation band
 * @return Strain relaxation times \f$\tau_\epsilon\f$ of the 1/L model
 * @throws std::runtime_error if the simplex does not converge
 * @throws std::invalid_argument if Q is not positive and finite
 *
 * @code
 * constexpr int N_SLS = 3;
 * const specfem::utilities::Band<specfem::units::Hertz> band(0.1 * Hz,
 *                                                            10.0 * Hz);
 * const auto tau_sigma = specfem::attenuation::compute_tau_sigma<N_SLS>(band);
 * const auto tau_eps =
 *     specfem::attenuation::compute_tau_eps<N_SLS>(200.0, tau_sigma, band);
 * @endcode
 */
template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_eps(type_real Q,
                const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                                   Kokkos::HostSpace> &tau_sigma,
                const specfem::utilities::Band<T> &band);

} // namespace attenuation
} // namespace specfem

#include "compute_tau_eps.tpp"
