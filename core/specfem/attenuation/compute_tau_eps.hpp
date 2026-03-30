#pragma once

#include "compute_tau_sigma.hpp"
#include "maxwell.hpp"
#include "specfem/constants.hpp"
#include "specfem/optimization.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

/**
 * @brief Objective function for \f$\tau_\epsilon\f$ optimization
 *
 * This callable struct computes the misfit between the achieved \f$Q\f$ (from
 * the Maxwell solid model) and the target \f$Q\f$ value. It is designed to be
 * used with the Nelder-Mead optimizer.
 *
 * The objective is computed as the sum of absolute relative errors:
 * \f[
 *   \sum_i \frac{| \tan\delta(f_i) - \frac{1}{Q_\text{target}}
 * |}{\frac{1}{Q_\text{target}}}
 * \f]
 *
 * where \f$\tan\delta = B/A\f$ from the Maxwell solid moduli.
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationObjective {
  type_real Q;  ///< Target quality factor \f$Q\f$
  type_real iQ; ///< \f$1/Q\f$ (target \f$\tan\delta\f$)
  Kokkos::View<type_real[specfem::constants::NF_ATTENUATION],
               Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f; ///< Evaluation frequencies \f$f\f$ (Hz)
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma; ///< Stress relaxation times \f$\tau_\sigma\f$

  /**
   * @brief Evaluate the objective function for given \f$\tau_\epsilon\f$ values
   *
   * @param tau_eps Candidate strain relaxation times \f$\tau_\epsilon\f$
   * @return Sum of squared misfit between achieved and target \f$1/Q\f$
   */
  type_real operator()(
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
          tau_eps) const {

    // Compute Maxwell moduli for all frequencies
    auto maxwell_factors = maxwell<specfem::constants::NF_ATTENUATION, N_SLS>(
        f, tau_sigma, tau_eps);

    // Compute sum of absolute relative errors (L1 norm, normalized by 1/Q)
    // Matches Fortran: xi = sqrt( (tan_delta - 1/Q)^2 / (1/Q)^2 )
    //                     = |tan_delta - 1/Q| * Q
    type_real misfit = 0.0;
    const type_real iQ2 = iQ * iQ;
    for (int i = 0; i < specfem::constants::NF_ATTENUATION; ++i) {
      // tan_delta = B / A ≈ 1/Q
      type_real tan_delta = maxwell_factors.imag(i) / maxwell_factors.real(i);
      type_real diff = tan_delta - iQ;
      misfit += std::sqrt(diff * diff / iQ2);
    }

    return misfit;
  }
};

/**
 * @brief Compute strain relaxation times via simplex optimization
 *
 * This function finds the strain relaxation times \f$\tau_\epsilon\f$ that
 * achieve a target quality factor \f$Q\f$ for a generalized Maxwell solid with
 * \f$N_\text{SLS}\f$ standard linear solids.
 *
 * The algorithm:
 * 1. Sets up \f$N_F=100\f$ evaluation frequencies equally spaced in
 * \f$\log_{10}\f$
 * 3. Uses Nelder-Mead simplex to minimize the misfit between achieved
 *    and target \f$1/Q\f$ values over the frequency range
 *
 * The initial guess for \f$\tau_\epsilon\f$ is \f$\tau_\sigma\f$ (no
 * attenuation), and the optimization typically converges within a few hundred
 * iterations.
 *
 * @tparam N_SLS Number of standard linear solids (requires \f$N_\text{SLS} >
 * 1\f$)
 *
 * @param Q Target quality factor \f$Q\f$ (must be positive)
 * @param tau_sigma Pre-computed stress relaxation times \f$\tau_\sigma\f$ from
 * compute_tau_sigma
 * @param min_frequency Minimum frequency \f$f_\text{min}\f$ (Hz) for frequency
 * range
 * @param max_frequency Maximum frequency \f$f_\text{max}\f$ (Hz) for frequency
 * range
 *
 * @return View containing \f$N_\text{SLS}\f$ strain relaxation times
 * \f$\tau_\epsilon\f$
 *
 * @note The returned \f$\tau_\epsilon\f$ values should satisfy \f$\tau_\epsilon
 * > \tau_\sigma\f$ for positive \f$Q\f$ (physical attenuation).
 *
 * @code
 * // Example: compute tau_eps for Q=200 with 3 SLS
 * constexpr int N_SLS = 3;
 * type_real Q = 200.0;
 * type_real min_period = 0.01;
 * type_real max_period = 10.0;
 * auto tau_sigma = compute_tau_sigma<N_SLS>(min_frequency, max_frequency);
 * auto tau_eps = compute_tau_eps<N_SLS>(Q, tau_sigma, min_frequency,
 * max_frequency);
 * @endcode
 */
template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_eps(
    type_real Q,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real min_frequency, type_real max_frequency);

} // namespace attenuation
} // namespace specfem

extern template Kokkos::View<type_real[specfem::constants::N_SLS],
                             Kokkos::LayoutRight, Kokkos::HostSpace>
    specfem::attenuation::compute_tau_eps<specfem::constants::N_SLS>(
        type_real,
        Kokkos::View<type_real[specfem::constants::N_SLS], Kokkos::LayoutRight,
                     Kokkos::HostSpace>,
        type_real, type_real);
