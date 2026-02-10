#pragma once

#include "compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "maxwell.hpp"
#include "specfem/optimization.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

namespace specfem {
namespace attenuation {

/**
 * @brief Number of frequencies used for evaluating attenuation objective
 *
 * Matching SPECFEM3D Fortran implementation which uses 100 frequencies.
 */
constexpr int NF_ATTENUATION = 100;

/**
 * @brief Objective function for \f$\tau_\epsilon\f$ optimization
 *
 * This callable struct computes the misfit between the achieved \f$Q\f$ (from
 * the Maxwell solid model) and the target \f$Q\f$ value. It is designed to be
 * used with the Nelder-Mead optimizer.
 *
 * The objective is computed as the sum of squared differences:
 * \f[
 *   \sum_i \left( \tan\delta(f_i) - \frac{1}{Q_\text{target}} \right)^2
 * \f]
 *
 * where \f$\tan\delta = B/A\f$ from the Maxwell solid moduli.
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationObjective {
  type_real Q;  ///< Target quality factor \f$Q\f$
  type_real iQ; ///< \f$1/Q\f$ (target \f$\tan\delta\f$)
  Kokkos::View<type_real[NF_ATTENUATION], Kokkos::LayoutRight,
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
    auto moduli = maxwell<NF_ATTENUATION, N_SLS>(f, tau_sigma, tau_eps);

    // Compute sum of squared misfit
    type_real misfit = 0.0;
    for (int i = 0; i < NF_ATTENUATION; ++i) {
      // tan_delta = B / A ≈ 1/Q
      type_real tan_delta = moduli.B(i) / moduli.A(i);
      type_real diff = tan_delta - iQ;
      misfit += diff * diff;
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
 * 1. Computes \f$\tau_\sigma\f$ from the period range using compute_tau_sigma
 * 2. Sets up \f$N_F=100\f$ evaluation frequencies equally spaced in
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
 * @param min_period Minimum period \f$T_\text{min}\f$ (s) for frequency range
 * @param max_period Maximum period \f$T_\text{max}\f$ (s) for frequency range
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
 * auto tau_sigma = compute_tau_sigma<N_SLS>(min_period, max_period);
 * auto tau_eps = compute_tau_eps<N_SLS>(Q, tau_sigma, min_period, max_period);
 * @endcode
 */
template <int N_SLS>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
compute_tau_eps(
    type_real Q,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real min_period, type_real max_period);

} // namespace attenuation
} // namespace specfem
