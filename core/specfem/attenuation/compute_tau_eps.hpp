#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Misfit of the Q fit, in double precision.
 *
 * \f[
 *   \sum_i \frac{|\tan\delta(f_i) - 1/Q|}{1/Q},
 * \f]
 * with \f$\tan\delta = B/A\f$ from specfem::attenuation::maxwell(). A
 * vanishing storage modulus \f$A\f$ is penalized with a large \f$\tan\delta\f$.
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationObjective {
  double iQ = 0; ///< \f$1/Q\f$, the target loss tangent
  Kokkos::View<double[specfem::constants::NF_ATTENUATION], Kokkos::LayoutRight,
               Kokkos::HostSpace>
      f; ///< Evaluation frequencies (Hz)
  Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_sigma; ///< Stress relaxation times \f$\tau_\sigma\f$

  /**
   * @brief Evaluate the misfit for candidate strain relaxation times.
   * @param tau_eps Candidate \f$\tau_\epsilon\f$
   * @return Sum of absolute relative \f$1/Q\f$ misfits
   */
  double operator()(const Kokkos::View<double[N_SLS], Kokkos::LayoutRight,
                                       Kokkos::HostSpace> &tau_eps) const;
};

/**
 * @brief Strain relaxation times that give a constant Q over the band.
 *
 * Fits the generalized Maxwell solid of specfem::attenuation::maxwell() to
 * \f$Q(\omega) = Q\f$ at specfem::constants::NF_ATTENUATION frequencies
 * equally spaced in \f$\log_{10}\f$ over the band, with
 * specfem::optimization::NelderMeadSimplex started from
 * \f$\tau_\epsilon = \tau_\sigma(1 + 2/Q)\f$. This is the
 * attenuation_invert_by_simplex of the SPECFEM Fortran codes: the search runs
 * in double precision with their single-precision fminsearch tolerances and
 * simplex steps.
 *
 * Q is used as given; callers that limit Q do so through
 * specfem::attenuation::QualityFactorTable.
 *
 * @note Stricter than the Fortran codes, which accept a fit that stops on
 * its iteration budget: here any fit that does not meet the simplex
 * tolerances throws, so an unconverged \f$\tau_\epsilon\f$ never reaches the
 * solver.
 *
 * @tparam N_SLS Number of standard linear solids (> 1)
 * @tparam T Unit of the band (frequency or period)
 * @param Q Target quality factor, positive and finite
 * @param tau_sigma Stress relaxation times from compute_tau_sigma()
 * @param band Attenuation band
 * @return Strain relaxation times \f$\tau_\epsilon\f$
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
