#pragma once

#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Relaxation coefficients of one fitted Q.
 *
 * For \f$L = N_\text{SLS}\f$ mechanisms of the 1/L-normalized solid
 * (specfem::attenuation::maxwell(); Tromp, Ch. 5, eq. 5.33):
 * - @f$ \beta_i = (\tau_{\epsilon_i}/\tau_{\sigma_i} - 1)/L @f$, the modulus
 *   defect of mechanism \f$i\f$ in units of the relaxed modulus:
 *   \f$\delta M_i = M_R\,\beta_i\f$ (eqs. 5.36, 5.62, 5.63);
 * - @f$ M_U/M_R = 1 + \sum_i \beta_i
 *   = \frac{1}{L}\sum_i \tau_{\epsilon_i}/\tau_{\sigma_i} @f$, the
 *   unrelaxed-to-relaxed modulus ratio (eqs. 5.34, 5.58, 5.59).
 *
 * Memory variable \f$i\f$ is driven by the strain with coefficient
 * \f$\delta M_i/\tau_{\sigma_i}\f$ (eqs. 5.64, 5.65; see
 * relaxation_rate()).
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationPropertyValues {
  /// Per-mechanism modulus defects \f$\beta_i\f$
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> beta;
  type_real unrelaxed_ratio = 1; ///< \f$M_U/M_R\f$

  /**
   * @brief Strain coefficient of memory variable j for a given unrelaxed
   * modulus.
   *
   * \f$\delta M_j/\tau_{\sigma_j} = M_R\,\beta_j/\tau_{\sigma_j}\f$ with
   * \f$M_R = M_U/(M_U/M_R)\f$, the coefficient of tr(ε) or d in eqs. 5.64
   * and 5.65. For shear the caller multiplies by 2: the solver's shear
   * memory variable is \f$2\mathbf{r}_\mu\f$, so that the stress is
   * \f$2\mu_U\mathbf{d} - \sum_\ell 2\mathbf{r}^\ell_\mu\f$ as (5.48) and
   * (5.50) require.
   *
   * @param unrelaxed_modulus Unrelaxed modulus \f$M_U\f$ at the point
   * @param j Mechanism index
   * @param tau_sigma_j Stress relaxation time of mechanism j
   * @return Strain coefficient of memory variable j
   */
  type_real relaxation_rate(const type_real unrelaxed_modulus, const int j,
                            const type_real tau_sigma_j) const {
    return unrelaxed_modulus / unrelaxed_ratio * beta(j) / tau_sigma_j;
  }
};

/**
 * @brief Compute the relaxation coefficients from relaxation times.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_s Stress relaxation times @f$ \tau_\sigma @f$
 * @param tau_eps Strain relaxation times @f$ \tau_\epsilon @f$
 * @return Defects @f$ \beta_i @f$ and @f$ M_U/M_R @f$ (see
 * AttenuationPropertyValues)
 */
template <int N_SLS>
AttenuationPropertyValues<N_SLS> get_attenuation_property_values(
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_s,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps);

/**
 * @brief Multipliers taking a reference-frequency modulus to the relaxed and
 * unrelaxed moduli of the fitted solid.
 *
 * A model modulus \f$M_0\f$ is given at the reference frequency \f$f_0\f$.
 * The relaxed modulus is \f$M_R = \text{relaxed}\cdot M_0\f$ and the
 * unrelaxed one \f$M_U = \text{ratio}\cdot M_R\f$.
 */
struct ModulusScales {
  double relaxed = 1.0; ///< \f$M_R/M_0\f$
  double ratio = 1.0;   ///< \f$M_U/M_R\f$
  /**
   * @brief Total multiplier used by scalar property storage.
   * @return \f$M_U/M_0\f$ in runtime precision
   */
  type_real unrelaxed() const { return relaxed * ratio; }
};

/**
 * @brief Compute the modulus scales of a fitted Q.
 *
 * Chooses \f$M_R\f$ so that the storage modulus at the band center matches
 * the constant-Q absorption band (Liu et al. 1976; Aki & Richards 1980,
 * eq. 5.81):
 * \f[
 *   M_1(\omega_c) = M_R\,A(\omega_c)
 *   = M_0\left[1 + \frac{2}{\pi Q}\ln\frac{f_c}{f_0}\right],
 * \f]
 * with \f$A\f$ the real part of specfem::attenuation::maxwell(). Hence
 * \f$\text{relaxed} = [1 + 2\ln(f_c/f_0)/(\pi Q)]/A(\omega_c)\f$ and
 * \f$\text{ratio} = A(\infty) = \frac{1}{L}\sum_i
 * \tau_{\epsilon_i}/\tau_{\sigma_i}\f$.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param f_c_source Band-center frequency @f$ f_c @f$ (Hz)
 * @param tau_eps Strain relaxation times @f$ \tau_\epsilon @f$
 * @param tau_sigma Stress relaxation times @f$ \tau_\sigma @f$
 * @param Q_val Quality factor @f$ Q @f$ of the fit
 * @param attenuation_f0_reference Reference frequency @f$ f_0 @f$ (Hz)
 * @return Relaxed scale and unrelaxed ratio
 * @throws std::runtime_error if @f$ M_U/M_0 @f$ is outside [0.5, 1.5]
 */
template <int N_SLS>
ModulusScales get_attenuation_modulus_scales(
    type_real f_c_source,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_sigma,
    type_real Q_val, type_real attenuation_f0_reference);

} // namespace attenuation
} // namespace specfem

extern template specfem::attenuation::AttenuationPropertyValues<
    specfem::constants::N_SLS>
specfem::attenuation::get_attenuation_property_values<
    specfem::constants::N_SLS>(
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &);

extern template specfem::attenuation::ModulusScales
specfem::attenuation::get_attenuation_modulus_scales<specfem::constants::N_SLS>(
    type_real,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    const Kokkos::View<type_real[specfem::constants::N_SLS],
                       Kokkos::LayoutRight, Kokkos::HostSpace> &,
    type_real, type_real);
