#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Attenuation property values for modulus calculations
 *
 * Stores coefficients for computing relaxed and unrelaxed moduli:
 * - @f$ \beta^{\text{defect}}_i = \tau_{\epsilon_i}/\tau_{\sigma_i} - 1 @f$
 * (modulus defect per mechanism)
 * - @f$ \text{OneMinusSumBeta} = \sum_i
 *   \tau_{\epsilon_i}/\tau_{\sigma_i} @f$
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct AttenuationPropertyValues {
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> beta;
  type_real one_minus_sum_beta;
};

/**
 * @brief Compute attenuation property values from relaxation times
 *
 * Computes @f$ \beta^{\text{defect}}_i = \tau_{\epsilon_i}/\tau_{\sigma_i} - 1
 * @f$ and
 * @f$ \text{OneMinusSumBeta} = 1 - \sum_i \beta_i = \sum_{i=1}^{N\_SLS}
 * \tau_{\epsilon_i}/\tau_{\sigma_i} @f$ for each standard linear solid.
 *
 * @tparam N_SLS Number of standard linear solids
 * @param tau_s Stress relaxation times @f$ \tau_\sigma @f$
 * @param tau_eps Strain relaxation times @f$ \tau_\epsilon @f$
 * @return AttenuationPropertyValues containing @f$ \beta @f$ and
 *         @f$ \text{one\_minus\_sum\_beta} @f$
 */
template <int N_SLS>
AttenuationPropertyValues<N_SLS> get_attenuation_property_values(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_s,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps);

/**
 * @brief Compute physical dispersion scaling factor for attenuation
 *
 * Computes @f$ \Psi @f$ to scale @f$ \mu_0 @f$ to the unrelaxed modulus:
 * @f[
 *   \Psi = \Psi_\mu \times \Psi_{\mu_0}
 * @f]
 * where:
 * - @f$ \Psi_{\mu_0} = 1 + \frac{2}{\pi Q} \ln(f_c / f_0) @f$ corrects for
 *   logarithmic frequency dependence (Aki & Richards 1980, eq. 5.81)
 * - @f$ \Psi_\mu = \frac{\sum(1 + \beta_i/N_{\text{SLS}})}{\sum[1 + \beta_i/(1
 *   + 1/(\omega\tau_{\sigma_i})^2)/N_{\text{SLS}}]} @f$ accounts for SLS
 *   frequency dispersion
 *
 * @tparam N_SLS Number of standard linear solids
 * @param f_c_source Central frequency of the source @f$ f_c @f$ (Hz)
 * @param tau_eps Strain relaxation times @f$ \tau_\epsilon @f$
 * @param tau_sigma Stress relaxation times @f$ \tau_\sigma @f$
 * @param Q_val Target quality factor @f$ Q @f$
 * @param attenuation_f0_reference Reference frequency @f$ f_0 @f$ (Hz)
 * @return Scale factor @f$ \Psi @f$ (expected range [0.5, 1.5])
 * @throws std::runtime_error if @f$ \Psi @f$ is outside [0.5, 1.5]
 */
template <int N_SLS>
type_real get_attenuation_scale_factor(
    type_real f_c_source,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real Q_val, type_real attenuation_f0_reference);

} // namespace attenuation
} // namespace specfem
