#pragma once

#include "constants.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Result containing attenuation property values
 *
 * Holds the coefficients \f$\beta\f$ and the sum
 * \f$\sum_i \tau_{\epsilon_i}/\tau_{\sigma_i}\f$ used for calculation between
 * relaxed and unrelaxed moduli.
 *
 * After computation:
 * - \f$\beta_i = \tau_{\epsilon_i}/\tau_{\sigma_i} - 1\f$ (modulus defect per
 *   mechanism)
 * - \f$\text{one\_minus\_sum\_beta} = \sum_i
 *   \tau_{\epsilon_i}/\tau_{\sigma_i}\f$
 *
 * @tparam N_SLS Number of standard linear solids
 *
 * @see Komatitsch & Tromp 1999, eq. (7)
 */
template <int N_SLS> struct AttenuationPropertyValues {
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace> beta;
  type_real one_minus_sum_beta;
};

/**
 * @brief Compute attenuation property values from relaxation times
 *
 * Computes coefficients useful for calculation between relaxed and unrelaxed
 * moduli. For each standard linear solid mechanism \f$i\f$:
 * \f[
 *   \beta_i = \frac{\tau_{\epsilon_i}}{\tau_{\sigma_i}} - 1
 * \f]
 *
 * and the sum:
 * \f[
 *   \text{one\_minus\_sum\_beta} = \sum_{i=1}^{N\_SLS}
 *   \frac{\tau_{\epsilon_i}}{\tau_{\sigma_i}}
 * \f]
 *
 * @tparam N_SLS Number of standard linear solids
 *
 * @param tau_s Stress relaxation times \f$\tau_\sigma\f$
 * @param tau_eps Strain relaxation times \f$\tau_\epsilon\f$
 *
 * @return AttenuationPropertyValues containing \f$\beta\f$ and
 *         \f$\text{one\_minus\_sum\_beta}\f$
 *
 * @see Komatitsch & Tromp 1999, eq. (7)
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
 * Computes the scaling factor to correct for physical dispersion due to
 * anelasticity. The factor accounts for velocity dispersion so that the
 * unrelaxed modulus can be computed from the reference modulus.
 *
 * The total scaling factor is:
 * \f[
 *   \text{scale\_factor} = \text{factor\_scale\_mu} \times
 *   \text{factor\_scale\_mu0}
 * \f]
 *
 * where:
 * \f[
 *   \text{factor\_scale\_mu0} = 1 + \frac{2 \ln(f_c / f_0)}{\pi Q}
 * \f]
 *
 * corrects for the logarithmic frequency dependence of velocity (Liu et al.
 * 1976, Aki & Richards 1980, eq. 5.81), and
 * \f$\text{factor\_scale\_mu}\f$ is the ratio of unrelaxed to
 * frequency-weighted relaxed modulus from the SLS mechanisms.
 *
 * @tparam N_SLS Number of standard linear solids
 *
 * @param f_c_source Central frequency of the source (Hz)
 * @param tau_eps Strain relaxation times \f$\tau_\epsilon\f$
 * @param tau_sigma Stress relaxation times \f$\tau_\sigma\f$
 * @param Q_val Target quality factor \f$Q\f$
 * @param attenuation_f0_reference Reference frequency \f$f_0\f$ (Hz) for
 *        attenuation model
 *
 * @return Physical dispersion scaling factor (expected to be between 0.5 and
 *         1.5)
 *
 * @throws std::runtime_error if the computed scale factor is outside [0.5, 1.5]
 *
 * @see Liu, H. P., Anderson, D. L. and Kanamori, H., Velocity dispersion due
 *      to anelasticity, Geophys. J. R. Astron. Soc., 47, 41-58, 1976
 * @see Aki, K. and Richards, P. G., Quantitative Seismology, 2nd ed.,
 *      eq. (5.81), 1980
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
