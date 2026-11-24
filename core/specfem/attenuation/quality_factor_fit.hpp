#pragma once

#include "compute_factors.hpp"
#include "compute_tau_eps.hpp"
#include "quality_factor_table.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace attenuation {

/**
 * @brief Everything derived from one quality factor.
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS> struct QualityFactorFit {
  /// Strain relaxation times \f$\tau_\epsilon\f$
  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps;
  /// Defects and \f$M_U/M_R\f$
  specfem::attenuation::AttenuationPropertyValues<N_SLS> properties;
  /// Reference-to-relaxed scale and \f$M_U/M_R\f$
  specfem::attenuation::ModulusScales scales;

  /**
   * @brief The fit of an unattenuated modulus.
   *
   * Unit scales and zero defects: the modulus is not shifted and its
   * relaxation rates vanish.
   *
   * @return Identity fit
   */
  static QualityFactorFit identity() {
    QualityFactorFit fit;
    fit.properties.beta = decltype(fit.properties.beta)("beta");
    return fit;
  }
};

/**
 * @brief Fit Q and derive its relaxation coefficients and modulus scales.
 *
 * Runs compute_tau_eps(), get_attenuation_property_values() and
 * get_attenuation_modulus_scales() in turn. Q is used as given.
 *
 * @tparam N_SLS Number of standard linear solids (> 1)
 * @tparam T Unit of the band (frequency or period)
 * @param Q Quality factor, positive and finite
 * @param tau_sigma Stress relaxation times from compute_tau_sigma()
 * @param band Attenuation band
 * @param f_c_source Band-center frequency (Hz)
 * @param attenuation_f0_reference Model reference frequency (Hz)
 * @return The fit and everything derived from it
 * @throws std::runtime_error if the Q fit does not converge or
 * \f$M_U/M_0\f$ is outside [0.5, 1.5]
 */
template <int N_SLS, typename T>
QualityFactorFit<N_SLS> fit_quality_factor(
    const type_real Q,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_sigma,
    const specfem::utilities::Band<T> &band, const type_real f_c_source,
    const type_real attenuation_f0_reference) {
  QualityFactorFit<N_SLS> fit;
  fit.tau_eps =
      specfem::attenuation::compute_tau_eps<N_SLS>(Q, tau_sigma, band);
  fit.properties = specfem::attenuation::get_attenuation_property_values<N_SLS>(
      tau_sigma, fit.tau_eps);
  fit.scales = specfem::attenuation::get_attenuation_modulus_scales<N_SLS>(
      f_c_source, fit.tau_eps, tau_sigma, Q, attenuation_f0_reference);
  return fit;
}

/**
 * @brief Bulk and shear tables of QualityFactorFit.
 *
 * Clips Q to specfem::constants::Q_MAXIMUM by default, as the SPECFEM3D and
 * SPECFEM3D_GLOBE meshers do, so 3D callers cannot forget it. 2D passes 0 to
 * use Q as given, as SPECFEM2D does.
 *
 * @tparam N_SLS Number of standard linear solids
 */
template <int N_SLS>
struct QualityFactorFitTables
    : specfem::attenuation::QualityFactorTables<QualityFactorFit<N_SLS>> {
  /**
   * @brief Create empty tables.
   * @param maximum Q limit (nonpositive: none); see QualityFactorTable
   */
  explicit QualityFactorFitTables(
      const type_real maximum = specfem::constants::Q_MAXIMUM)
      : specfem::attenuation::QualityFactorTables<QualityFactorFit<N_SLS>>(
            maximum) {}
};

} // namespace attenuation
} // namespace specfem
