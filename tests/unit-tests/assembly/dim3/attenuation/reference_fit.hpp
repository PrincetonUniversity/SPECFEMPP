#pragma once

#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/attenuation/quality_factor_fit.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <array>

namespace specfem::assembly::attenuation_test_impl {

/**
 * @brief Fit of one shear Q, in the form the Fortran shift references take.
 *
 * The scales come from specfem::attenuation::fit_quality_factor(), which is
 * checked against the formulation in tests/unit-tests/attenuation. The
 * memory coefficients are written out from the fitted relaxation times, so
 * they do not reuse the solver's rate expression.
 */
struct ReferenceFit {
  double relaxed_scale = 1; ///< \f$M_R/M_0\f$
  double ratio = 1;         ///< \f$M_U/M_R\f$
  /// Modulus-free shear coefficients
  /// \f$2(\tau_{\epsilon_j}/\tau_{\sigma_j}-1)/(L\,\tau_{\sigma_j})\f$
  std::array<double, specfem::constants::N_SLS> rates{};
};

/**
 * @brief Fit a shear Q as the 3D solver does.
 *
 * @param q Quality factor; limited to (0, Q_MAXIMUM] like the 3D Q tables
 * @param fmin, fmax Attenuation band (Hz)
 * @param fc Band-center frequency (Hz)
 * @param f0 Reference frequency (Hz)
 * @return Scales and modulus-free memory coefficients
 */
inline ReferenceFit reference_fit(double q, const double fmin,
                                  const double fmax, const double fc,
                                  const double f0) {
  using specfem::units::unit_symbols::Hz;
  constexpr int n = specfem::constants::N_SLS;
  if (q <= 0 || q > specfem::constants::Q_MAXIMUM)
    q = specfem::constants::Q_MAXIMUM;
  const specfem::utilities::Band<specfem::units::Hertz> band(fmin * Hz,
                                                             fmax * Hz);
  const auto tau_sigma = specfem::attenuation::compute_tau_sigma<n>(band);
  const auto fit =
      specfem::attenuation::fit_quality_factor<n>(q, tau_sigma, band, fc, f0);
  ReferenceFit reference;
  reference.relaxed_scale = fit.scales.relaxed;
  reference.ratio = fit.scales.ratio;
  for (int j = 0; j < n; ++j) {
    const double sigma = tau_sigma(j);
    reference.rates[j] = 2 * (fit.tau_eps(j) / sigma - 1) / n / sigma;
  }
  return reference;
}

} // namespace specfem::assembly::attenuation_test_impl
