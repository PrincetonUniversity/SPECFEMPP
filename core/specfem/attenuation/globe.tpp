#pragma once

#include "globe.hpp"
#include "specfem/optimization.hpp"
#include "specfem/units.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

template <int NF, int N_SLS>
specfem::attenuation::globe::MaxwellFactors<NF>
specfem::attenuation::globe::maxwell(
    const Kokkos::View<double[NF], Kokkos::LayoutRight, Kokkos::HostSpace> &f,
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_s,
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps) {
  specfem::attenuation::globe::MaxwellFactors<NF> result;
  result.real = Kokkos::View<double[NF], Kokkos::LayoutRight,
                             Kokkos::HostSpace>("real");
  result.imag = Kokkos::View<double[NF], Kokkos::LayoutRight,
                             Kokkos::HostSpace>("imag");
  for (int i = 0; i < NF; ++i) {
    const double w = 2.0 * std::numbers::pi * f(i);
    const double w2 = w * w;
    double a = 1.0 - N_SLS;
    double b = 0.0;
    for (int j = 0; j < N_SLS; ++j) {
      const double denom = 1.0 + w2 * (tau_s(j) * tau_s(j));
      a += (1.0 + w2 * tau_eps(j) * tau_s(j)) / denom;
      b += w * (tau_eps(j) - tau_s(j)) / denom;
    }
    result.real(i) = a;
    result.imag(i) = b;
  }
  return result;
}

template <int N_SLS>
double specfem::attenuation::globe::AttenuationObjective<N_SLS>::operator()(
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps) const {
  constexpr int nf = specfem::constants::NF_ATTENUATION;
  const auto moduli =
      specfem::attenuation::globe::maxwell<nf, N_SLS>(f, tau_sigma, tau_eps);
  // Globe's HUGEVAL penalty when the storage modulus vanishes.
  constexpr double hugeval = 1.e30;
  double misfit = 0.0;
  const double iQ2 = iQ * iQ;
  for (int i = 0; i < nf; ++i) {
    const double tan_delta = std::abs(moduli.real(i)) > 0.0
                                 ? moduli.imag(i) / moduli.real(i)
                                 : hugeval;
    misfit += std::sqrt((tan_delta - iQ) * (tan_delta - iQ) / iQ2);
  }
  return misfit;
}

template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::attenuation::globe::compute_tau_eps(
    type_real Q,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    const specfem::utilities::Band<T> &band) {
  static_assert(N_SLS > 1,
                "N_SLS must be greater than 1 for tau_eps computation");

  const double q = specfem::attenuation::globe::effective_quality_factor(Q);
  const specfem::utilities::Band<specfem::units::Hertz> frequency_band(band);

  // Globe's attenuation_invert_by_simplex: frequencies equally spaced in
  // log10, evaluated in double precision.
  constexpr int nf = specfem::constants::NF_ATTENUATION;
  specfem::attenuation::globe::AttenuationObjective<N_SLS> objective;
  objective.iQ = 1.0 / q;
  objective.f = decltype(objective.f)("frequencies");
  objective.tau_sigma = decltype(objective.tau_sigma)("tau_sigma");
  const double exp1 = std::log10(frequency_band.min.raw());
  const double exp2 = std::log10(frequency_band.max.raw());
  for (int i = 0; i < nf; ++i) {
    objective.f(i) = std::pow(10.0, exp1 + i * (exp2 - exp1) / (nf - 1));
  }
  for (int i = 0; i < N_SLS; ++i) {
    objective.tau_sigma(i) = tau_sigma(i);
  }

  specfem::optimization::FminsearchOptions<N_SLS> options;
  options.x0 = decltype(options.x0)("tau_eps_init");
  for (int i = 0; i < N_SLS; ++i) {
    const double sigma = objective.tau_sigma(i);
    options.x0(i) = sigma + (sigma * 2.0 / q);
  }

  const auto result = specfem::optimization::optimize(
      specfem::optimization::Fminsearch{}, objective, options);
  if (!result.converged) {
    std::ostringstream message;
    message << "Attenuation Q fit did not converge for Q = " << q
            << " within globe's simplex budgets";
    throw std::runtime_error(message.str());
  }

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_epsilon");
  for (int i = 0; i < N_SLS; ++i) {
    tau_eps(i) = result.x(i);
  }
  return tau_eps;
}

template <int N_SLS>
specfem::attenuation::AttenuationPropertyValues<N_SLS>
specfem::attenuation::globe::get_attenuation_property_values(
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_s,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_eps) {
  specfem::attenuation::AttenuationPropertyValues<N_SLS> result;
  result.beta =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "beta");
  double one_minus_sum_beta = 1.0;
  for (int i = 0; i < N_SLS; ++i) {
    const double tauinv = std::abs(tau_s(i)) > 0 ? 1.0 / tau_s(i) : 0.0;
    // Globe's beta; the stored defect has the opposite sign.
    const double globe_beta = 1.0 - tau_eps(i) * tauinv;
    result.beta(i) = -globe_beta;
    one_minus_sum_beta -= globe_beta;
  }
  result.one_minus_sum_beta = one_minus_sum_beta;
  return result;
}

template <int N_SLS>
specfem::attenuation::globe::ModulusScales
specfem::attenuation::globe::get_attenuation_modulus_scales(
    type_real f_c_source,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_eps,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    type_real Q_val, type_real attenuation_f0_reference) {
  const double q = specfem::attenuation::globe::effective_quality_factor(Q_val);
  const double f_c = f_c_source;
  const double f_0 = attenuation_f0_reference;

  // Liu et al. 1976; Aki & Richards 1980, eq. (5.81)
  const double factor_scale_mu0 =
      f_0 > 0.0 ? 1.0 + 2.0 * std::log(f_c / f_0) / (std::numbers::pi * q)
                : 1.0;

  // A, B and Omega of the complex modulus at the band center.
  const double w = 2.0 * std::numbers::pi * f_c;
  double a = 1.0, b = 0.0;
  // Relaxed-to-unrelaxed ratio, as in get_attenuation_property_values but
  // kept in double precision.
  double ratio = 1.0;
  for (int i = 0; i < N_SLS; ++i) {
    const double epsilon = tau_eps(i);
    const double sigma = tau_sigma(i);
    const double denom = 1.0 + w * w * epsilon * epsilon;
    if (denom != 0.0) {
      a -= w * w * epsilon * (epsilon - sigma) / denom;
      b += w * (epsilon - sigma) / denom;
    }
    const double tauinv = std::abs(sigma) > 0 ? 1.0 / sigma : 0.0;
    ratio -= 1.0 - epsilon * tauinv;
  }
  const double big_omega =
      a != 0.0 ? a * (std::sqrt(1.0 + b * b / (a * a)) - 1.0) : 0.0;
  const double factor_scale_mu =
      big_omega != 0.0 ? b * b / (2.0 * big_omega) : 1.0;

  specfem::attenuation::globe::ModulusScales scales;
  scales.relaxed = factor_scale_mu * factor_scale_mu0;
  scales.ratio = ratio;

  if (scales.relaxed < 0.5 || scales.relaxed > 1.5) {
    std::ostringstream message;
    message << "Error in get_attenuation_modulus_scales(): relaxed scale "
               "factor = "
            << scales.relaxed << " should be between 0.5 and 1.5. "
            << "factor_scale_mu = " << factor_scale_mu
            << ", factor_scale_mu0 = " << factor_scale_mu0 << ", Q = " << q
            << ", f_c_source = " << f_c
            << ", attenuation_f0_reference = " << f_0
            << ". Please check your reference frequency.";
    throw std::runtime_error(message.str());
  }
  return scales;
}

template <int N_SLS, typename T>
specfem::attenuation::globe::QualityFactorFit<N_SLS>
specfem::attenuation::globe::fit_quality_factor(
    type_real Q,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    const specfem::utilities::Band<T> &band, type_real f_c_source,
    type_real attenuation_f0_reference) {
  specfem::attenuation::globe::QualityFactorFit<N_SLS> fit;
  fit.tau_eps =
      specfem::attenuation::globe::compute_tau_eps<N_SLS>(Q, tau_sigma, band);
  fit.properties =
      specfem::attenuation::globe::get_attenuation_property_values<N_SLS>(
          tau_sigma, fit.tau_eps);
  fit.scales = specfem::attenuation::globe::get_attenuation_modulus_scales<N_SLS>(
      f_c_source, fit.tau_eps, tau_sigma, Q, attenuation_f0_reference);
  return fit;
}
