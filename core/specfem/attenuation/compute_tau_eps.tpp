#pragma once

#include "compute_tau_eps.hpp"
#include "maxwell.hpp"
#include "specfem/constants.hpp"
#include "specfem/optimization.hpp"
#include "specfem/setup.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>

template <int N_SLS>
double specfem::attenuation::AttenuationObjective<N_SLS>::operator()(
    const Kokkos::View<double[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        &tau_eps) const {
  constexpr int nf = specfem::constants::NF_ATTENUATION;
  const auto moduli =
      specfem::attenuation::maxwell<nf, N_SLS, double>(f, tau_sigma, tau_eps);
  // Penalty when the storage modulus vanishes.
  constexpr double hugeval = 1.e30;
  double misfit = 0.0;
  for (int i = 0; i < nf; ++i) {
    const double tan_delta = std::abs(moduli.real(i)) > 0.0
                                 ? moduli.imag(i) / moduli.real(i)
                                 : hugeval;
    misfit += std::abs(tan_delta - iQ) / iQ;
  }
  return misfit;
}

template <int N_SLS, typename T>
Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
specfem::attenuation::compute_tau_eps(
    const type_real Q,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    const specfem::utilities::Band<T> &band) {
  static_assert(N_SLS > 1,
                "N_SLS must be greater than 1 for tau_eps computation");
  if (!(Q > 0) || !std::isfinite(Q)) {
    throw std::invalid_argument(
        "Quality factor must be positive and finite for the Q fit");
  }

  const double q = Q;
  const specfem::utilities::Band<specfem::units::Hertz> frequency_band(band);

  // Frequencies equally spaced in log10, in double precision.
  constexpr int nf = specfem::constants::NF_ATTENUATION;
  specfem::attenuation::AttenuationObjective<N_SLS> objective;
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

  // The Fortran fminsearch runs in double precision but writes its
  // tolerances and simplex steps as single-precision literals.
  specfem::optimization::NelderMeadOptions<N_SLS, double> options;
  options.tol_f = static_cast<double>(1.0e-4f);
  options.tol_x = static_cast<double>(1.0e-4f);
  options.initial_step = static_cast<double>(0.05f);
  options.zero_step = static_cast<double>(0.00025f);
  options.x0 = decltype(options.x0)("tau_eps_init");
  // tan(delta) ~ 1/Q at omega * tau_sigma = 1 for tau_eps ~ tau_sigma.
  for (int i = 0; i < N_SLS; ++i) {
    const double sigma = objective.tau_sigma(i);
    options.x0(i) = sigma + (sigma * 2.0 / q);
  }

  const auto result = specfem::optimization::optimize(
      specfem::optimization::NelderMeadSimplex{}, objective, options);
  if (!result.converged) {
    std::ostringstream message;
    message << "Attenuation Q fit did not converge for Q = " << q
            << " within the simplex budgets";
    throw std::runtime_error(message.str());
  }

  Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
      tau_eps("tau_epsilon");
  for (int i = 0; i < N_SLS; ++i) {
    tau_eps(i) = result.x(i);
  }
  return tau_eps;
}
