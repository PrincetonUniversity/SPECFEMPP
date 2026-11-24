#pragma once

#include "compute_factors.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <numbers>
#include <sstream>
#include <stdexcept>

template <int N_SLS>
specfem::attenuation::AttenuationPropertyValues<N_SLS>
specfem::attenuation::get_attenuation_property_values(
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_s,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_eps) {
  specfem::attenuation::AttenuationPropertyValues<N_SLS> result;
  result.beta =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "beta");
  double ratio = 1.0;
  for (int i = 0; i < N_SLS; ++i) {
    const double defect =
        (static_cast<double>(tau_eps(i)) / tau_s(i) - 1.0) / N_SLS;
    result.beta(i) = defect;
    ratio += defect;
  }
  result.unrelaxed_ratio = ratio;
  return result;
}

template <int N_SLS>
specfem::attenuation::ModulusScales
specfem::attenuation::get_attenuation_modulus_scales(
    const type_real f_c_source,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_eps,
    const Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight,
                       Kokkos::HostSpace> &tau_sigma,
    const type_real Q_val, const type_real attenuation_f0_reference) {
  const double q = Q_val;
  const double f_c = f_c_source;
  const double f_0 = attenuation_f0_reference;

  // Absorption-band modulus at f_c relative to f_0 (Liu et al. 1976; Aki &
  // Richards 1980, eq. 5.81).
  const double band_scale =
      1.0 + 2.0 * std::log(f_c / f_0) / (std::numbers::pi * q);

  // Storage modulus A(omega_c) and A(infinity) of the 1/L solid, M_R = 1.
  double storage = 1.0;
  double ratio = 1.0;
  for (int i = 0; i < N_SLS; ++i) {
    const double defect =
        (static_cast<double>(tau_eps(i)) / tau_sigma(i) - 1.0) / N_SLS;
    const double omega_tau = 2.0 * std::numbers::pi * f_c * tau_sigma(i);
    storage += defect / (1.0 + 1.0 / (omega_tau * omega_tau));
    ratio += defect;
  }

  specfem::attenuation::ModulusScales scales;
  scales.relaxed = band_scale / storage;
  scales.ratio = ratio;

  const double total = scales.relaxed * scales.ratio;
  if (total < 0.5 || total > 1.5) {
    std::ostringstream message;
    message << "Error in get_attenuation_modulus_scales(): unrelaxed scale "
               "factor = "
            << total << " should be between 0.5 and 1.5. "
            << "relaxed = " << scales.relaxed << ", ratio = " << scales.ratio
            << ", Q = " << q << ", f_c_source = " << f_c
            << ", attenuation_f0_reference = " << f_0
            << ". Please check your reference frequency.";
    throw std::runtime_error(message.str());
  }
  return scales;
}
