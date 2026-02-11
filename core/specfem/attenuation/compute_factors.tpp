#pragma once
#include "compute_factors.hpp"
#include "specfem/constants.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace specfem {
namespace attenuation {

template <int N_SLS>
AttenuationPropertyValues<N_SLS> get_attenuation_property_values(
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_s,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps) {

  AttenuationPropertyValues<N_SLS> result;
  result.beta =
      Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>(
          "beta");

  // See Komatitsch & Tromp 1999, eq. (7)
  // Coefficients beta = tau_eps / tau_s
  for (int i = 0; i < N_SLS; ++i) {
    result.beta(i) = tau_eps(i) / tau_s(i);
  }

  // Sum of coefficients beta, then subtract 1 from each beta
  // to get the modulus defect
  result.one_minus_sum_beta = 0.0;
  for (int i = 0; i < N_SLS; ++i) {
    result.one_minus_sum_beta += result.beta(i);
    result.beta(i) -= 1.0;
  }

  return result;
}

template <int N_SLS>
type_real get_attenuation_scale_factor(
    type_real f_c_source,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_eps,
    Kokkos::View<type_real[N_SLS], Kokkos::LayoutRight, Kokkos::HostSpace>
        tau_sigma,
    type_real Q_val, type_real attenuation_f0_reference) {

  // Quantity by which to scale mu_0 to get mu
  // See Liu et al. 1976; Aki & Richards 1980, eq. (5.81)
  const type_real factor_scale_mu0 =
      1.0 +
      2.0 * std::log(f_c_source / attenuation_f0_reference) / (pi * Q_val);

  // Quantity by which to scale mu to get mu_unrelaxed
  type_real sum_unrelaxed = 1.0;
  type_real sum_weighted = 1.0;

  for (int i = 0; i < N_SLS; ++i) {
    const type_real defect = tau_eps(i) / tau_sigma(i) - 1.0;
    sum_unrelaxed += defect / N_SLS;

    const type_real omega_tau = 2.0 * pi * f_c_source * tau_sigma(i);
    sum_weighted +=
        defect / (1.0 + 1.0 / (omega_tau * omega_tau)) / N_SLS;
  }

  const type_real factor_scale_mu = sum_unrelaxed / sum_weighted;

  // Total factor by which to scale mu0 to get mu_unrelaxed
  const type_real scale_factor = factor_scale_mu * factor_scale_mu0;

  // Check that the correction factor is close to one
  if (scale_factor < 0.5 || scale_factor > 1.5) {
    std::ostringstream msg;
    msg << "Error in get_attenuation_scale_factor(): "
        << "scale factor = " << scale_factor
        << " should be between 0.5 and 1.5. "
        << "factor_scale_mu = " << factor_scale_mu
        << ", factor_scale_mu0 = " << factor_scale_mu0
        << ", Q = " << Q_val
        << ", f_c_source = " << f_c_source
        << ", attenuation_f0_reference = " << attenuation_f0_reference
        << ". Please check your reference frequency.";
    throw std::runtime_error(msg.str());
  }

  return scale_factor;
}

} // namespace attenuation
} // namespace specfem
