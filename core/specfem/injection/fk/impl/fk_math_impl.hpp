#pragma once

#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/complex_matrix.hpp"
#include <Kokkos_Core.hpp>
#include <numbers>

namespace specfem {
namespace injection {
namespace fk_impl {

/// @brief Mathematical constant pi.
inline constexpr double pi = std::numbers::pi_v<double>;

/// @brief Near-zero guard used for singular-denominator checks.
inline constexpr double tinyval = 1.0e-9;

/**
 * @brief Compute the vertical P- and S-wave slownesses for an elastic layer.
 *
 * Returns \f$\eta_\alpha = -i\sqrt{1/v_P^2 - p^2}\f$ and
 * \f$\eta_\beta = -i\sqrt{1/v_S^2 - p^2}\f$, matching the Tong (2014)
 * sign convention.
 *
 * @param layer          Elastic layer whose material velocities are used.
 * @param ray_parameter  Horizontal slowness p = sin(theta)/v (s/m).
 * @return Array {eta_alpha, eta_beta}.
 */
KOKKOS_INLINE_FUNCTION
Kokkos::Array<Kokkos::complex<double>, 2>
vertical_slowness(const specfem::injection::fk::ElasticIsotropicLayer &layer,
                  type_real ray_parameter) {
  const double p = static_cast<double>(ray_parameter);
  const double vp = static_cast<double>(layer.p_velocity);
  const double vs = static_cast<double>(layer.s_velocity);

  const Kokkos::complex<double> neg_i(0.0, -1.0);

  const Kokkos::complex<double> eta_alpha =
      neg_i * Kokkos::sqrt(1.0 / (vp * vp) - p * p);

  const Kokkos::complex<double> eta_beta =
      neg_i * Kokkos::sqrt(1.0 / (vs * vs) - p * p);

  return { eta_alpha, eta_beta };
}

/**
 * @brief Compute the vertical P-wave slowness for an acoustic layer.
 *
 * Returns \f$\eta_\alpha = -i\sqrt{1/v_P^2 - p^2}\f$, matching the Tong
 * (2014) sign convention.  Acoustic layers have no shear, so only
 * @f$\eta_\alpha@f$ is returned.
 *
 * @param layer          Acoustic layer whose p_velocity is used.
 * @param ray_parameter  Horizontal slowness p = sin(theta)/v (s/m).
 * @return eta_alpha.
 */
KOKKOS_INLINE_FUNCTION
Kokkos::complex<double>
vertical_slowness(const specfem::injection::fk::AcousticLayer &layer,
                  type_real ray_parameter) {
  const double p = static_cast<double>(ray_parameter);
  const double vp = static_cast<double>(layer.p_velocity);

  const Kokkos::complex<double> neg_i(0.0, -1.0);

  return neg_i * Kokkos::sqrt(1.0 / (vp * vp) - p * p);
}

/**
 * @brief Compute the elastic gamma0 factor for a layer.
 *
 * \f$ \gamma_0 = 2 v_S^2 p^2 \f$
 *
 * @param layer          Elastic layer whose shear velocity is used.
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @return gamma0 (dimensionless).
 */
KOKKOS_INLINE_FUNCTION
double
elastic_gamma0(const specfem::injection::fk::ElasticIsotropicLayer &layer,
               type_real ray_parameter) {
  const double p = static_cast<double>(ray_parameter);
  const double vs = static_cast<double>(layer.s_velocity);
  return 2.0 * vs * vs * p * p;
}

/**
 * @brief Compute the elastic gamma1 factor for a layer.
 *
 * \f$ \gamma_1 = 1 - 1/\gamma_0 \f$
 *
 * @param layer          Elastic layer whose shear velocity is used.
 * @param ray_parameter  Horizontal slowness p (s/m).
 * @return gamma1 (dimensionless).
 */
KOKKOS_INLINE_FUNCTION
double
elastic_gamma1(const specfem::injection::fk::ElasticIsotropicLayer &layer,
               type_real ray_parameter) {
  return 1.0 - 1.0 / elastic_gamma0(layer, ray_parameter);
}

} // namespace fk_impl
} // namespace injection
} // namespace specfem
