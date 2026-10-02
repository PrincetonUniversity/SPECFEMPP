#pragma once

#include "specfem/medium/dim3/elastic/anisotropic/elasticity_tensor.hpp"

#include <cmath>
#include <stdexcept>

namespace specfem::globe {

/**
 * @brief Reject azimuthal anisotropy until its storage convention is supported.
 * @param gc_prime Normalized cosine azimuthal coefficient.
 * @param gs_prime Normalized sine azimuthal coefficient.
 * @throws std::runtime_error if either coefficient exceeds numerical noise.
 *
 * Full tensors synthesized through CUSTOM_REAL and rotated back to recover the
 * azimuthal coefficients can leave dimensionless residuals near single-
 * precision roundoff. Values at or below 1e-6 are therefore treated as zero.
 */
inline void ensure_supported_azimuthal_anisotropy(const double gc_prime,
                                                  const double gs_prime) {
  constexpr double numerical_noise = 1.e-6;
  if (std::abs(gc_prime) > numerical_noise ||
      std::abs(gs_prime) > numerical_noise) {
    throw std::runtime_error(
        "Globe model returned non-zero Gc_prime/Gs_prime above numerical "
        "noise; azimuthal anisotropy is deferred and cannot yet be stored");
  }
}

/**
 * @brief Select or construct a globe model's Cartesian elasticity tensor.
 *
 * @param is_full_anisotropy Whether @p model_cij contains full anisotropy.
 * @param model_cij Evaluator coefficients, already Cartesian when full.
 * @param rho Density.
 * @param vpv Vertical P-wave velocity.
 * @param vph Horizontal P-wave velocity.
 * @param vsv Vertical S-wave velocity.
 * @param vsh Horizontal S-wave velocity.
 * @param eta Dimensionless Love anisotropy parameter.
 * @param theta Geocentric colatitude in radians.
 * @param phi Geocentric longitude in radians.
 * @return Cartesian elasticity tensor.
 *
 * Full anisotropy is returned unchanged. Otherwise, the velocity/Love
 * parameters are interpreted as radial transverse isotropy and rotated once
 * into global Cartesian axes.
 */
KOKKOS_INLINE_FUNCTION
specfem::medium_physics::elasticity_tensor<double> elasticity_from_model(
    const bool is_full_anisotropy,
    const specfem::medium_physics::elasticity_tensor<double> &model_cij,
    const double rho, const double vpv, const double vph, const double vsv,
    const double vsh, const double eta, const double theta, const double phi) {
  if (is_full_anisotropy) {
    return model_cij;
  }
  const auto radial = specfem::medium_physics::love_to_radial_elasticity(
      rho, vpv, vph, vsv, vsh, eta);
  return specfem::medium_physics::rotate_elasticity_radial_to_global(
      radial, theta, phi);
}

} // namespace specfem::globe
