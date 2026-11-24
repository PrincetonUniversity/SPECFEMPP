#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/layered_model.hpp"
#include <cmath>

namespace fk_injection {

/**
 * @brief Classification of the incident teleseismic plane wave.
 */
enum class incident_wave_type {
  p, ///< Incident P wave
  sv ///< Incident SV wave
};

/**
 * @brief Describes the incident teleseismic plane wave for FK synthesis.
 *
 * All angles are stored in **radians**.
 */
struct IncidentWave {
  incident_wave_type type = incident_wave_type::p; ///< Wave type (P or SV)
  type_real azimuth_phi = 0; ///< Azimuth angle \f$\phi\f$ in radians
  type_real take_off_theta =
      0; ///< Take-off angle \f$\theta\f$ from vertical, in radians
  type_real origin_x = 0;    ///< x-coordinate of the reference origin in m
  type_real origin_y = 0;    ///< y-coordinate of the reference origin in m
  type_real origin_z = 0;    ///< z-coordinate of the reference origin in m,
                             ///< up, with the model top at z = 0
  type_real origin_time = 0; ///< FK time at simulation step 0 in s
  type_real amplitude = 1;   ///< Plane-wave amplitude scaling factor
  type_real gaussian_half_duration =
      0; ///< Half-duration of the source-time Gaussian in s (0 = delta)

  /**
   * @brief Compute the horizontal ray parameter.
   *
   * \f$ p = \frac{\sin\theta}{V_\text{hs}} \f$, where \f$V_\text{hs}\f$ is the
   * half-space velocity of this wave's type.
   *
   * @param model Layered model whose half-space the wave starts in
   * @return horizontal ray parameter in s/m
   */
  double ray_parameter(const LayeredModel &model) const {
    const ElasticIsotropicLayer &halfspace = model.halfspace();
    const double halfspace_velocity =
        (type == incident_wave_type::p)
            ? static_cast<double>(halfspace.p_velocity)
            : static_cast<double>(halfspace.s_velocity);
    return std::sin(static_cast<double>(take_off_theta)) / halfspace_velocity;
  }
};

} // namespace fk_injection
