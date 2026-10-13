#pragma once

#include "specfem/setup.hpp"

namespace specfem {
namespace injection {
namespace fk {

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
 * All angles are stored in **radians**.  The back-azimuth to azimuth-phi
 * convention conversion lives in the runtime configuration layer, not here.
 *
 * The @c ray_parameter() helper derives \f$ p = \sin\theta / V_\text{hs} \f$
 * from the take-off angle and the half-space velocity supplied by the caller.
 */
struct IncidentWave {
  incident_wave_type type = incident_wave_type::p; ///< Wave type (P or SV)
  type_real azimuth_phi = 0; ///< Azimuth angle \f$\phi\f$ in radians
  type_real take_off_theta =
      0; ///< Take-off angle \f$\theta\f$ from vertical, in radians
  type_real origin_x = 0;    ///< x-coordinate of the reference origin in m
  type_real origin_y = 0;    ///< y-coordinate of the reference origin in m
  type_real origin_z = 0;    ///< z-coordinate of the reference origin in m
  type_real origin_time = 0; ///< Reference origin time in s
  type_real amplitude = 1;   ///< Plane-wave amplitude scaling factor
  type_real gaussian_half_duration =
      0; ///< Half-duration of the source-time Gaussian in s (0 = delta)

  /**
   * @brief Compute the horizontal ray parameter.
   *
   * \f$ p = \frac{\sin\theta}{V_\text{hs}} \f$
   *
   * @param halfspace_velocity P- or S-wave velocity of the half-space in m/s,
   *        matching the wave type of this incident wave.
   * @return horizontal ray parameter in s/m
   */
  type_real ray_parameter(type_real halfspace_velocity) const;
};

} // namespace fk
} // namespace injection
} // namespace specfem
