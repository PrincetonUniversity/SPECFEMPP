#pragma once

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Geocentric spherical coordinates (physics/SPECFEM convention).
 *
 * @f$ \theta @f$ is colatitude (0 at North Pole, @f$ \pi @f$ at South Pole).
 */
struct geocentric_coordinates {
  double r;     ///< meters (radius from Earth center)
  double theta; ///< radians (colatitude: 0 at North Pole, pi at South Pole)
  double phi;   ///< radians (longitude: 0 at prime meridian)
};

} // namespace coordinate_systems
} // namespace specfem
